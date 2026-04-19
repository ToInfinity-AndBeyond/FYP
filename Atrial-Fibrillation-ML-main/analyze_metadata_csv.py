#!/usr/bin/env python3
"""Summarize the MIMIC-III-Ext-PPG metadata CSV.

This script is designed for the PhysioNet metadata format used by:
https://www.physionet.org/content/mimic-iii-ext-ppg/1.1.0/

It produces:
1. A human-readable text report.
2. CSV summaries for missingness, categorical columns, numeric columns, and
   parsed list-like columns such as SQI vectors and ICD code lists.
3. A few simple plots for quick inspection.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


ID_COLUMNS = ["record_id", "event_id", "segment_id", "signal_file_name", "patient", "folder_path"]
TIME_COLUMNS = ["start_segment", "start_record", "event_time"]
CORE_CATEGORICAL_COLUMNS = [
    "event_rhythm",
    "clinical_information_system",
    "gender",
    "ethnicity",
    "strat_fold",
]
NUMERIC_COLUMNS = [
    "median_30s_sbp",
    "iqr_30s_sbp",
    "median_30s_dbp",
    "iqr_30s_dbp",
    "median_30s_hr",
    "iqr_30s_hr",
    "median_30s_rr",
    "iqr_30s_rr",
    "subject_id",
    "hadm_id",
    "icustay_id",
    "age",
    "weight",
    "height",
]
LIST_COLUMNS = [
    "vector_10s_median_sbp",
    "vector_10s_iqr_sbp",
    "vector_10s_median_dbp",
    "vector_10s_iqr_dbp",
    "vector_10s_hr",
    "vector_10s_pleth_sqi",
    "vector_10s_ecg_sqi",
    "vector_10s_abp_sqi",
    "icd9",
    "icd10_truncated",
]
SQI_COLUMNS = ["vector_10s_pleth_sqi", "vector_10s_ecg_sqi", "vector_10s_abp_sqi", "resp_sqi"]
TOP_K = 20


@dataclass
class RunningNumericStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: pd.Series) -> None:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return
        self.count += int(numeric.shape[0])
        self.total += float(numeric.sum())
        self.total_sq += float((numeric ** 2).sum())
        self.minimum = min(self.minimum, float(numeric.min()))
        self.maximum = max(self.maximum, float(numeric.max()))

    def as_dict(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "non_null_count": 0,
                "mean": math.nan,
                "std": math.nan,
                "min": math.nan,
                "max": math.nan,
            }
        mean = self.total / self.count
        variance = max((self.total_sq / self.count) - (mean ** 2), 0.0)
        return {
            "non_null_count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the MIMIC-III-Ext-PPG metadata CSV.")
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("/vol/bitbucket/mc1920/FYP/metadata.csv"),
        help="Path to metadata.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/metadata_analysis"),
        help="Directory to write reports, CSVs, and plots",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Rows per chunk while streaming the CSV",
    )
    parser.add_argument(
        "--max-category-values",
        type=int,
        default=TOP_K,
        help="Number of top values to keep per categorical or list column",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional limit on the number of streamed chunks, useful for a quick smoke test",
    )
    return parser.parse_args()


def clean_list_like_text(value: str) -> str:
    cleaned = re.sub(r"np\.float64\(([^()]*)\)", r"\1", value)
    cleaned = re.sub(r"\bnan\b", "None", cleaned)
    return cleaned


def parse_list_cell(value) -> List[object]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(clean_list_like_text(text))
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if item is not None and not (isinstance(item, float) and math.isnan(item))]
    return [parsed]


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_bar_plot(series: pd.Series, title: str, ylabel: str, output_path: Path) -> None:
    if series.empty:
        return
    plt.figure(figsize=(10, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    if not args.metadata_csv.exists():
        raise FileNotFoundError(f"metadata CSV not found: {args.metadata_csv}")

    header = pd.read_csv(args.metadata_csv, nrows=0)
    columns = header.columns.tolist()

    available_categorical = [col for col in CORE_CATEGORICAL_COLUMNS if col in columns]
    available_numeric = [col for col in NUMERIC_COLUMNS if col in columns]
    available_list_columns = [col for col in LIST_COLUMNS if col in columns]
    available_sqi_columns = [col for col in SQI_COLUMNS if col in columns]

    total_rows = 0
    null_counts = Counter()
    categorical_counts = {col: Counter() for col in available_categorical}
    numeric_stats = {col: RunningNumericStats() for col in available_numeric}
    list_value_counts = {col: Counter() for col in available_list_columns}
    unique_tracker = {
        "record_id": set(),
        "subject_id": set(),
        "hadm_id": set(),
        "icustay_id": set(),
    }
    time_ranges = {}
    sqi_scalar_counts = Counter()
    sqi_vector_counts = defaultdict(Counter)

    reader = pd.read_csv(args.metadata_csv, chunksize=args.chunksize, low_memory=False)
    processed_chunks = 0
    for chunk in reader:
        processed_chunks += 1
        total_rows += len(chunk)
        for col in columns:
            null_counts[col] += int(chunk[col].isna().sum())

        for key in unique_tracker:
            if key in chunk.columns:
                unique_tracker[key].update(chunk[key].dropna().astype(str).unique().tolist())

        for col in available_categorical:
            categorical_counts[col].update(chunk[col].fillna("MISSING").astype(str).value_counts().to_dict())

        for col in available_numeric:
            numeric_stats[col].update(chunk[col])

        for col in TIME_COLUMNS:
            if col not in chunk.columns:
                continue
            parsed = pd.to_datetime(chunk[col], errors="coerce")
            if parsed.notna().any():
                current_min = parsed.min()
                current_max = parsed.max()
                if col not in time_ranges:
                    time_ranges[col] = [current_min, current_max]
                else:
                    time_ranges[col][0] = min(time_ranges[col][0], current_min)
                    time_ranges[col][1] = max(time_ranges[col][1], current_max)

        for col in available_list_columns:
            parsed_series = chunk[col].map(parse_list_cell)
            for items in parsed_series:
                list_value_counts[col].update(str(item) for item in items)

        for col in available_sqi_columns:
            if col == "resp_sqi":
                sqi_scalar_counts.update(chunk[col].dropna().astype(int).astype(str).tolist())
                continue
            parsed_series = chunk[col].map(parse_list_cell)
            for items in parsed_series:
                sqi_vector_counts[col].update(str(item) for item in items)

        if args.max_chunks is not None and processed_chunks >= args.max_chunks:
            break

    missing_df = (
        pd.DataFrame(
            {
                "column": columns,
                "missing_count": [null_counts[col] for col in columns],
                "missing_fraction": [null_counts[col] / total_rows for col in columns],
            }
        )
        .sort_values(["missing_fraction", "missing_count"], ascending=False)
        .reset_index(drop=True)
    )
    missing_df.to_csv(args.output_dir / "missingness_summary.csv", index=False)

    numeric_df = pd.DataFrame(
        [{"column": col, **numeric_stats[col].as_dict()} for col in available_numeric]
    ).sort_values("column")
    numeric_df.to_csv(args.output_dir / "numeric_summary.csv", index=False)

    categorical_frames = []
    for col, counts in categorical_counts.items():
        top_counts = counts.most_common(args.max_category_values)
        categorical_frames.append(
            pd.DataFrame(top_counts, columns=["value", "count"]).assign(column=col)
        )
    categorical_df = pd.concat(categorical_frames, ignore_index=True) if categorical_frames else pd.DataFrame()
    if not categorical_df.empty:
        categorical_df = categorical_df[["column", "value", "count"]]
        categorical_df.to_csv(args.output_dir / "categorical_top_values.csv", index=False)

    list_frames = []
    for col, counts in list_value_counts.items():
        top_counts = counts.most_common(args.max_category_values)
        list_frames.append(
            pd.DataFrame(top_counts, columns=["value", "count"]).assign(column=col)
        )
    list_df = pd.concat(list_frames, ignore_index=True) if list_frames else pd.DataFrame()
    if not list_df.empty:
        list_df = list_df[["column", "value", "count"]]
        list_df.to_csv(args.output_dir / "list_column_top_values.csv", index=False)

    rhythm_series = pd.Series(dict(categorical_counts.get("event_rhythm", {}))).sort_values(ascending=False).head(15)
    save_bar_plot(
        rhythm_series,
        title="Top Heart Rhythm Labels",
        ylabel="Segment count",
        output_path=args.output_dir / "event_rhythm_top15.png",
    )

    missing_top_series = missing_df.head(15).set_index("column")["missing_fraction"]
    save_bar_plot(
        missing_top_series,
        title="Columns With Highest Missing Fraction",
        ylabel="Missing fraction",
        output_path=args.output_dir / "missingness_top15.png",
    )

    fold_series = pd.Series(dict(categorical_counts.get("strat_fold", {}))).sort_index()
    save_bar_plot(
        fold_series,
        title="Segments Per Stratification Fold",
        ylabel="Segment count",
        output_path=args.output_dir / "strat_fold_counts.png",
    )

    report_lines = [
        "MIMIC-III-Ext-PPG metadata.csv analysis",
        f"Source CSV: {args.metadata_csv}",
        f"Total rows (30-second segments): {total_rows:,}",
        f"Processed chunks: {processed_chunks:,}",
        f"Columns: {len(columns)}",
        "",
        "Identifier coverage",
        f"- Unique records: {len(unique_tracker['record_id']):,}",
        f"- Unique subjects: {len(unique_tracker['subject_id']):,}",
        f"- Unique hospital admissions: {len(unique_tracker['hadm_id']):,}",
        f"- Unique ICU stays: {len(unique_tracker['icustay_id']):,}",
        "",
        "Detected column groups",
        f"- ID columns: {', '.join([c for c in ID_COLUMNS if c in columns])}",
        f"- Time columns: {', '.join([c for c in TIME_COLUMNS if c in columns])}",
        f"- Numeric columns: {', '.join(available_numeric)}",
        f"- List-like columns: {', '.join(available_list_columns)}",
        "",
        "Top rhythm labels",
    ]
    for label, count in categorical_counts.get("event_rhythm", Counter()).most_common(10):
        report_lines.append(f"- {label}: {count:,}")

    report_lines.extend(["", "Top missing columns"])
    for _, row in missing_df.head(10).iterrows():
        report_lines.append(
            f"- {row['column']}: {int(row['missing_count']):,} missing ({row['missing_fraction']:.2%})"
        )

    report_lines.extend(["", "Time ranges"])
    for col, (minimum, maximum) in time_ranges.items():
        report_lines.append(f"- {col}: {minimum} -> {maximum}")

    report_lines.extend(["", "SQI code overview"])
    for col, counts in sqi_vector_counts.items():
        compact = ", ".join(f"{code}:{count:,}" for code, count in counts.most_common(8))
        report_lines.append(f"- {col}: {compact}")
    if sqi_scalar_counts:
        compact = ", ".join(f"{code}:{count:,}" for code, count in sqi_scalar_counts.most_common(8))
        report_lines.append(f"- resp_sqi: {compact}")

    (args.output_dir / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
