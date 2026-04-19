#!/usr/bin/env python3
"""Design an AF-vs-SR cohort from MIMIC-III-Ext-PPG metadata.csv."""

from __future__ import annotations

import argparse
import ast
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


DEFAULT_METADATA = Path("/vol/bitbucket/mc1920/FYP/metadata.csv")
DEFAULT_OUTPUT = Path("artifacts/mimic_ext_ppg_cohort")
DEFAULT_COLUMNS = [
    "record_id",
    "event_id",
    "segment_id",
    "signal_file_name",
    "folder_path",
    "patient",
    "subject_id",
    "hadm_id",
    "icustay_id",
    "event_rhythm",
    "strat_fold",
    "vector_10s_pleth_sqi",
    "vector_10s_ecg_sqi",
    "vector_10s_abp_sqi",
    "resp_sqi",
    "median_30s_hr",
    "median_30s_rr",
    "median_30s_sbp",
    "median_30s_dbp",
    "start_segment",
    "event_time",
    "age",
    "gender",
    "height",
    "weight",
    "ethnicity",
    "clinical_information_system",
    "icd9",
    "icd10_truncated",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an AF-vs-SR cohort design from metadata.csv.")
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument(
        "--subset-prefix",
        type=str,
        default=None,
        help="Optional folder_path prefix filter such as 'p00/'",
    )
    parser.add_argument(
        "--positive-rhythms",
        nargs="+",
        default=["AF"],
        help="Rhythm labels to map to the positive class.",
    )
    parser.add_argument(
        "--negative-rhythms",
        nargs="+",
        default=["SR"],
        help="Rhythm labels to map to the negative class.",
    )
    parser.add_argument(
        "--min-good-pleth-segments",
        type=int,
        default=2,
        help="Minimum number of 10-second PLETH SQI entries equal to 1.",
    )
    parser.add_argument(
        "--min-pleth-good-fraction",
        type=float,
        default=2.0 / 3.0,
        help="Minimum fraction of good PLETH SQI entries equal to 1.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Optional cap per class after filtering, useful for balanced smoke datasets.",
    )
    return parser.parse_args()


def clean_list_like_text(value: str) -> str:
    cleaned = re.sub(r"np\.float64\(([^()]*)\)", r"\1", value)
    cleaned = re.sub(r"\bnan\b", "None", cleaned)
    return cleaned


def parse_list_cell(value) -> list[object]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(clean_list_like_text(text))
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if item is not None and not (isinstance(item, float) and math.isnan(item))]
    return [parsed]


def compute_good_fraction(value) -> tuple[int, int, float]:
    items = parse_list_cell(value)
    total = len(items)
    if total == 0:
        return 0, 0, 0.0
    good = sum(1 for item in items if str(item) == "1")
    return good, total, good / total


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    positive = set(args.positive_rhythms)
    negative = set(args.negative_rhythms)
    allowed = positive | negative

    rows = []
    raw_rhythm_counts = Counter()
    filtered_out = Counter()
    subject_to_folds: dict[str, set[str]] = defaultdict(set)
    processed_chunks = 0
    total_rows = 0

    reader = pd.read_csv(args.metadata_csv, usecols=DEFAULT_COLUMNS, chunksize=args.chunksize, low_memory=False)
    for chunk in reader:
        processed_chunks += 1
        total_rows += len(chunk)

        rhythm = chunk["event_rhythm"].fillna("MISSING").astype(str)
        raw_rhythm_counts.update(rhythm.value_counts().to_dict())

        if args.subset_prefix is not None:
            chunk = chunk.loc[chunk["folder_path"].astype(str).str.startswith(args.subset_prefix)].copy()
            rhythm = chunk["event_rhythm"].fillna("MISSING").astype(str)

        chunk = chunk.loc[rhythm.isin(allowed)].copy()
        if chunk.empty:
            if args.max_chunks is not None and processed_chunks >= args.max_chunks:
                break
            continue

        pleth_stats = chunk["vector_10s_pleth_sqi"].map(compute_good_fraction)
        chunk["pleth_good_count"] = [x[0] for x in pleth_stats]
        chunk["pleth_sqi_count"] = [x[1] for x in pleth_stats]
        chunk["pleth_good_fraction"] = [x[2] for x in pleth_stats]
        chunk["accepted_for_ppg"] = (
            (chunk["pleth_good_count"] >= args.min_good_pleth_segments)
            & (chunk["pleth_good_fraction"] >= args.min_pleth_good_fraction)
        )

        filtered_out["bad_pleth_sqi"] += int((~chunk["accepted_for_ppg"]).sum())
        chunk = chunk.loc[chunk["accepted_for_ppg"]].copy()
        if chunk.empty:
            if args.max_chunks is not None and processed_chunks >= args.max_chunks:
                break
            continue

        chunk["label"] = chunk["event_rhythm"].map(lambda value: 1 if value in positive else 0)
        chunk["label_name"] = chunk["label"].map({1: "AF", 0: "SR"})
        chunk["wfdb_record_path"] = chunk.apply(
            lambda row: f"{row['folder_path']}/{row['signal_file_name']}",
            axis=1,
        )

        for subject_id, fold in chunk[["subject_id", "strat_fold"]].dropna().astype(str).itertuples(index=False):
            subject_to_folds[subject_id].add(fold)

        rows.append(chunk)
        if args.max_chunks is not None and processed_chunks >= args.max_chunks:
            break

    if not rows:
        raise RuntimeError("No cohort rows survived filtering. Relax the rhythm or SQI constraints.")

    cohort_df = pd.concat(rows, ignore_index=True)

    if args.max_samples_per_class is not None:
        balanced_frames = []
        for label, group in cohort_df.groupby("label", sort=True):
            balanced_frames.append(group.head(args.max_samples_per_class))
        cohort_df = pd.concat(balanced_frames, ignore_index=True)

    cohort_csv = args.output_dir / "af_sr_cohort.csv"
    cohort_df.to_csv(cohort_csv, index=False)

    class_summary = (
        cohort_df.groupby(["label", "label_name"])
        .agg(
            segments=("label", "size"),
            subjects=("subject_id", "nunique"),
            median_age=("age", "median"),
            median_hr=("median_30s_hr", "median"),
        )
        .reset_index()
    )
    class_summary.to_csv(args.output_dir / "class_summary.csv", index=False)

    fold_summary = (
        cohort_df.groupby(["strat_fold", "label_name"])
        .agg(segments=("label", "size"), subjects=("subject_id", "nunique"))
        .reset_index()
        .sort_values(["strat_fold", "label_name"])
    )
    fold_summary.to_csv(args.output_dir / "fold_summary.csv", index=False)

    subject_fold_conflicts = sorted(
        (subject_id, sorted(folds))
        for subject_id, folds in subject_to_folds.items()
        if len(folds) > 1
    )
    leakage_df = pd.DataFrame(subject_fold_conflicts, columns=["subject_id", "folds"])
    leakage_df.to_csv(args.output_dir / "subject_fold_conflicts.csv", index=False)

    design_lines = [
        "Recommended AF-vs-SR cohort design",
        f"Source metadata: {args.metadata_csv}",
        f"Processed chunks: {processed_chunks}",
        f"Rows scanned: {total_rows:,}",
        "",
        "Task definition",
        f"- Positive class: {sorted(positive)}",
        f"- Negative class: {sorted(negative)}",
        "- Modeling target: binary AF detection from 30-second PPG segments",
        "",
        "Quality filter",
        f"- Keep rows where vector_10s_pleth_sqi has at least {args.min_good_pleth_segments} good 10-second subsegments",
        f"- Keep rows where pleth good fraction is at least {args.min_pleth_good_fraction:.2f}",
        "- Treat PLETH SQI code 1 as usable/good",
        "",
        "Split policy",
        "- Preserve provided strat_fold values from metadata",
        "- Suggested fixed split for first experiments: train folds 0-7, val fold 8, test fold 9",
        "- Alternative: full 10-fold cross-validation using strat_fold",
        "",
        "Cohort size after filtering",
    ]

    label_counts = cohort_df["label_name"].value_counts()
    for label_name, count in label_counts.items():
        design_lines.append(f"- {label_name}: {count:,} segments")
    design_lines.append(f"- Total subjects: {cohort_df['subject_id'].nunique():,}")
    design_lines.append(f"- Total segments: {len(cohort_df):,}")
    design_lines.append("")
    design_lines.append("Filtering losses")
    for name, count in filtered_out.items():
        design_lines.append(f"- {name}: {count:,}")
    design_lines.append("")
    design_lines.append("Top raw rhythm counts before AF/SR filtering")
    for rhythm_name, count in raw_rhythm_counts.most_common(10):
        design_lines.append(f"- {rhythm_name}: {count:,}")
    design_lines.append("")
    design_lines.append("Leakage check")
    design_lines.append(f"- Subjects assigned to multiple folds: {len(subject_fold_conflicts):,}")
    if subject_fold_conflicts[:5]:
        for subject_id, folds in subject_fold_conflicts[:5]:
            design_lines.append(f"- Example conflict: subject {subject_id} in folds {folds}")

    (args.output_dir / "design_report.txt").write_text("\n".join(design_lines) + "\n", encoding="utf-8")
    print("\n".join(design_lines))
    print(f"\nSaved cohort CSV to: {cohort_csv}")


if __name__ == "__main__":
    main()
