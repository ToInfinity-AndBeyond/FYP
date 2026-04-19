from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from signal_pipeline import (
    apply_quality_overrides,
    default_ecg_config,
    default_ppg_config,
    load_quality_overrides,
    process_dataframe,
    save_dataset_bundle,
)

try:
    import wfdb
except ImportError:  # pragma: no cover - runtime dependency check
    wfdb = None


METADATA_COLUMNS = [
    "record_id",
    "event_id",
    "segment_id",
    "signal_file_name",
    "patient",
    "folder_path",
    "start_segment",
    "start_record",
    "event_time",
    "event_rhythm",
    "vector_10s_pleth_sqi",
    "vector_10s_ecg_sqi",
    "vector_10s_abp_sqi",
    "resp_sqi",
    "subject_id",
    "hadm_id",
    "icustay_id",
    "clinical_information_system",
    "age",
    "weight",
    "height",
    "gender",
    "ethnicity",
    "icd9",
    "icd10_truncated",
    "strat_fold",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PPG/ECG training bundles from MIMIC-III-Ext-PPG WFDB files.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/vol/bitbucket/mc1920/FYP/1.1.0"),
        help="Root containing p00, p01, ..., metadata.csv",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("/vol/bitbucket/mc1920/FYP/metadata.csv"),
        help="Path to metadata.csv",
    )
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=None,
        help="Optional prefiltered cohort CSV. If provided, metadata.csv filtering is skipped.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/mimic_ext_ppg"),
        help="Where to write summary CSV/NPZ outputs",
    )
    parser.add_argument(
        "--subset-prefix",
        default="p00/",
        help="Keep only metadata rows whose folder_path starts with this prefix",
    )
    parser.add_argument(
        "--strat-folds",
        type=parse_fold_list,
        default=None,
        help="Optional comma-separated strat_fold ids to keep, e.g. 0,1,2",
    )
    parser.add_argument(
        "--signal-type",
        choices=["ppg", "ecg", "both"],
        default="ppg",
        help="Which modality to export into bundle format",
    )
    parser.add_argument(
        "--positive-rhythms",
        nargs="+",
        default=["AF"],
        help="Rhythm labels mapped to class 1",
    )
    parser.add_argument(
        "--negative-rhythms",
        nargs="+",
        default=["SR"],
        help="Rhythm labels mapped to class 0",
    )
    parser.add_argument(
        "--min-good-pleth-segments",
        type=int,
        default=2,
        help="Minimum number of 10-second PLETH SQI windows equal to 1",
    )
    parser.add_argument(
        "--min-pleth-good-fraction",
        type=float,
        default=2.0 / 3.0,
        help="Minimum fraction of good PLETH SQI windows equal to 1",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Optional cap on filtered metadata rows for smoke tests",
    )
    parser.add_argument(
        "--quality-json",
        type=Path,
        default=None,
        help="Optional JSON file with quality gate overrides",
    )
    return parser.parse_args()


def clean_list_like_text(value: str) -> str:
    cleaned = re.sub(r"np\.float64\(([^()]*)\)", r"\1", value)
    cleaned = re.sub(r"\bnan\b", "None", cleaned)
    return cleaned


def parse_fold_list(value: str) -> list[int]:
    folds = []
    for part in str(value).split(","):
        normalized = part.strip()
        if not normalized:
            continue
        folds.append(int(normalized))
    if not folds:
        raise argparse.ArgumentTypeError("Fold list must contain at least one integer.")
    return folds


def parse_list_cell(value: Any) -> list[object]:
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
        output = []
        for item in parsed:
            if item is None:
                continue
            if isinstance(item, float) and math.isnan(item):
                continue
            output.append(item)
        return output
    return [parsed]


def pleth_quality_ok(value: Any, min_good_segments: int, min_good_fraction: float) -> bool:
    items = parse_list_cell(value)
    if not items:
        return False
    good_count = sum(1 for item in items if str(item) == "1")
    return good_count >= min_good_segments and (good_count / len(items)) >= min_good_fraction


def load_filtered_metadata(args: argparse.Namespace) -> pd.DataFrame:
    if args.cohort_csv is not None:
        cohort_df = pd.read_csv(args.cohort_csv, low_memory=False)
        required = {"folder_path", "signal_file_name", "record_id", "event_rhythm", "label"}
        missing = required - set(cohort_df.columns)
        if missing:
            missing_display = ", ".join(sorted(missing))
            raise ValueError(f"--cohort-csv is missing required columns: {missing_display}")
        subset = cohort_df.loc[cohort_df["folder_path"].astype(str).str.startswith(args.subset_prefix)].copy()
        if args.strat_folds is not None:
            if "strat_fold" not in subset.columns:
                raise ValueError("--strat-folds requires a 'strat_fold' column in --cohort-csv.")
            subset = subset.loc[pd.to_numeric(subset["strat_fold"], errors="raise").astype(int).isin(args.strat_folds)].copy()
        if args.max_segments is not None:
            subset = subset.head(args.max_segments).copy()
        return subset

    positive = set(args.positive_rhythms)
    negative = set(args.negative_rhythms)
    allowed = positive | negative
    chunks: list[pd.DataFrame] = []
    kept = 0

    for chunk in pd.read_csv(args.metadata_csv, usecols=METADATA_COLUMNS, chunksize=100_000, low_memory=False):
        mask = chunk["folder_path"].astype(str).str.startswith(args.subset_prefix)
        mask &= chunk["event_rhythm"].fillna("MISSING").astype(str).isin(allowed)
        if not mask.any():
            continue
        subset = chunk.loc[mask].copy()
        subset = subset.loc[
            subset["vector_10s_pleth_sqi"].map(
                lambda value: pleth_quality_ok(value, args.min_good_pleth_segments, args.min_pleth_good_fraction)
            )
        ].copy()
        if subset.empty:
            continue
        subset["label"] = subset["event_rhythm"].map(lambda value: 1 if value in positive else 0)
        if args.strat_folds is not None:
            subset = subset.loc[pd.to_numeric(subset["strat_fold"], errors="raise").astype(int).isin(args.strat_folds)].copy()
            if subset.empty:
                continue
        chunks.append(subset)
        kept += len(subset)
        if args.max_segments is not None and kept >= args.max_segments:
            break

    if not chunks:
        raise RuntimeError("No metadata rows matched the requested subset/rhythm/SQI filters.")

    filtered = pd.concat(chunks, ignore_index=True)
    if args.max_segments is not None:
        filtered = filtered.head(args.max_segments).copy()
    return filtered


def read_wfdb_segment(record_base: Path) -> pd.DataFrame:
    if wfdb is None:
        raise ImportError(
            "wfdb is required for build_mimic_ext_ppg_dataset.py. "
            "Install it in your environment, e.g. inside your project venv."
        )

    record = wfdb.rdrecord(str(record_base))
    if record.p_signal is None:
        raise ValueError(f"No physical signal array found in {record_base}")

    frame = pd.DataFrame(record.p_signal, columns=[str(name) for name in record.sig_name])
    output = pd.DataFrame({"Time": [idx / float(record.fs) for idx in range(frame.shape[0])]})

    signal_map: dict[str, str] = {}
    for column in frame.columns:
        upper = column.upper()
        if upper == "PLETH":
            signal_map[column] = "PPG"
        elif upper == "ABP":
            signal_map[column] = "ABP"
        elif upper == "RESP" or upper == "IMP":
            signal_map[column] = "resp"
        elif upper == "II" or "ECG" in upper:
            signal_map[column] = "ECG"

    if "PPG" not in signal_map.values():
        raise KeyError(f"No PPG/PLETH channel found in {record_base}")

    for source, target in signal_map.items():
        output[target] = frame[source].astype(float)

    return output


def augment_summary_rows(rows: pd.DataFrame, metadata_row: pd.Series) -> pd.DataFrame:
    enriched = rows.copy()
    for column in METADATA_COLUMNS:
        enriched[column] = metadata_row[column] if column in metadata_row.index else pd.NA
    enriched["wfdb_record_path"] = f"{metadata_row['folder_path']}/{metadata_row['signal_file_name']}"
    enriched["metadata_label"] = int(metadata_row["label"])
    return enriched


def resolve_record_base(dataset_root: Path, folder_path: str, signal_file_name: str) -> Path:
    folder = Path(str(folder_path))
    signal_name = str(signal_file_name)
    # In MIMIC-III-Ext-PPG metadata, folder_path usually already includes the
    # record basename: pXX/pXXXXXX/<signal_file_name>
    if folder.name == signal_name:
        return dataset_root / folder
    return dataset_root / folder / signal_name


def build_for_signal_type(
    metadata_df: pd.DataFrame,
    dataset_root: Path,
    output_dir: Path,
    signal_type: str,
    quality_json: Path | None,
) -> None:
    config = default_ppg_config() if signal_type == "ppg" else default_ecg_config()
    config = apply_quality_overrides(config, load_quality_overrides(quality_json))

    summary_blocks: list[pd.DataFrame] = []
    segments: list[Any] = []
    skipped_missing_file = 0
    skipped_missing_channel = 0
    first_missing_file: Path | None = None

    total_rows = len(metadata_df)
    for row_index, metadata_row in enumerate(metadata_df.itertuples(index=False), start=1):
        record_base = resolve_record_base(dataset_root, metadata_row.folder_path, metadata_row.signal_file_name)
        if not record_base.with_suffix(".hea").exists() or not record_base.with_suffix(".dat").exists():
            skipped_missing_file += 1
            if first_missing_file is None:
                first_missing_file = record_base
            continue

        try:
            waveform_df = read_wfdb_segment(record_base)
            label = int(metadata_row.label)
            record_rows, record_segments = process_dataframe(
                dataframe=waveform_df,
                label=label,
                record_id=str(metadata_row.record_id),
                config=config,
            )
        except KeyError:
            skipped_missing_channel += 1
            continue

        if record_rows.empty:
            continue

        record_rows = augment_summary_rows(record_rows, pd.Series(metadata_row._asdict()))
        summary_blocks.append(record_rows)
        segments.append(record_segments)

        if row_index == 1 or row_index == total_rows or row_index % 500 == 0:
            print(
                f"[{signal_type}] processed {row_index}/{total_rows} "
                f"(kept={sum(len(block) for block in summary_blocks)} skipped_missing_file={skipped_missing_file} "
                f"skipped_missing_channel={skipped_missing_channel})",
                flush=True,
            )

    if not summary_blocks:
        raise RuntimeError(
            f"No usable {signal_type} segments were built from the requested metadata subset. "
            f"skipped_missing_file={skipped_missing_file}, "
            f"skipped_missing_channel={skipped_missing_channel}, "
            f"first_missing_file={first_missing_file}"
        )

    summary_df = pd.concat(summary_blocks, ignore_index=True)
    saved = save_dataset_bundle(summary_df, np.concatenate(segments, axis=0), output_dir / signal_type, config)

    accepted_count = int(summary_df["accepted"].sum()) if "accepted" in summary_df.columns else 0
    print(f"[{signal_type}] metadata rows selected: {total_rows}")
    print(f"[{signal_type}] bundle rows: {summary_df.shape[0]}")
    print(f"[{signal_type}] accepted rows: {accepted_count}")
    print(f"[{signal_type}] unique subjects: {summary_df['subject_id'].nunique()}")
    print(f"[{signal_type}] skipped missing file: {skipped_missing_file}")
    print(f"[{signal_type}] skipped missing channel: {skipped_missing_channel}")
    for name, path in saved.items():
        print(f"  - {name}: {path}")


def main() -> None:
    args = parse_args()
    metadata_df = load_filtered_metadata(args)
    print(
        "metadata subset:",
        {
            "rows": int(metadata_df.shape[0]),
            "subjects": int(metadata_df["subject_id"].nunique()),
            "rhythms": metadata_df["event_rhythm"].value_counts().to_dict(),
            "subset_prefix": args.subset_prefix,
        },
        flush=True,
    )

    signal_types = ["ppg", "ecg"] if args.signal_type == "both" else [args.signal_type]
    for signal_type in signal_types:
        build_for_signal_type(
            metadata_df=metadata_df,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            signal_type=signal_type,
            quality_json=args.quality_json,
        )


if __name__ == "__main__":
    main()
