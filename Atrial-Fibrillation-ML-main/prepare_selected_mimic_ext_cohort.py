#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a build-ready cohort CSV from selected MIMIC-III-Ext-PPG segments.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("artifacts/mimic_ext_subject_selection/selected_segments.csv"),
        help="Selected segment manifest CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/mimic_ext_subject_selection/selected_segments_build_cohort.csv"),
        help="Output cohort CSV used by build_mimic_ext_ppg_dataset.py.",
    )
    parser.add_argument(
        "--train-fold",
        type=int,
        default=0,
        help="Metadata fold id assigned to the selected subset so it is train-only in metadata_folds experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    required = {"folder_path", "signal_file_name", "record_id", "label", "label_name", "subject_id"}
    missing = required - set(df.columns)
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise ValueError(f"Input CSV missing required columns: {missing_display}")

    cohort = df.copy()
    cohort["event_rhythm"] = cohort["label_name"].astype(str)
    cohort["strat_fold"] = int(args.train_fold)

    ordered_columns = []
    for column in (
        "record_id",
        "event_rhythm",
        "label",
        "strat_fold",
        "folder_path",
        "signal_file_name",
        "subject_id",
    ):
        if column in cohort.columns:
            ordered_columns.append(column)
    ordered_columns.extend([column for column in cohort.columns if column not in ordered_columns])
    cohort = cohort.loc[:, ordered_columns]

    cohort.to_csv(args.output_csv, index=False)
    print(
        {
            "rows": int(cohort.shape[0]),
            "subjects": int(cohort["subject_id"].nunique()),
            "train_fold": int(args.train_fold),
            "rhythms": cohort["event_rhythm"].value_counts().to_dict(),
            "output_csv": str(args.output_csv),
        }
    )


if __name__ == "__main__":
    main()
