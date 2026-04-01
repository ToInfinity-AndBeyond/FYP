from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
    "peak_count",
    "heart_band_energy_ratio",
    "signal_skewness",
    "template_correlation",
    "acc_variance",
    "estimated_hr_bpm",
    "quality_score",
    "ibi_count",
    "mean_ibi_ms",
    "median_ibi_ms",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50",
    "mean_hr_bpm",
    "std_hr_bpm",
    "cv_ibi",
    "sample_entropy",
    "signal_spectral_entropy",
]

GROUP_CANDIDATES = ["case_id", "subject_id", "record_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Neuton-ready tabular train/holdout CSVs from accepted segment summary files."
    )
    parser.add_argument("--summary-path", type=Path, required=True, help="Accepted segment summary CSV path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for exported CSVs and metadata")
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="Fraction of groups reserved for holdout validation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=DEFAULT_FEATURE_COLUMNS,
        help="Candidate numeric feature columns to export",
    )
    parser.add_argument(
        "--group-column",
        default=None,
        help="Optional explicit group column for patient-wise split; defaults to case_id/subject_id/record_id",
    )
    parser.add_argument(
        "--prefix",
        default="neuton",
        help="Filename prefix for exported train/holdout CSVs",
    )
    return parser.parse_args()


def choose_group_column(summary_df: pd.DataFrame, explicit_group_column: str | None) -> str:
    if explicit_group_column:
        if explicit_group_column not in summary_df.columns:
            raise ValueError(f"Group column '{explicit_group_column}' was not found in the summary CSV.")
        return explicit_group_column

    for column in GROUP_CANDIDATES:
        if column in summary_df.columns:
            return column
    raise ValueError("No suitable group column found. Expected one of: case_id, subject_id, record_id.")


def choose_feature_columns(summary_df: pd.DataFrame, requested_columns: list[str]) -> tuple[list[str], list[str]]:
    present_columns = [column for column in requested_columns if column in summary_df.columns]
    if not present_columns:
        raise ValueError("None of the requested feature columns are present in the summary CSV.")

    dropped_columns = []
    selected_columns = []
    for column in present_columns:
        if summary_df[column].isna().all():
            dropped_columns.append(column)
            continue
        selected_columns.append(column)

    if not selected_columns:
        raise ValueError("All requested feature columns are empty.")
    return selected_columns, dropped_columns


def stratified_group_split(
    summary_df: pd.DataFrame,
    group_column: str,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[object], list[object]]:
    grouped = summary_df.groupby(group_column, as_index=False).agg(label=("label", "max"))
    rng = np.random.default_rng(seed)

    train_groups: list[object] = []
    holdout_groups: list[object] = []

    for label_value in [0, 1]:
        groups = grouped.loc[grouped["label"] == label_value, group_column].to_numpy()
        groups = groups.copy()
        rng.shuffle(groups)
        holdout_count = int(round(groups.shape[0] * holdout_fraction))
        if groups.shape[0] >= 2:
            holdout_count = min(max(1, holdout_count), groups.shape[0] - 1)
        holdout_groups.extend(groups[:holdout_count].tolist())
        train_groups.extend(groups[holdout_count:].tolist())

    return train_groups, holdout_groups


def fill_with_train_medians(
    summary_df: pd.DataFrame,
    train_mask: np.ndarray,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    medians = summary_df.loc[train_mask, feature_columns].median(axis=0)
    medians = medians.fillna(0.0)

    filled = summary_df.copy()
    filled[feature_columns] = filled[feature_columns].fillna(medians)
    filled[feature_columns] = filled[feature_columns].replace([np.inf, -np.inf], 0.0)
    median_map = {column: float(medians[column]) for column in feature_columns}
    return filled, median_map


def export_split(
    summary_df: pd.DataFrame,
    feature_columns: list[str],
    mask: np.ndarray,
    target_path: Path,
) -> None:
    export_df = summary_df.loc[mask, feature_columns + ["label"]].copy()
    export_df.columns = [column.lower() for column in export_df.columns]
    export_df.to_csv(target_path, index=False, encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_df = pd.read_csv(args.summary_path)
    if "label" not in summary_df.columns:
        raise ValueError("Summary CSV must contain a 'label' column.")

    group_column = choose_group_column(summary_df, args.group_column)
    feature_columns, dropped_columns = choose_feature_columns(summary_df, args.feature_columns)

    train_groups, holdout_groups = stratified_group_split(
        summary_df=summary_df,
        group_column=group_column,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )

    train_mask = summary_df[group_column].isin(train_groups).to_numpy()
    holdout_mask = summary_df[group_column].isin(holdout_groups).to_numpy()
    if not train_mask.any() or not holdout_mask.any():
        raise ValueError("Train/holdout split failed; try adjusting the holdout fraction.")

    filled_df, median_map = fill_with_train_medians(summary_df, train_mask, feature_columns)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.prefix}_train.csv"
    holdout_path = args.output_dir / f"{args.prefix}_holdout.csv"
    metadata_path = args.output_dir / f"{args.prefix}_metadata.json"

    export_split(filled_df, feature_columns, train_mask, train_path)
    export_split(filled_df, feature_columns, holdout_mask, holdout_path)

    metadata = {
        "summary_path": str(args.summary_path),
        "group_column": group_column,
        "feature_columns": feature_columns,
        "dropped_all_nan_columns": dropped_columns,
        "holdout_fraction": args.holdout_fraction,
        "seed": args.seed,
        "train_rows": int(train_mask.sum()),
        "holdout_rows": int(holdout_mask.sum()),
        "train_groups": int(len(train_groups)),
        "holdout_groups": int(len(holdout_groups)),
        "train_positive_rows": int(filled_df.loc[train_mask, "label"].sum()),
        "holdout_positive_rows": int(filled_df.loc[holdout_mask, "label"].sum()),
        "train_fill_medians": median_map,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("export complete:")
    print(f"  train_csv: {train_path}")
    print(f"  holdout_csv: {holdout_path}")
    print(f"  metadata_json: {metadata_path}")
    print(
        json.dumps(
            {
                "group_column": group_column,
                "feature_count": len(feature_columns),
                "dropped_all_nan_columns": dropped_columns,
                "train_rows": int(train_mask.sum()),
                "holdout_rows": int(holdout_mask.sum()),
                "train_groups": int(len(train_groups)),
                "holdout_groups": int(len(holdout_groups)),
            }
        )
    )


if __name__ == "__main__":
    main()
