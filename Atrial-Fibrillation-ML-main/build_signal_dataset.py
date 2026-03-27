from __future__ import annotations

import argparse
from pathlib import Path

from signal_pipeline import (
    apply_quality_overrides,
    build_dataset_from_csvs,
    default_ecg_config,
    default_ppg_config,
    load_quality_overrides,
    load_window_label_table,
    save_dataset_bundle,
)


def discover_csvs(dataset_root: Path) -> list[Path]:
    patterns = [
        "mimic_perform_af_csv/*_data.csv",
        "mimic_perform_non_af_csv/*_data.csv",
    ]
    csv_paths: list[Path] = []
    for pattern in patterns:
        csv_paths.extend(sorted(dataset_root.glob(pattern)))
    return sorted(csv_paths)


def build_for_signal_type(
    dataset_root: Path,
    output_dir: Path,
    signal_type: str,
    limit_files: int | None,
    window_label_csv: Path | None,
    fallback_to_path_labels: bool,
    quality_json: Path | None,
) -> None:
    csv_paths = discover_csvs(dataset_root)
    if limit_files is not None:
        csv_paths = csv_paths[:limit_files]

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {dataset_root}")

    if signal_type == "ppg":
        config = default_ppg_config()
    elif signal_type == "ecg":
        config = default_ecg_config()
    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")
    config = apply_quality_overrides(config, load_quality_overrides(quality_json))

    window_label_table = load_window_label_table(window_label_csv) if window_label_csv is not None else None
    summary_df, segments = build_dataset_from_csvs(
        csv_paths,
        config,
        window_label_table=window_label_table,
        fallback_to_path_labels=fallback_to_path_labels,
    )
    saved = save_dataset_bundle(summary_df, segments, output_dir / signal_type, config)

    accepted_count = int(summary_df["accepted"].sum()) if not summary_df.empty else 0
    total_count = int(summary_df.shape[0])
    af_count = int(summary_df["label"].sum()) if not summary_df.empty else 0

    print(f"[{signal_type}] records processed: {len(csv_paths)}")
    print(f"[{signal_type}] segments: {total_count}")
    print(f"[{signal_type}] accepted: {accepted_count}")
    print(f"[{signal_type}] AF segments: {af_count}")
    print(f"[{signal_type}] outputs:")
    for name, path in saved.items():
        print(f"  - {name}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AF-ready signal-processing datasets from MIMIC CSV files.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root that contains mimic_perform_af_csv and mimic_perform_non_af_csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "signal_pipeline",
        help="Directory where summary CSV/NPZ outputs will be written",
    )
    parser.add_argument(
        "--signal-type",
        choices=["ppg", "ecg", "both"],
        default="both",
        help="Which modality to process",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional limit for smoke tests",
    )
    parser.add_argument(
        "--window-label-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with per-window labels. Expected columns: record_id, segment_index, label. "
            "If provided, unlabeled windows are skipped unless --fallback-to-path-labels is set."
        ),
    )
    parser.add_argument(
        "--fallback-to-path-labels",
        action="store_true",
        help="When using --window-label-csv, fall back to AF/non-AF folder labels for windows not present in the CSV.",
    )
    parser.add_argument(
        "--quality-json",
        type=Path,
        default=None,
        help="Optional JSON file containing tuned quality gate overrides.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    signal_types = ["ppg", "ecg"] if args.signal_type == "both" else [args.signal_type]
    for signal_type in signal_types:
        build_for_signal_type(
            args.dataset_root,
            args.output_dir,
            signal_type,
            args.limit_files,
            args.window_label_csv,
            args.fallback_to_path_labels,
            args.quality_json,
        )


if __name__ == "__main__":
    main()
