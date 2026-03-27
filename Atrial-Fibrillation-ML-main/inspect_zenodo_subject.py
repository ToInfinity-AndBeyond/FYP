from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zenodo_longterm_loader import (
    build_subject_window_labels,
    load_zenodo_ecg_mat,
    load_zenodo_ppg_mat,
    summarize_ppg_segments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a Zenodo long-term AF subject pair (ECG + PPG), "
            "summarize PPG segments, and export ECG-derived 30-second window labels."
        )
    )
    parser.add_argument(
        "--ecg-mat",
        type=Path,
        required=True,
        help="Path to a subject ECG MAT file such as 001_ECG.mat",
    )
    parser.add_argument(
        "--ppg-mat",
        type=Path,
        required=True,
        help="Path to a subject PPG MAT file such as 001_PPG.mat",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "zenodo_inspect",
        help="Directory where inspection CSV files will be written",
    )
    parser.add_argument(
        "--window-length-sec",
        type=float,
        default=30.0,
        help="PPG window length in seconds",
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=30.0,
        help="PPG window stride in seconds",
    )
    parser.add_argument(
        "--positive-fraction-threshold",
        type=float,
        default=0.5,
        help="Minimum AF fraction inside a window to assign an AF label",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ecg_record = load_zenodo_ecg_mat(args.ecg_mat)
    ppg_segments, _ = load_zenodo_ppg_mat(args.ppg_mat, ecg_record=ecg_record)
    ppg_summary = summarize_ppg_segments(ppg_segments)
    window_labels = build_subject_window_labels(
        ecg_record,
        ppg_segments,
        window_length_sec=args.window_length_sec,
        stride_sec=args.stride_sec,
        positive_fraction_threshold=args.positive_fraction_threshold,
    )

    subject_id = args.ecg_mat.stem.replace("_ECG", "")
    output_dir = args.output_dir / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)

    ppg_summary_path = output_dir / f"{subject_id}_ppg_segment_summary.csv"
    window_label_path = output_dir / f"{subject_id}_window_labels.csv"
    ppg_summary.to_csv(ppg_summary_path, index=False)
    window_labels.to_csv(window_label_path, index=False)

    ecg_duration_hours = ecg_record.ecg.size / ecg_record.sample_rate_hz / 3600.0
    ppg_duration_hours = ppg_summary["duration_sec"].sum() / 3600.0 if not ppg_summary.empty else 0.0
    total_windows = int(window_labels.shape[0])
    labeled_windows = int(window_labels["use_for_training"].sum()) if not window_labels.empty else 0
    af_windows = int(window_labels["label"].fillna(0).sum()) if not window_labels.empty else 0
    ambiguous_windows = total_windows - labeled_windows
    af_beats = int((ecg_record.af_annotation > 0.5).sum())
    beat_count = int(ecg_record.af_annotation.size)

    print(f"subject_id: {subject_id}")
    print(
        "ECG summary: "
        f"duration_hours={ecg_duration_hours:.2f}, sample_rate_hz={ecg_record.sample_rate_hz:.1f}, "
        f"beats={beat_count}, af_beats={af_beats}, af_beat_fraction={af_beats / max(beat_count, 1):.4f}"
    )
    print(
        "PPG summary: "
        f"segments={int(ppg_summary.shape[0])}, total_duration_hours={ppg_duration_hours:.2f}, "
        f"sample_rate_hz={ppg_segments[0].ppg_sample_rate_hz if ppg_segments else 0:.1f}"
    )
    if not ppg_summary.empty:
        first_rows = ppg_summary.head(5).to_dict(orient="records")
        print(f"PPG first_segments: {first_rows}")
    print(
        "Window labels: "
        f"total={total_windows}, labeled={labeled_windows}, af={af_windows}, ambiguous={ambiguous_windows}"
    )
    print(f"ppg_summary_csv: {ppg_summary_path}")
    print(f"window_label_csv: {window_label_path}")


if __name__ == "__main__":
    main()
