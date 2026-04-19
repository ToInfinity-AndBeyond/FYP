from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from build_zenodo_datasets import resample_1d, resample_acc, slice_ecg_window
from signal_pipeline import default_ecg_config, default_ppg_config, process_segment
from zenodo_longterm_loader import _signal_header_to_dict, load_zenodo_ecg_mat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Zenodo raw vs processed ECG/PPG windows.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/vol/bitbucket/mc1920/zenodo_11242869"),
        help="Root containing 001_ECG.mat / 001_PPG.mat files.",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("/vol/bitbucket/mc1920/zenodo_build_by_subject"),
        help="Root containing subject-wise physio_distill bundles.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/homes/mc1920/FYP/Atrial-Fibrillation-ML-main/artifacts/visuals/zenodo_preprocessing_overview.png"),
        help="Where to save the visualization PNG.",
    )
    parser.add_argument(
        "--window-length-sec",
        type=float,
        default=30.0,
        help="Window length used during preprocessing.",
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=30.0,
        help="Window stride used during preprocessing.",
    )
    return parser.parse_args()


def load_all_accepted_rows(build_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for csv_path in sorted(build_root.glob("*/physio_distill/physio_multimodal_accepted_segment_summary.csv")):
        df = pd.read_csv(csv_path)
        if "joint_accepted" not in df.columns:
            continue
        df = df.loc[(df["joint_accepted"] == True) & (df["start_time_sec"] >= 0)].copy()
        if df.empty:
            continue
        df["summary_csv"] = str(csv_path)
        rows.append(df)
    if not rows:
        raise RuntimeError("No accepted Zenodo rows were found for visualization.")
    return pd.concat(rows, ignore_index=True)


def pick_examples(summary_df: pd.DataFrame) -> list[pd.Series]:
    sr_candidates = summary_df.loc[summary_df["label"] == 0].copy()
    af_candidates = summary_df.loc[summary_df["label"] == 1].copy()
    if sr_candidates.empty or af_candidates.empty:
        raise RuntimeError("Need at least one SR and one AF accepted example to render.")

    sr_candidates = sr_candidates.sort_values(
        ["ppg_quality_score", "ppg_template_correlation", "ppg_sdnn_ms"],
        ascending=[False, False, True],
    )
    af_candidates = af_candidates.sort_values(
        ["ppg_quality_score", "ppg_template_correlation", "ppg_sdnn_ms"],
        ascending=[False, False, False],
    )
    return [sr_candidates.iloc[0], af_candidates.iloc[0]]


def render_example(
    axes_row: np.ndarray,
    row: pd.Series,
    dataset_root: Path,
    window_length_sec: float,
    stride_sec: float,
) -> None:
    subject_id = f"{int(row['subject_id']):03d}"
    ecg_path = dataset_root / f"{subject_id}_ECG.mat"
    ppg_path = dataset_root / f"{subject_id}_PPG.mat"

    ecg_record = load_zenodo_ecg_mat(ecg_path)
    raw_ppg_window, raw_acc_window, ppg_sample_rate_hz, acc_sample_rate_hz = load_ppg_window(
        ppg_path=ppg_path,
        ppg_segment_index=int(row["ppg_segment_index"]),
        window_index=int(row["segment_index"]),
        window_length_sec=window_length_sec,
        stride_sec=stride_sec,
    )

    ppg_config = default_ppg_config()
    ppg_resampled = resample_1d(
        raw_ppg_window,
        source_rate_hz=ppg_sample_rate_hz,
        target_rate_hz=ppg_config.sample_rate_hz,
    )
    acc_resampled = resample_acc(
        raw_acc_window,
        source_rate_hz=acc_sample_rate_hz,
        target_rate_hz=ppg_config.sample_rate_hz,
    )
    processed_ppg = process_segment(ppg_resampled, ppg_config, acc_segment=acc_resampled)

    ecg_config = default_ecg_config(sample_rate_hz=ecg_record.sample_rate_hz)
    raw_ecg_window = slice_ecg_window(
        ecg_record=ecg_record,
        start_time_sec=float(row["start_time_sec"]),
        window_length_sec=window_length_sec,
    )
    processed_ecg = process_segment(raw_ecg_window, ecg_config)

    ppg_time = np.arange(ppg_resampled.size, dtype=float) / ppg_config.sample_rate_hz
    ecg_time = np.arange(raw_ecg_window.size, dtype=float) / ecg_record.sample_rate_hz

    label_name = "AF" if int(row["label"]) == 1 else "SR"
    summary_text = (
        f"{label_name} | subject {subject_id}\n"
        f"PPG q={row['ppg_quality_score']:.3f} | ECG q={row['ecg_quality_score']:.3f}\n"
        f"PPG HR={row['ppg_mean_hr_bpm']:.1f} bpm | SDNN={row['ppg_sdnn_ms']:.1f} ms\n"
        f"Timing corr={row['timing_ibi_corr']:.3f} | delay={row['timing_median_delay_ms']:.1f} ms"
    )

    axes_row[0].plot(ppg_time, ppg_resampled, color="#4C78A8", linewidth=1.0)
    axes_row[0].set_title(f"Raw PPG ({label_name})")
    axes_row[0].set_ylabel("Amplitude")

    axes_row[1].plot(ppg_time, processed_ppg.filtered_signal, color="#F58518", linewidth=1.0)
    axes_row[1].set_title("Bandpass Filtered PPG")

    axes_row[2].plot(ppg_time, processed_ppg.normalized_signal, color="#54A24B", linewidth=1.0)
    peak_times = processed_ppg.peaks.astype(float) / ppg_config.sample_rate_hz
    axes_row[2].scatter(
        peak_times,
        processed_ppg.normalized_signal[processed_ppg.peaks],
        s=10,
        color="#E45756",
        zorder=3,
        label="PPG peaks",
    )
    axes_row[2].legend(loc="upper right", fontsize=8, frameon=False)
    axes_row[2].set_title("Normalized PPG + Peaks")

    axes_row[3].plot(ecg_time, raw_ecg_window, color="#7A5195", linewidth=0.9)
    ecg_peak_times = processed_ecg.peaks.astype(float) / ecg_record.sample_rate_hz
    axes_row[3].scatter(
        ecg_peak_times,
        raw_ecg_window[processed_ecg.peaks],
        s=9,
        color="#EF5675",
        zorder=3,
        label="ECG R-peaks",
    )
    axes_row[3].legend(loc="upper right", fontsize=8, frameon=False)
    axes_row[3].set_title("Aligned Raw ECG + R-peaks")

    freqs, times, spec = sp_signal.spectrogram(
        processed_ppg.normalized_signal,
        fs=ppg_config.sample_rate_hz,
        nperseg=min(256, processed_ppg.normalized_signal.size),
        noverlap=min(192, max(0, min(256, processed_ppg.normalized_signal.size) - 1)),
        scaling="density",
        mode="magnitude",
    )
    spec = np.maximum(spec, 1e-10)
    mesh = axes_row[4].pcolormesh(times, freqs, 20 * np.log10(spec), shading="gouraud", cmap="magma")
    axes_row[4].set_ylim(0.0, 8.0)
    axes_row[4].set_title("PPG Spectrogram (dB)")
    axes_row[4].text(
        1.02,
        0.98,
        summary_text,
        transform=axes_row[4].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "#CCCCCC"},
    )
    plt.colorbar(mesh, ax=axes_row[4], fraction=0.046, pad=0.04)

    for axis in axes_row:
        axis.set_xlabel("Time (s)")
        axis.grid(alpha=0.2)


def load_ppg_window(
    ppg_path: Path,
    ppg_segment_index: int,
    window_index: int,
    window_length_sec: float,
    stride_sec: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    with h5py.File(ppg_path, "r") as file_handle:
        header = _signal_header_to_dict(file_handle, file_handle["signalHeader"])
        label_to_rate = {
            str(label): float(rate)
            for label, rate in zip(header["signal_labels"], header["samples_in_record"])
        }

        def load_segment_array(dataset_name: str) -> np.ndarray:
            reference = file_handle[dataset_name][ppg_segment_index, 0]
            return np.asarray(file_handle[reference]).astype(np.float32).reshape(-1)

        ppg_green = load_segment_array("PPG_GREEN")
        acc_x = load_segment_array("Accelerometer_X")
        acc_y = load_segment_array("Accelerometer_Y")
        acc_z = load_segment_array("Accelerometer_Z")

    ppg_sample_rate_hz = float(label_to_rate["PPG_GREEN"])
    acc_sample_rate_hz = float(label_to_rate["Accelerometer_X"])
    window_samples = int(round(window_length_sec * ppg_sample_rate_hz))
    stride_samples = int(round(stride_sec * ppg_sample_rate_hz))
    start_sample = window_index * stride_samples
    end_sample = start_sample + window_samples

    ppg_window = ppg_green[start_sample:end_sample]
    acc_start = int(round(start_sample * acc_sample_rate_hz / ppg_sample_rate_hz))
    acc_end = int(round(end_sample * acc_sample_rate_hz / ppg_sample_rate_hz))
    acc_window = np.column_stack(
        [
            acc_x[acc_start:acc_end],
            acc_y[acc_start:acc_end],
            acc_z[acc_start:acc_end],
        ]
    ).astype(np.float32)
    return ppg_window.astype(np.float32), acc_window, ppg_sample_rate_hz, acc_sample_rate_hz


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = load_all_accepted_rows(args.build_root)
    examples = pick_examples(summary_df)

    figure, axes = plt.subplots(len(examples), 5, figsize=(24, 5.5 * len(examples)), constrained_layout=True)
    if len(examples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, row in enumerate(examples):
        render_example(
            axes_row=axes[row_index],
            row=row,
            dataset_root=args.dataset_root,
            window_length_sec=args.window_length_sec,
            stride_sec=args.stride_sec,
        )

    figure.suptitle(
        "Zenodo Signal Processing Sanity Check: Raw PPG -> Filtered -> Peaks -> ECG Alignment -> Spectrogram",
        fontsize=16,
    )
    figure.savefig(args.output_path, dpi=180, bbox_inches="tight")
    print(f"saved visualization to {args.output_path}")


if __name__ == "__main__":
    main()
