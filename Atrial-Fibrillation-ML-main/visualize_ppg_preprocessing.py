from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from build_mimic_ext_ppg_dataset import read_wfdb_segment, resolve_record_base
from signal_pipeline import default_ppg_config, process_segment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize raw vs processed PPG segments.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/vol/bitbucket/mc1920/FYP/1.1.0"),
        help="Root containing p00/p01/... waveform files.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("/vol/bitbucket/mc1920/mimic_ext_ppg_p00_by_fold/fold_9/ppg/ppg_accepted_segment_summary.csv"),
        help="Accepted segment summary CSV to sample from.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/vol/bitbucket/mc1920/preprocessing_visuals/p00_preprocessing_overview.png"),
        help="Where to save the PNG figure.",
    )
    return parser.parse_args()


def pick_examples(summary_df: pd.DataFrame) -> list[pd.Series]:
    examples: list[pd.Series] = []
    for label in (0, 1):
        subset = summary_df.loc[(summary_df["accepted"] == True) & (summary_df["label"] == label)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(["quality_score", "template_correlation"], ascending=False)
        examples.append(subset.iloc[0])
    if not examples:
        raise RuntimeError("No accepted PPG segments found to visualize.")
    return examples


def segment_time_axis(num_samples: int, sample_rate_hz: float) -> np.ndarray:
    return np.arange(num_samples, dtype=float) / sample_rate_hz


def render_example(ax_row: np.ndarray, row: pd.Series, dataset_root: Path, sample_rate_hz: float) -> None:
    record_base = resolve_record_base(dataset_root, row["folder_path"], row["signal_file_name"])
    waveform_df = read_wfdb_segment(record_base)

    start_sample = int(row["start_sample"])
    end_sample = int(row["end_sample"])
    raw_signal = waveform_df["PPG"].to_numpy(dtype=float)[start_sample:end_sample]
    processed = process_segment(raw_signal, default_ppg_config())
    time_axis = segment_time_axis(raw_signal.size, sample_rate_hz)

    label_name = "AF" if int(row["label"]) == 1 else "SR"
    summary_text = (
        f"{label_name} | q={row['quality_score']:.3f}\n"
        f"HR={row['mean_hr_bpm']:.1f} bpm | SDNN={row['sdnn_ms']:.1f} ms\n"
        f"RMSSD={row['rmssd_ms']:.1f} ms | peaks={int(row['peak_count'])}"
    )

    ax_row[0].plot(time_axis, raw_signal, color="#4C78A8", linewidth=1.0)
    ax_row[0].set_title(f"Raw PPG\n{row['signal_file_name']}")
    ax_row[0].set_ylabel("Amplitude")

    ax_row[1].plot(time_axis, processed.filtered_signal, color="#F58518", linewidth=1.0)
    ax_row[1].set_title("Bandpass Filtered")

    ax_row[2].plot(time_axis, processed.normalized_signal, color="#54A24B", linewidth=1.0)
    peak_times = processed.peaks.astype(float) / sample_rate_hz
    ax_row[2].scatter(
        peak_times,
        processed.normalized_signal[processed.peaks],
        s=10,
        color="#E45756",
        label="Detected peaks",
        zorder=3,
    )
    ax_row[2].legend(loc="upper right", fontsize=8, frameon=False)
    ax_row[2].set_title("Normalized + Peaks")

    freqs, times, spec = sp_signal.spectrogram(
        processed.normalized_signal,
        fs=sample_rate_hz,
        nperseg=min(256, processed.normalized_signal.size),
        noverlap=min(192, max(0, min(256, processed.normalized_signal.size) - 1)),
        scaling="density",
        mode="magnitude",
    )
    spec = np.maximum(spec, 1e-10)
    mesh = ax_row[3].pcolormesh(times, freqs, 20 * np.log10(spec), shading="gouraud", cmap="magma")
    ax_row[3].set_ylim(0.0, 8.0)
    ax_row[3].set_title("Spectrogram (dB)")
    ax_row[3].text(
        1.02,
        0.98,
        summary_text,
        transform=ax_row[3].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#CCCCCC"},
    )
    plt.colorbar(mesh, ax=ax_row[3], fraction=0.046, pad=0.04)

    for axis in ax_row:
        axis.set_xlabel("Time (s)")
        axis.grid(alpha=0.2)


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(args.summary_csv)
    examples = pick_examples(summary_df)
    config = default_ppg_config()

    figure, axes = plt.subplots(len(examples), 4, figsize=(20, 5 * len(examples)), constrained_layout=True)
    if len(examples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, row in enumerate(examples):
        render_example(axes[idx], row, args.dataset_root, config.sample_rate_hz)

    figure.suptitle("PPG Preprocessing Sanity Check: Raw -> Filtered -> Peaks -> Spectrogram", fontsize=16)
    figure.savefig(args.output_path, dpi=180, bbox_inches="tight")
    print(f"saved visualization to {args.output_path}")


if __name__ == "__main__":
    main()
