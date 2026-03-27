from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signal_pipeline import (
    bandpass_cheby2,
    default_ecg_config,
    default_ppg_config,
    detect_beats,
    interpolate_nan,
    zscore_normalize,
)


@dataclass
class ModalityResult:
    signal: np.ndarray
    peaks: np.ndarray
    ibi_ms: np.ndarray
    mean_hr_bpm: float


def load_segment(csv_path: Path, start_sec: float, length_sec: float, sample_rate_hz: float) -> tuple[np.ndarray, pd.DataFrame]:
    dataframe = pd.read_csv(csv_path)
    start_idx = int(round(start_sec * sample_rate_hz))
    length_samples = int(round(length_sec * sample_rate_hz))
    end_idx = start_idx + length_samples
    return dataframe["Time"].to_numpy(dtype=float)[start_idx:end_idx], dataframe.iloc[start_idx:end_idx].copy()


def process_modality(segment_df: pd.DataFrame, column: str, config) -> ModalityResult:
    signal = interpolate_nan(segment_df[column].to_numpy(dtype=float))
    signal = bandpass_cheby2(signal, config)
    signal = zscore_normalize(signal)
    peaks = detect_beats(signal, config)
    ibi_ms = np.diff(peaks) / config.sample_rate_hz * 1000.0 if peaks.size >= 2 else np.empty(0, dtype=float)
    mean_hr_bpm = float(np.mean(60.0 / np.maximum(np.diff(peaks) / config.sample_rate_hz, 1e-8))) if peaks.size >= 2 else float("nan")
    return ModalityResult(signal=signal, peaks=peaks, ibi_ms=ibi_ms, mean_hr_bpm=mean_hr_bpm)


def greedy_match_peaks(ecg_peaks: np.ndarray, ppg_peaks: np.ndarray, sample_rate_hz: float) -> list[tuple[int, int, float]]:
    used_ppg: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for ecg_peak in ecg_peaks:
        candidates = []
        for ppg_peak in ppg_peaks:
            delay_sec = (ppg_peak - ecg_peak) / sample_rate_hz
            if 0.03 <= delay_sec <= 0.45 and int(ppg_peak) not in used_ppg:
                candidates.append((abs(delay_sec - 0.15), int(ppg_peak), float(delay_sec)))
        if not candidates:
            continue
        _, matched_ppg, delay_sec = min(candidates, key=lambda item: item[0])
        used_ppg.add(matched_ppg)
        matches.append((int(ecg_peak), matched_ppg, delay_sec))

    return matches


def summarize_alignment(ecg: ModalityResult, ppg: ModalityResult, sample_rate_hz: float) -> dict[str, float]:
    matches = greedy_match_peaks(ecg.peaks, ppg.peaks, sample_rate_hz)
    delays_ms = np.array([delay_sec * 1000.0 for _, _, delay_sec in matches], dtype=float)

    summary = {
        "ecg_peak_count": float(ecg.peaks.size),
        "ppg_peak_count": float(ppg.peaks.size),
        "matched_peak_count": float(len(matches)),
        "ecg_mean_hr_bpm": float(ecg.mean_hr_bpm),
        "ppg_mean_hr_bpm": float(ppg.mean_hr_bpm),
        "hr_abs_diff_bpm": float(abs(ecg.mean_hr_bpm - ppg.mean_hr_bpm)),
        "median_delay_ms": float(np.median(delays_ms)) if delays_ms.size else float("nan"),
        "mean_delay_ms": float(np.mean(delays_ms)) if delays_ms.size else float("nan"),
        "std_delay_ms": float(np.std(delays_ms)) if delays_ms.size else float("nan"),
        "ibi_mae_ms": float("nan"),
        "ibi_corr": float("nan"),
    }

    if len(matches) >= 3:
        ecg_match = np.array([ecg_peak for ecg_peak, _, _ in matches], dtype=float)
        ppg_match = np.array([ppg_peak for _, ppg_peak, _ in matches], dtype=float)
        ecg_ibi = np.diff(ecg_match) / sample_rate_hz * 1000.0
        ppg_ibi = np.diff(ppg_match) / sample_rate_hz * 1000.0
        summary["ibi_mae_ms"] = float(np.mean(np.abs(ecg_ibi - ppg_ibi)))
        if ecg_ibi.size >= 2 and ppg_ibi.size >= 2:
            summary["ibi_corr"] = float(np.corrcoef(ecg_ibi, ppg_ibi)[0, 1])

    return summary


def plot_comparison(
    time_values: np.ndarray,
    ecg: ModalityResult,
    ppg: ModalityResult,
    summary: dict[str, float],
    output_path: Path,
) -> None:
    median_delay_ms = summary["median_delay_ms"]
    delay_sec = 0.0 if not np.isfinite(median_delay_ms) else median_delay_ms / 1000.0
    shifted_ppg_time = time_values - delay_sec

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), constrained_layout=True)

    axes[0].plot(time_values, ecg.signal, color="#2c3e50", linewidth=1.0, label="ECG filtered")
    axes[0].scatter(time_values[ecg.peaks], ecg.signal[ecg.peaks], color="#e74c3c", s=18, label="ECG peaks")
    axes[0].set_title("Filtered ECG")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(time_values, ppg.signal, color="#16a085", linewidth=1.0, label="PPG filtered")
    axes[1].scatter(time_values[ppg.peaks], ppg.signal[ppg.peaks], color="#8e44ad", s=18, label="PPG peaks")
    axes[1].set_title("Filtered PPG")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    axes[2].plot(time_values, ecg.signal, color="#34495e", linewidth=1.0, label="ECG filtered")
    axes[2].plot(shifted_ppg_time, ppg.signal, color="#27ae60", linewidth=1.0, alpha=0.85, label=f"PPG shifted by {delay_sec*1000:.1f} ms")
    axes[2].set_title("ECG vs PPG Overlay After Delay Compensation")
    axes[2].legend()
    axes[2].grid(alpha=0.2)

    text_lines = [
        f"ECG peaks: {int(summary['ecg_peak_count'])}",
        f"PPG peaks: {int(summary['ppg_peak_count'])}",
        f"Matched peaks: {int(summary['matched_peak_count'])}",
        f"ECG mean HR: {summary['ecg_mean_hr_bpm']:.2f} bpm",
        f"PPG mean HR: {summary['ppg_mean_hr_bpm']:.2f} bpm",
        f"|HR diff|: {summary['hr_abs_diff_bpm']:.2f} bpm",
        f"Median ECG->PPG delay: {summary['median_delay_ms']:.1f} ms",
        f"IBI MAE: {summary['ibi_mae_ms']:.1f} ms",
        f"IBI corr: {summary['ibi_corr']:.3f}",
    ]
    axes[3].axis("off")
    axes[3].text(0.02, 0.98, "\n".join(text_lines), va="top", ha="left", fontsize=12, family="monospace")
    axes[3].set_title("Rhythm Agreement Summary")

    for ax in axes[:3]:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized amplitude")

    fig.suptitle("ECG vs PPG Rhythm Agreement", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare same-record ECG and PPG timing agreement.")
    parser.add_argument("--csv-path", type=Path, required=True, help="Path to a *_data.csv file containing both ECG and PPG")
    parser.add_argument("--start-sec", type=float, default=120.0, help="Segment start time in seconds")
    parser.add_argument("--length-sec", type=float, default=30.0, help="Segment length in seconds")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/filter_viz/ecg_ppg_alignment.png"),
        help="PNG path to write",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_rate_hz = default_ecg_config().sample_rate_hz
    time_values, segment_df = load_segment(args.csv_path, args.start_sec, args.length_sec, sample_rate_hz)

    ecg = process_modality(segment_df, "ECG", default_ecg_config())
    ppg = process_modality(segment_df, "PPG", default_ppg_config())
    summary = summarize_alignment(ecg, ppg, sample_rate_hz)
    plot_comparison(time_values, ecg, ppg, summary, args.output)

    print(f"saved_plot: {args.output}")
    for key, value in summary.items():
        if np.isfinite(value):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: nan")


if __name__ == "__main__":
    main()
