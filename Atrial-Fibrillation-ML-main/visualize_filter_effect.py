from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from signal_pipeline import (
    bandpass_cheby2,
    default_ecg_config,
    default_ppg_config,
    detect_beats,
    interpolate_nan,
    zscore_normalize,
)


def load_segment(
    csv_path: Path,
    signal_column: str,
    sample_rate_hz: float,
    start_sec: float,
    length_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    dataframe = pd.read_csv(csv_path)
    if signal_column not in dataframe.columns:
        raise KeyError(f"Missing signal column: {signal_column}")

    start_idx = int(round(start_sec * sample_rate_hz))
    segment_len = int(round(length_sec * sample_rate_hz))
    end_idx = start_idx + segment_len

    signal_values = dataframe[signal_column].to_numpy(dtype=float)
    if "Time" in dataframe.columns:
        time_values = dataframe["Time"].to_numpy(dtype=float)
    else:
        time_values = np.arange(signal_values.size, dtype=float) / sample_rate_hz

    if end_idx > signal_values.size:
        raise ValueError(
            f"Requested segment ends at sample {end_idx}, but file only has {signal_values.size} samples."
        )

    return time_values[start_idx:end_idx], signal_values[start_idx:end_idx]


def make_figure(
    time_values: np.ndarray,
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    normalized_signal: np.ndarray,
    peaks: np.ndarray,
    sample_rate_hz: float,
    signal_name: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), constrained_layout=True)

    axes[0].plot(time_values, raw_signal, color="#7f8c8d", linewidth=1.0)
    axes[0].set_title(f"Raw {signal_name.upper()} Segment")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.2)

    axes[1].plot(time_values, raw_signal, color="#bdc3c7", linewidth=1.0, label="Raw")
    axes[1].plot(time_values, filtered_signal, color="#d35400", linewidth=1.2, label="Band-pass filtered")
    axes[1].set_title("Raw vs Filtered")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    axes[2].plot(time_values, normalized_signal, color="#1f77b4", linewidth=1.0, label="Normalized filtered")
    if peaks.size > 0:
        axes[2].scatter(time_values[peaks], normalized_signal[peaks], color="#c0392b", s=18, label="Detected peaks")
    axes[2].set_title("Filtered Signal With Detected Beats")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Z-score")
    axes[2].legend()
    axes[2].grid(alpha=0.2)

    raw_freqs, raw_psd = sp_signal.welch(raw_signal, fs=sample_rate_hz, nperseg=min(raw_signal.size, 1024))
    filt_freqs, filt_psd = sp_signal.welch(filtered_signal, fs=sample_rate_hz, nperseg=min(filtered_signal.size, 1024))
    axes[3].semilogy(raw_freqs, raw_psd + 1e-12, color="#95a5a6", linewidth=1.0, label="Raw PSD")
    axes[3].semilogy(filt_freqs, filt_psd + 1e-12, color="#27ae60", linewidth=1.2, label="Filtered PSD")
    axes[3].set_xlim(0, min(20.0, sample_rate_hz / 2.0))
    axes[3].set_title("Power Spectrum Before/After Filtering")
    axes[3].set_xlabel("Frequency (Hz)")
    axes[3].set_ylabel("PSD")
    axes[3].legend()
    axes[3].grid(alpha=0.2)

    fig.suptitle(f"{signal_name.upper()} Filter Effect Visualization", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a PNG that shows raw vs filtered AF signal segments.")
    parser.add_argument("--csv-path", type=Path, required=True, help="Path to a *_data.csv file")
    parser.add_argument("--signal-type", choices=["ppg", "ecg"], default="ppg", help="Signal modality to visualize")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Segment start time in seconds")
    parser.add_argument("--length-sec", type=float, default=30.0, help="Segment length in seconds")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/filter_viz/filter_effect.png"),
        help="PNG path to write",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = default_ppg_config() if args.signal_type == "ppg" else default_ecg_config()
    time_values, raw_signal = load_segment(
        csv_path=args.csv_path,
        signal_column=config.signal_column,
        sample_rate_hz=config.sample_rate_hz,
        start_sec=args.start_sec,
        length_sec=args.length_sec,
    )

    raw_signal = interpolate_nan(raw_signal)
    filtered_signal = bandpass_cheby2(raw_signal, config)
    normalized_signal = zscore_normalize(filtered_signal)
    peaks = detect_beats(normalized_signal, config)

    make_figure(
        time_values=time_values,
        raw_signal=raw_signal,
        filtered_signal=filtered_signal,
        normalized_signal=normalized_signal,
        peaks=peaks,
        sample_rate_hz=config.sample_rate_hz,
        signal_name=args.signal_type,
        output_path=args.output,
    )

    print(f"saved_plot: {args.output}")
    print(f"detected_peaks: {int(peaks.size)}")


if __name__ == "__main__":
    main()
