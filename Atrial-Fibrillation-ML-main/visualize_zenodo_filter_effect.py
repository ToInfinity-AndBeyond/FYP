from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal

from build_zenodo_datasets import (
    resample_1d,
    resample_acc,
    slice_ecg_window,
    slice_ppg_window,
)
from signal_pipeline import (
    apply_quality_overrides,
    bandpass_cheby2,
    default_ecg_config,
    default_ppg_config,
    detect_beats,
    load_quality_overrides,
    process_segment,
    zscore_normalize,
)
from zenodo_longterm_loader import (
    build_subject_window_labels,
    load_zenodo_ecg_mat,
    load_zenodo_ppg_mat,
)


def make_figure(
    time_values: np.ndarray,
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    normalized_signal: np.ndarray,
    peaks: np.ndarray,
    sample_rate_hz: float,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), constrained_layout=True)

    axes[0].plot(time_values, raw_signal, color="#7f8c8d", linewidth=1.0)
    axes[0].set_title("Raw Resampled Window")
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

    fig.suptitle(title, fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def select_window(
    window_label_df,
    labeled_window_index: int | None,
    ppg_segment_index: int | None,
    segment_index: int | None,
    desired_label: int | None,
):
    usable = window_label_df.loc[window_label_df["use_for_training"] == 1].copy()
    if desired_label is not None:
        usable = usable.loc[usable["label"] == desired_label].copy()
    if usable.empty:
        raise ValueError("No usable windows matched the selection criteria.")

    if ppg_segment_index is not None and segment_index is not None:
        match = usable.loc[
            (usable["ppg_segment_index"] == ppg_segment_index)
            & (usable["segment_index"] == segment_index)
        ]
        if match.empty:
            raise ValueError(
                f"No usable window matched ppg_segment_index={ppg_segment_index} and segment_index={segment_index}."
            )
        return match.iloc[0]

    if labeled_window_index is None:
        labeled_window_index = 0
    if labeled_window_index < 0 or labeled_window_index >= len(usable):
        raise IndexError(f"labeled_window_index must be in [0, {len(usable) - 1}]")
    return usable.iloc[labeled_window_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize raw vs filtered Zenodo ECG/PPG windows as a PNG.")
    parser.add_argument("--dataset-root", type=Path, default=Path("zenodo_longterm_af"))
    parser.add_argument("--subject-id", required=True, help="Subject ID such as 001")
    parser.add_argument("--signal-type", choices=["ppg", "ecg"], default="ppg")
    parser.add_argument("--window-length-sec", type=float, default=30.0)
    parser.add_argument("--stride-sec", type=float, default=30.0)
    parser.add_argument("--positive-fraction-threshold", type=float, default=0.5)
    parser.add_argument("--target-sample-rate-hz", type=float, default=125.0)
    parser.add_argument("--labeled-window-index", type=int, default=0, help="Index within usable labeled windows.")
    parser.add_argument("--ppg-segment-index", type=int, default=None, help="Optional raw PPG segment index.")
    parser.add_argument("--segment-index", type=int, default=None, help="Optional 30-second window index inside the PPG segment.")
    parser.add_argument("--desired-label", type=int, choices=[0, 1], default=None, help="Optionally select only AF(1) or non-AF(0) windows.")
    parser.add_argument("--ppg-quality-json", type=Path, default=None)
    parser.add_argument("--ecg-quality-json", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/filter_viz/zenodo_filter_effect.png"),
        help="PNG path to write",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ecg_mat = args.dataset_root / f"{args.subject_id}_ECG.mat"
    ppg_mat = args.dataset_root / f"{args.subject_id}_PPG.mat"

    ecg_record = load_zenodo_ecg_mat(ecg_mat)
    ppg_segments, _ = load_zenodo_ppg_mat(ppg_mat, ecg_record=ecg_record)
    window_label_df = build_subject_window_labels(
        ecg_record=ecg_record,
        ppg_segments=ppg_segments,
        window_length_sec=args.window_length_sec,
        stride_sec=args.stride_sec,
        positive_fraction_threshold=args.positive_fraction_threshold,
    )

    selected = select_window(
        window_label_df=window_label_df,
        labeled_window_index=args.labeled_window_index,
        ppg_segment_index=args.ppg_segment_index,
        segment_index=args.segment_index,
        desired_label=args.desired_label,
    )

    ppg_config = apply_quality_overrides(
        default_ppg_config(sample_rate_hz=args.target_sample_rate_hz),
        load_quality_overrides(args.ppg_quality_json),
    )
    ecg_config = apply_quality_overrides(
        default_ecg_config(sample_rate_hz=args.target_sample_rate_hz),
        load_quality_overrides(args.ecg_quality_json),
    )

    if args.signal_type == "ppg":
        segment_lookup = {segment.segment_index: segment for segment in ppg_segments}
        ppg_segment = segment_lookup[int(selected["ppg_segment_index"])]
        raw_window, acc_window = slice_ppg_window(
            ppg_segment,
            window_index=int(selected["segment_index"]),
            window_length_sec=args.window_length_sec,
            stride_sec=args.stride_sec,
        )
        raw_resampled = resample_1d(raw_window, ppg_segment.ppg_sample_rate_hz, ppg_config.sample_rate_hz)
        acc_resampled = resample_acc(acc_window, ppg_segment.acc_sample_rate_hz, ppg_config.sample_rate_hz)
        processed = process_segment(raw_resampled, ppg_config, acc_segment=acc_resampled)
        config = ppg_config
        title = (
            f"Zenodo {args.subject_id} PPG Filter Effect | "
            f"label={int(selected['label'])} af_fraction={float(selected['af_fraction']):.3f} "
            f"accepted={int(processed.quality_metrics['accepted'])}"
        )
        raw_for_plot = raw_resampled
    else:
        raw_window = slice_ecg_window(
            ecg_record=ecg_record,
            start_time_sec=float(selected["start_time_sec"]),
            window_length_sec=args.window_length_sec,
        )
        raw_resampled = resample_1d(raw_window, ecg_record.sample_rate_hz, ecg_config.sample_rate_hz)
        processed = process_segment(raw_resampled, ecg_config)
        config = ecg_config
        title = (
            f"Zenodo {args.subject_id} ECG Filter Effect | "
            f"label={int(selected['label'])} af_fraction={float(selected['af_fraction']):.3f} "
            f"accepted={int(processed.quality_metrics['accepted'])}"
        )
        raw_for_plot = raw_resampled

    time_values = np.arange(raw_for_plot.size, dtype=float) / config.sample_rate_hz
    make_figure(
        time_values=time_values,
        raw_signal=raw_for_plot,
        filtered_signal=processed.filtered_signal,
        normalized_signal=processed.normalized_signal,
        peaks=processed.peaks,
        sample_rate_hz=config.sample_rate_hz,
        title=title,
        output_path=args.output,
    )

    print(f"saved_plot: {args.output}")
    print(f"selected_record_id: zenodo_{args.subject_id}")
    print(f"selected_ppg_segment_index: {int(selected['ppg_segment_index'])}")
    print(f"selected_window_index: {int(selected['segment_index'])}")
    print(f"selected_label: {int(selected['label'])}")
    print(f"selected_af_fraction: {float(selected['af_fraction']):.6f}")
    print(f"accepted: {int(processed.quality_metrics['accepted'])}")
    print(f"detected_peaks: {int(processed.peaks.size)}")


if __name__ == "__main__":
    main()
