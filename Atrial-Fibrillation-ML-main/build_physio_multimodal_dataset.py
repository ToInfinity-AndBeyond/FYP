from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from signal_pipeline import (
    EPS,
    SignalPipelineConfig,
    apply_quality_overrides,
    default_ecg_config,
    default_ppg_config,
    infer_label_from_path,
    interpolate_nan,
    load_quality_overrides,
    load_window_label_table,
    process_segment,
    signal_spectral_entropy,
    zscore_normalize,
)


MAX_IBI_LENGTH = 128


@dataclass(frozen=True)
class RespConfig:
    sample_rate_hz: float = 125.0
    low_hz: float = 0.05
    high_hz: float = 0.70
    order: int = 4
    stopband_attenuation_db: float = 20.0


def discover_csvs(dataset_root: Path) -> list[Path]:
    patterns = [
        "mimic_perform_af_csv/*_data.csv",
        "mimic_perform_non_af_csv/*_data.csv",
    ]
    csv_paths: list[Path] = []
    for pattern in patterns:
        csv_paths.extend(sorted(dataset_root.glob(pattern)))
    return sorted(csv_paths)


def preprocess_respiration(resp_segment: np.ndarray, config: RespConfig) -> np.ndarray:
    values = interpolate_nan(resp_segment)
    sos = sp_signal.cheby2(
        config.order,
        config.stopband_attenuation_db,
        [config.low_hz, config.high_hz],
        btype="bandpass",
        fs=config.sample_rate_hz,
        output="sos",
    )
    filtered = sp_signal.sosfiltfilt(sos, values)
    return zscore_normalize(filtered)


def pad_sequence(values: np.ndarray, max_length: int = MAX_IBI_LENGTH) -> tuple[np.ndarray, int]:
    padded = np.zeros(max_length, dtype=np.float32)
    valid_length = int(min(values.size, max_length))
    if valid_length > 0:
        padded[:valid_length] = values[:valid_length].astype(np.float32)
    return padded, valid_length


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3 or x.size != y.size:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < EPS or y_std < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def greedy_match_peaks(ecg_peaks: np.ndarray, ppg_peaks: np.ndarray, sample_rate_hz: float) -> list[tuple[int, int, float]]:
    used_ppg: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for ecg_peak in ecg_peaks:
        candidates = []
        for ppg_peak in ppg_peaks:
            delay_sec = (int(ppg_peak) - int(ecg_peak)) / sample_rate_hz
            if 0.03 <= delay_sec <= 0.45 and int(ppg_peak) not in used_ppg:
                candidates.append((abs(delay_sec - 0.15), int(ppg_peak), float(delay_sec)))
        if not candidates:
            continue
        _, matched_ppg, delay_sec = min(candidates, key=lambda item: item[0])
        used_ppg.add(matched_ppg)
        matches.append((int(ecg_peak), matched_ppg, delay_sec))

    return matches


def summarize_timing_consistency(
    ecg_peaks: np.ndarray,
    ppg_peaks: np.ndarray,
    sample_rate_hz: float,
) -> dict[str, float]:
    matches = greedy_match_peaks(ecg_peaks, ppg_peaks, sample_rate_hz)
    delays_ms = np.asarray([delay_sec * 1000.0 for _, _, delay_sec in matches], dtype=np.float32)

    summary = {
        "timing_matched_peak_count": float(len(matches)),
        "timing_median_delay_ms": float(np.median(delays_ms)) if delays_ms.size else float("nan"),
        "timing_mean_delay_ms": float(np.mean(delays_ms)) if delays_ms.size else float("nan"),
        "timing_std_delay_ms": float(np.std(delays_ms)) if delays_ms.size else float("nan"),
        "timing_ibi_mae_ms": float("nan"),
        "timing_ibi_corr": float("nan"),
    }

    if len(matches) >= 3:
        ecg_match = np.array([ecg_peak for ecg_peak, _, _ in matches], dtype=np.float32)
        ppg_match = np.array([ppg_peak for _, ppg_peak, _ in matches], dtype=np.float32)
        ecg_ibi = np.diff(ecg_match) / sample_rate_hz * 1000.0
        ppg_ibi = np.diff(ppg_match) / sample_rate_hz * 1000.0
        summary["timing_ibi_mae_ms"] = float(np.mean(np.abs(ecg_ibi - ppg_ibi)))
        summary["timing_ibi_corr"] = safe_corrcoef(ecg_ibi, ppg_ibi)

    return summary


def summarize_respiration(
    resp_signal: np.ndarray,
    ppg_signal: np.ndarray,
    ppg_peaks: np.ndarray,
    sample_rate_hz: float,
) -> dict[str, float]:
    freqs, psd = sp_signal.welch(resp_signal, fs=sample_rate_hz, nperseg=min(resp_signal.size, 512))
    band_mask = (freqs >= 0.05) & (freqs <= 0.70)
    if np.any(band_mask):
        resp_band_freqs = freqs[band_mask]
        resp_band_psd = psd[band_mask]
        peak_idx = int(np.argmax(resp_band_psd))
        resp_rate_bpm = float(resp_band_freqs[peak_idx] * 60.0)
    else:
        resp_rate_bpm = float("nan")

    beat_amp_corr = float("nan")
    ibi_resp_corr = float("nan")
    if ppg_peaks.size >= 4:
        beat_amplitudes = ppg_signal[ppg_peaks]
        resp_at_peaks = resp_signal[ppg_peaks]
        beat_amp_corr = safe_corrcoef(beat_amplitudes.astype(np.float32), resp_at_peaks.astype(np.float32))

        ibi_seconds = np.diff(ppg_peaks).astype(np.float32) / sample_rate_hz
        midpoint_idx = ((ppg_peaks[:-1] + ppg_peaks[1:]) / 2.0).astype(int)
        midpoint_idx = np.clip(midpoint_idx, 0, resp_signal.size - 1)
        resp_at_midpoints = resp_signal[midpoint_idx]
        ibi_resp_corr = safe_corrcoef(ibi_seconds, resp_at_midpoints.astype(np.float32))

    return {
        "resp_rate_bpm": resp_rate_bpm,
        "resp_spectral_entropy": signal_spectral_entropy(resp_signal, sample_rate_hz),
        "resp_ppg_amplitude_corr": beat_amp_corr,
        "resp_ibi_corr": ibi_resp_corr,
    }


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def build_multimodal_dataset(
    csv_paths: list[Path],
    ppg_config: SignalPipelineConfig,
    ecg_config: SignalPipelineConfig,
    resp_config: RespConfig,
    window_label_table: pd.DataFrame | None = None,
    fallback_to_path_labels: bool = False,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, float | int | str | bool]] = []
    ppg_segments: list[np.ndarray] = []
    ecg_segments: list[np.ndarray] = []
    resp_segments: list[np.ndarray] = []
    ppg_ibi_sequences: list[np.ndarray] = []
    ecg_ibi_sequences: list[np.ndarray] = []
    ppg_ibi_lengths: list[int] = []
    ecg_ibi_lengths: list[int] = []

    segment_samples = int(round(ppg_config.segment.length_seconds * ppg_config.sample_rate_hz))
    stride_samples = int(round(ppg_config.segment.stride_seconds * ppg_config.sample_rate_hz))
    label_groups: dict[str, pd.DataFrame] = {}
    if window_label_table is not None and not window_label_table.empty:
        label_groups = {
            str(record_id): group.copy()
            for record_id, group in window_label_table.groupby("record_id", sort=False)
        }

    for csv_path in csv_paths:
        dataframe = pd.read_csv(csv_path)
        record_id = csv_path.stem.replace("_data", "")
        record_window_labels = label_groups.get(record_id)
        if window_label_table is not None and record_window_labels is None and not fallback_to_path_labels:
            continue
        label_lookup = None
        if record_window_labels is not None:
            label_lookup = (
                record_window_labels.sort_values("segment_index")
                .set_index("segment_index")
                .to_dict(orient="index")
            )
        default_label = infer_label_from_path(csv_path) if (record_window_labels is None or fallback_to_path_labels) else None

        time_values = dataframe["Time"].to_numpy(dtype=float)
        raw_ppg = dataframe[ppg_config.signal_column].to_numpy(dtype=float)
        raw_ecg = dataframe[ecg_config.signal_column].to_numpy(dtype=float)
        resp_available = "resp" in dataframe.columns
        raw_resp = dataframe["resp"].to_numpy(dtype=float) if resp_available else np.zeros_like(raw_ppg, dtype=float)

        for start in range(0, raw_ppg.size - segment_samples + 1, stride_samples):
            end = start + segment_samples
            segment_index = int(start // stride_samples)
            label_source = "path"
            label_metadata: dict[str, float | int | str | bool] = {}
            if label_lookup is not None:
                label_entry = label_lookup.get(segment_index)
                if label_entry is None:
                    continue
                current_label = int(label_entry["label"])
                label_source = str(label_entry.get("label_source", "window_label_csv"))
                label_metadata = {
                    f"label_{key}": value
                    for key, value in label_entry.items()
                    if key not in {"label", "label_source", "use_for_training"}
                }
            elif default_label is not None:
                current_label = int(default_label)
            else:
                continue

            ppg_processed = process_segment(raw_ppg[start:end], ppg_config)
            ecg_processed = process_segment(raw_ecg[start:end], ecg_config)
            if resp_available:
                resp_processed = preprocess_respiration(raw_resp[start:end], resp_config)
                resp_metrics = summarize_respiration(
                    resp_processed,
                    ppg_processed.normalized_signal,
                    ppg_processed.peaks,
                    ppg_config.sample_rate_hz,
                )
            else:
                resp_processed = np.zeros(segment_samples, dtype=np.float32)
                resp_metrics = {
                    "resp_rate_bpm": float("nan"),
                    "resp_spectral_entropy": float("nan"),
                    "resp_ppg_amplitude_corr": float("nan"),
                    "resp_ibi_corr": float("nan"),
                }

            timing_metrics = summarize_timing_consistency(
                ecg_processed.peaks,
                ppg_processed.peaks,
                ppg_config.sample_rate_hz,
            )

            ppg_ibi_padded, ppg_ibi_length = pad_sequence(ppg_processed.ibi_seconds)
            ecg_ibi_padded, ecg_ibi_length = pad_sequence(ecg_processed.ibi_seconds)

            ppg_accepted = bool(ppg_processed.quality_metrics["accepted"])
            ecg_accepted = bool(ecg_processed.quality_metrics["accepted"])
            joint_accepted = ppg_accepted and ecg_accepted

            row: dict[str, float | int | str | bool] = {
                "record_id": record_id,
                "label": int(current_label),
                "label_source": label_source,
                "segment_index": segment_index,
                "start_sample": int(start),
                "end_sample": int(end),
                "start_time_sec": float(time_values[start]),
                "end_time_sec": float(time_values[end - 1]),
                "ppg_quality_score": float(ppg_processed.quality_metrics["quality_score"]),
                "resp_available": int(resp_available),
                "ppg_accepted": ppg_accepted,
                "ecg_accepted": ecg_accepted,
                "joint_accepted": joint_accepted,
                "joint_rejection_reason": ";".join(
                    reason
                    for reason in [
                        ppg_processed.quality_metrics["rejection_reason"],
                        ecg_processed.quality_metrics["rejection_reason"],
                    ]
                    if isinstance(reason, str) and reason
                ),
                "ppg_ibi_length": int(ppg_ibi_length),
                "ecg_ibi_length": int(ecg_ibi_length),
            }
            row.update(label_metadata)
            row.update(prefix_metrics(ppg_processed.quality_metrics, "ppg_"))
            row.update(prefix_metrics(ppg_processed.feature_metrics, "ppg_"))
            row.update(prefix_metrics(ecg_processed.quality_metrics, "ecg_"))
            row.update(prefix_metrics(ecg_processed.feature_metrics, "ecg_"))
            row.update(timing_metrics)
            row.update(resp_metrics)

            rows.append(row)
            ppg_segments.append(ppg_processed.normalized_signal.astype(np.float32))
            ecg_segments.append(ecg_processed.normalized_signal.astype(np.float32))
            resp_segments.append(resp_processed.astype(np.float32))
            ppg_ibi_sequences.append(ppg_ibi_padded)
            ecg_ibi_sequences.append(ecg_ibi_padded)
            ppg_ibi_lengths.append(ppg_ibi_length)
            ecg_ibi_lengths.append(ecg_ibi_length)

    if not rows:
        empty_summary = pd.DataFrame()
        empty_arrays = {
            "ppg_segments": np.empty((0, segment_samples), dtype=np.float32),
            "ecg_segments": np.empty((0, segment_samples), dtype=np.float32),
            "resp_segments": np.empty((0, segment_samples), dtype=np.float32),
            "ppg_ibi_sequences": np.empty((0, MAX_IBI_LENGTH), dtype=np.float32),
            "ecg_ibi_sequences": np.empty((0, MAX_IBI_LENGTH), dtype=np.float32),
            "ppg_ibi_lengths": np.empty((0,), dtype=np.int16),
            "ecg_ibi_lengths": np.empty((0,), dtype=np.int16),
            "labels": np.empty((0,), dtype=np.int8),
            "ppg_quality_score": np.empty((0,), dtype=np.float32),
            "joint_accepted": np.empty((0,), dtype=bool),
            "start_time_sec": np.empty((0,), dtype=np.float32),
        }
        return empty_summary, empty_arrays

    summary_df = pd.DataFrame(rows)
    arrays = {
        "ppg_segments": np.stack(ppg_segments).astype(np.float32),
        "ecg_segments": np.stack(ecg_segments).astype(np.float32),
        "resp_segments": np.stack(resp_segments).astype(np.float32),
        "ppg_ibi_sequences": np.stack(ppg_ibi_sequences).astype(np.float32),
        "ecg_ibi_sequences": np.stack(ecg_ibi_sequences).astype(np.float32),
        "ppg_ibi_lengths": np.asarray(ppg_ibi_lengths, dtype=np.int16),
        "ecg_ibi_lengths": np.asarray(ecg_ibi_lengths, dtype=np.int16),
        "labels": summary_df["label"].to_numpy(dtype=np.int8),
        "ppg_quality_score": summary_df["ppg_quality_score"].to_numpy(dtype=np.float32),
        "joint_accepted": summary_df["joint_accepted"].to_numpy(dtype=bool),
        "start_time_sec": summary_df["start_time_sec"].to_numpy(dtype=np.float32),
    }
    return summary_df, arrays


def save_multimodal_bundle(
    summary_df: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    output_dir: Path,
    ppg_config: SignalPipelineConfig,
    ecg_config: SignalPipelineConfig,
    resp_config: RespConfig,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "physio_multimodal_segment_summary.csv"
    accepted_summary_path = output_dir / "physio_multimodal_accepted_segment_summary.csv"
    segments_path = output_dir / "physio_multimodal_segments.npz"
    accepted_segments_path = output_dir / "physio_multimodal_accepted_segments.npz"
    config_path = output_dir / "physio_multimodal_config.json"

    summary_df.to_csv(summary_path, index=False)
    if summary_df.empty:
        accepted_mask = np.empty((0,), dtype=bool)
        summary_df.to_csv(accepted_summary_path, index=False)
    else:
        accepted_mask = summary_df["joint_accepted"].to_numpy(dtype=bool)
        summary_df.loc[accepted_mask].to_csv(accepted_summary_path, index=False)

    np.savez_compressed(segments_path, **arrays)
    accepted_arrays = {}
    for key, values in arrays.items():
        if values.shape[0] == accepted_mask.shape[0]:
            accepted_arrays[key] = values[accepted_mask]
        else:
            accepted_arrays[key] = values
    np.savez_compressed(accepted_segments_path, **accepted_arrays)

    config_payload = {
        "ppg": asdict(ppg_config),
        "ecg": asdict(ecg_config),
        "resp": asdict(resp_config),
        "max_ibi_length": MAX_IBI_LENGTH,
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    return {
        "summary_csv": summary_path,
        "accepted_summary_csv": accepted_summary_path,
        "segments_npz": segments_path,
        "accepted_segments_npz": accepted_segments_path,
        "config_json": config_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build synchronized PPG/ECG/resp multimodal datasets for physiology-aware distillation."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root that contains mimic_perform_af_csv and mimic_perform_non_af_csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "physio_distill",
        help="Directory where multimodal outputs will be written",
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
        "--ppg-quality-json",
        type=Path,
        default=None,
        help="Optional JSON file containing tuned PPG quality gate overrides.",
    )
    parser.add_argument(
        "--ecg-quality-json",
        type=Path,
        default=None,
        help="Optional JSON file containing tuned ECG quality gate overrides.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = discover_csvs(args.dataset_root)
    if args.limit_files is not None:
        csv_paths = csv_paths[: args.limit_files]
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {args.dataset_root}")

    ppg_config = apply_quality_overrides(default_ppg_config(), load_quality_overrides(args.ppg_quality_json))
    ecg_config = apply_quality_overrides(default_ecg_config(), load_quality_overrides(args.ecg_quality_json))
    resp_config = RespConfig(sample_rate_hz=ppg_config.sample_rate_hz)
    window_label_table = load_window_label_table(args.window_label_csv) if args.window_label_csv is not None else None

    summary_df, arrays = build_multimodal_dataset(
        csv_paths,
        ppg_config,
        ecg_config,
        resp_config,
        window_label_table=window_label_table,
        fallback_to_path_labels=args.fallback_to_path_labels,
    )
    saved = save_multimodal_bundle(summary_df, arrays, args.output_dir, ppg_config, ecg_config, resp_config)

    print(f"records processed: {len(csv_paths)}")
    joint_accepted = int(summary_df["joint_accepted"].sum()) if not summary_df.empty else 0
    af_segments = int(summary_df["label"].sum()) if not summary_df.empty else 0
    print(f"segments: {int(summary_df.shape[0])}")
    print(f"joint accepted: {joint_accepted}")
    print(f"AF segments: {af_segments}")
    print("outputs:")
    for name, path in saved.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
