from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional
import json
import warnings

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy import stats

EPS = 1e-8
WINDOW_LABEL_REQUIRED_COLUMNS = {"record_id", "segment_index", "label"}


@dataclass(frozen=True)
class BandpassFilterConfig:
    low_hz: float
    high_hz: float
    order: int = 4
    stopband_attenuation_db: float = 20.0


@dataclass(frozen=True)
class SegmentConfig:
    length_seconds: float = 30.0
    stride_seconds: float = 30.0
    zscore_normalize: bool = True
    use_adaptive_motion_cancellation: bool = True


@dataclass(frozen=True)
class QualityGateConfig:
    heart_band_hz: tuple[float, float] = (0.5, 3.5)
    total_band_hz: tuple[float, float] = (0.1, 8.0)
    min_heart_band_energy_ratio: float = 0.45
    max_abs_skewness: float = 2.0
    min_template_correlation: float = 0.45
    max_acc_variance: Optional[float] = None
    min_peak_count: int = 8


@dataclass(frozen=True)
class PeakDetectionConfig:
    min_hr_bpm: float = 35.0
    max_hr_bpm: float = 220.0
    prominence_scale: float = 0.35
    min_absolute_prominence: float = 0.02
    refine_radius_seconds: float = 0.20


@dataclass(frozen=True)
class SignalPipelineConfig:
    signal_name: str
    signal_column: str
    sample_rate_hz: float
    bandpass: BandpassFilterConfig
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    quality: QualityGateConfig = field(default_factory=QualityGateConfig)
    peaks: PeakDetectionConfig = field(default_factory=PeakDetectionConfig)
    acc_columns: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class ProcessedSegment:
    filtered_signal: np.ndarray
    normalized_signal: np.ndarray
    peaks: np.ndarray
    ibi_seconds: np.ndarray
    quality_metrics: dict[str, Any]
    feature_metrics: dict[str, float]


def load_quality_overrides(quality_json_path: Path | None) -> dict[str, Any] | None:
    if quality_json_path is None:
        return None
    payload = json.loads(quality_json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "quality_overrides" in payload and isinstance(payload["quality_overrides"], dict):
            return payload["quality_overrides"]
        if "quality" in payload and isinstance(payload["quality"], dict):
            return payload["quality"]
        return payload
    raise ValueError(f"Unsupported quality override format in {quality_json_path}")


def apply_quality_overrides(
    config: SignalPipelineConfig,
    overrides: dict[str, Any] | None,
) -> SignalPipelineConfig:
    if not overrides:
        return config
    valid_fields = set(QualityGateConfig.__dataclass_fields__.keys())
    filtered_overrides = {
        key: value
        for key, value in overrides.items()
        if key in valid_fields and value is not None
    }
    if not filtered_overrides:
        return config
    return replace(config, quality=replace(config.quality, **filtered_overrides))


def default_ppg_config(sample_rate_hz: float = 125.0) -> SignalPipelineConfig:
    return SignalPipelineConfig(
        signal_name="ppg",
        signal_column="PPG",
        sample_rate_hz=sample_rate_hz,
        bandpass=BandpassFilterConfig(low_hz=0.5, high_hz=8.0, order=4, stopband_attenuation_db=20.0),
        segment=SegmentConfig(length_seconds=30.0, stride_seconds=30.0, zscore_normalize=True),
        quality=QualityGateConfig(
            heart_band_hz=(0.5, 3.5),
            total_band_hz=(0.1, 8.0),
            min_heart_band_energy_ratio=0.45,
            max_abs_skewness=2.0,
            min_template_correlation=0.45,
            max_acc_variance=None,
            min_peak_count=8,
        ),
        peaks=PeakDetectionConfig(
            min_hr_bpm=35.0,
            max_hr_bpm=220.0,
            prominence_scale=0.30,
            min_absolute_prominence=0.02,
            refine_radius_seconds=0.20,
        ),
        acc_columns=("ACC_X", "ACC_Y", "ACC_Z"),
    )


def default_ecg_config(sample_rate_hz: float = 125.0) -> SignalPipelineConfig:
    return SignalPipelineConfig(
        signal_name="ecg",
        signal_column="ECG",
        sample_rate_hz=sample_rate_hz,
        bandpass=BandpassFilterConfig(low_hz=0.5, high_hz=40.0, order=4, stopband_attenuation_db=20.0),
        segment=SegmentConfig(length_seconds=30.0, stride_seconds=30.0, zscore_normalize=True),
        quality=QualityGateConfig(
            heart_band_hz=(0.5, 6.0),
            total_band_hz=(0.1, 40.0),
            min_heart_band_energy_ratio=0.25,
            max_abs_skewness=4.5,
            min_template_correlation=0.35,
            max_acc_variance=None,
            min_peak_count=8,
        ),
        peaks=PeakDetectionConfig(
            min_hr_bpm=35.0,
            max_hr_bpm=240.0,
            prominence_scale=0.40,
            min_absolute_prominence=0.04,
            refine_radius_seconds=0.12,
        ),
        acc_columns=("ACC_X", "ACC_Y", "ACC_Z"),
    )


def save_config(config: SignalPipelineConfig, output_path: Path) -> None:
    output_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def interpolate_nan(signal_values: np.ndarray) -> np.ndarray:
    values = np.asarray(signal_values, dtype=float).copy()
    if values.ndim != 1:
        raise ValueError("signal_values must be one-dimensional")

    nan_mask = np.isnan(values)
    if not np.any(nan_mask):
        return values

    if np.all(nan_mask):
        raise ValueError("signal contains only NaN values")

    valid_idx = np.flatnonzero(~nan_mask)
    values[nan_mask] = np.interp(np.flatnonzero(nan_mask), valid_idx, values[valid_idx])
    return values


def robust_scale(values: np.ndarray) -> float:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(1.4826 * mad + EPS)


def zscore_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values))
    if std < EPS:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def odd_window_length(length: int, minimum: int = 5) -> int:
    length = max(length, minimum)
    if length % 2 == 0:
        length += 1
    return length


def bandpass_cheby2(signal_values: np.ndarray, config: SignalPipelineConfig) -> np.ndarray:
    sos = sp_signal.cheby2(
        config.bandpass.order,
        config.bandpass.stopband_attenuation_db,
        [config.bandpass.low_hz, config.bandpass.high_hz],
        btype="bandpass",
        fs=config.sample_rate_hz,
        output="sos",
    )
    return sp_signal.sosfiltfilt(sos, signal_values)


def adaptive_lms_cancel(
    primary_signal: np.ndarray,
    reference_signal: np.ndarray,
    step_size: float = 0.05,
    taps: int = 8,
) -> np.ndarray:
    reference = np.asarray(reference_signal, dtype=float)
    if reference.ndim == 2:
        reference = np.linalg.norm(reference, axis=1)

    if reference.size == 0:
        return np.asarray(primary_signal, dtype=float).copy()

    reference = zscore_normalize(reference)
    primary = np.asarray(primary_signal, dtype=float)

    # Small resampling/rounding differences can leave ACC one or two samples shorter
    # than the primary signal. Pad or crop so the adaptive filter stays well-defined.
    if reference.size < primary.size:
        reference = np.pad(reference, (0, primary.size - reference.size), mode="edge")
    elif reference.size > primary.size:
        reference = reference[: primary.size]

    padded = np.pad(reference, (taps - 1, 0))
    weights = np.zeros(taps, dtype=float)
    cleaned = np.zeros_like(primary, dtype=float)

    for idx in range(primary.size):
        tap_input = padded[idx : idx + taps][::-1]
        if tap_input.size < taps:
            tap_input = np.pad(tap_input, (0, taps - tap_input.size), mode="edge")
        predicted_noise = float(np.dot(weights, tap_input))
        error = float(primary[idx] - predicted_noise)
        norm = float(np.dot(tap_input, tap_input) + EPS)
        weights += (step_size / norm) * error * tap_input
        cleaned[idx] = error

    return cleaned


def minimum_peak_distance_samples(config: SignalPipelineConfig) -> int:
    samples = config.sample_rate_hz * 60.0 / config.peaks.max_hr_bpm
    return max(1, int(round(samples)))


def refine_peaks(candidate_peaks: np.ndarray, signal_values: np.ndarray, minimum_distance: int) -> np.ndarray:
    if candidate_peaks.size == 0:
        return np.empty(0, dtype=int)

    order = np.argsort(signal_values[candidate_peaks])[::-1]
    kept: list[int] = []

    for peak_idx in candidate_peaks[order]:
        if all(abs(int(peak_idx) - prev) >= minimum_distance for prev in kept):
            kept.append(int(peak_idx))

    kept.sort()
    return np.asarray(kept, dtype=int)


def detect_ppg_peaks_dmm(ppg_signal: np.ndarray, config: SignalPipelineConfig) -> np.ndarray:
    smooth_window = odd_window_length(int(round(config.sample_rate_hz * 0.12)))
    smoothed = sp_signal.savgol_filter(ppg_signal, smooth_window, polyorder=3, mode="interp")

    vpg = np.gradient(smoothed)
    apg = np.gradient(vpg)
    zero_crossings = np.flatnonzero((vpg[:-1] > 0.0) & (vpg[1:] <= 0.0)) + 1

    prominence_threshold = max(
        config.peaks.min_absolute_prominence,
        config.peaks.prominence_scale * robust_scale(smoothed),
    )
    amplitude_threshold = float(np.median(smoothed) + 0.10 * robust_scale(smoothed))
    search_radius = max(1, int(round(config.peaks.refine_radius_seconds * config.sample_rate_hz)))
    minimum_distance = minimum_peak_distance_samples(config)

    candidates: list[int] = []
    for idx in zero_crossings:
        left = max(0, int(idx - search_radius))
        right = min(smoothed.size, int(idx + search_radius + 1))
        local_idx = left + int(np.argmax(smoothed[left:right]))
        curvature = float(np.mean(apg[max(0, local_idx - 1) : min(apg.size, local_idx + 2)]))
        if smoothed[local_idx] >= amplitude_threshold and curvature <= 0.0:
            candidates.append(local_idx)

    refined = refine_peaks(np.asarray(sorted(set(candidates)), dtype=int), smoothed, minimum_distance)
    if refined.size > 0:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="some peaks have a prominence of 0")
            prominences = sp_signal.peak_prominences(smoothed, refined)[0]
        refined = refined[prominences >= prominence_threshold]

    if refined.size >= 2:
        return refined

    fallback, properties = sp_signal.find_peaks(
        smoothed,
        distance=minimum_distance,
        prominence=prominence_threshold,
        height=amplitude_threshold,
    )
    return fallback.astype(int)


def detect_ecg_r_peaks(ecg_signal: np.ndarray, config: SignalPipelineConfig) -> np.ndarray:
    smooth_window = odd_window_length(int(round(config.sample_rate_hz * 0.08)))
    smoothed = sp_signal.savgol_filter(ecg_signal, smooth_window, polyorder=3, mode="interp")

    derivative = np.gradient(smoothed)
    energy = derivative ** 2
    ma_window = max(3, int(round(config.sample_rate_hz * 0.12)))
    kernel = np.ones(ma_window, dtype=float) / ma_window
    integrated = np.convolve(energy, kernel, mode="same")

    prominence_threshold = max(
        config.peaks.min_absolute_prominence,
        config.peaks.prominence_scale * robust_scale(integrated),
    )
    minimum_distance = minimum_peak_distance_samples(config)
    candidates, _ = sp_signal.find_peaks(integrated, distance=minimum_distance, prominence=prominence_threshold)

    refine_radius = max(1, int(round(config.peaks.refine_radius_seconds * config.sample_rate_hz)))
    refined: list[int] = []
    for candidate in candidates:
        left = max(0, int(candidate - refine_radius))
        right = min(smoothed.size, int(candidate + refine_radius + 1))
        refined.append(left + int(np.argmax(smoothed[left:right])))

    refined_peaks = refine_peaks(np.asarray(refined, dtype=int), smoothed, minimum_distance)
    if refined_peaks.size > 0:
        amplitudes = smoothed[refined_peaks]
        amplitude_threshold = float(np.median(smoothed) + 0.20 * robust_scale(smoothed))
        refined_peaks = refined_peaks[amplitudes >= amplitude_threshold]

    if refined_peaks.size >= 2:
        return refined_peaks

    fallback, _ = sp_signal.find_peaks(
        smoothed,
        distance=minimum_distance,
        prominence=max(config.peaks.min_absolute_prominence, 0.10 * robust_scale(smoothed)),
    )
    return fallback.astype(int)


def detect_beats(signal_values: np.ndarray, config: SignalPipelineConfig) -> np.ndarray:
    if config.signal_name.lower() == "ppg":
        return detect_ppg_peaks_dmm(signal_values, config)
    if config.signal_name.lower() == "ecg":
        return detect_ecg_r_peaks(signal_values, config)
    raise ValueError(f"Unsupported signal type: {config.signal_name}")


def interbeat_intervals_seconds(peaks: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    if peaks.size < 2:
        return np.empty(0, dtype=float)
    return np.diff(peaks).astype(float) / sample_rate_hz


def estimate_heart_band_energy_ratio(signal_values: np.ndarray, config: SignalPipelineConfig) -> float:
    frequencies, psd = sp_signal.welch(signal_values, fs=config.sample_rate_hz, nperseg=min(signal_values.size, 256))
    band_mask = (frequencies >= config.quality.heart_band_hz[0]) & (frequencies <= config.quality.heart_band_hz[1])
    total_mask = (frequencies >= config.quality.total_band_hz[0]) & (frequencies <= config.quality.total_band_hz[1])

    band_energy = float(np.trapz(psd[band_mask], frequencies[band_mask])) if np.any(band_mask) else 0.0
    total_energy = float(np.trapz(psd[total_mask], frequencies[total_mask])) if np.any(total_mask) else 0.0
    return band_energy / (total_energy + EPS)


def estimate_template_correlation(signal_values: np.ndarray, peaks: np.ndarray, sample_rate_hz: float) -> float:
    if peaks.size < 3:
        return float("nan")

    median_spacing = int(np.median(np.diff(peaks)))
    left = max(1, min(int(0.25 * median_spacing), int(round(0.30 * sample_rate_hz))))
    right = max(1, min(int(0.50 * median_spacing), int(round(0.50 * sample_rate_hz))))

    beats = []
    for peak in peaks:
        start = int(peak - left)
        end = int(peak + right)
        if start < 0 or end > signal_values.size:
            continue
        beat = signal_values[start:end]
        beats.append(zscore_normalize(beat))

    if len(beats) < 3:
        return float("nan")

    beat_matrix = np.vstack(beats)
    template = zscore_normalize(np.mean(beat_matrix, axis=0))
    correlations = []
    for beat in beat_matrix:
        corr = np.corrcoef(template, beat)[0, 1]
        if np.isfinite(corr):
            correlations.append(float(corr))

    if not correlations:
        return float("nan")
    return float(np.mean(correlations))


def sample_entropy(values: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    signal_values = np.asarray(values, dtype=float)
    if signal_values.size < m + 2:
        return float("nan")

    std = float(np.std(signal_values, ddof=1))
    if std < EPS:
        return 0.0
    tolerance = r_ratio * std

    def _match_count(length: int) -> float:
        templates = np.array([signal_values[idx : idx + length] for idx in range(signal_values.size - length + 1)])
        count = 0
        total = 0
        for idx in range(templates.shape[0] - 1):
            distances = np.max(np.abs(templates[idx + 1 :] - templates[idx]), axis=1)
            count += int(np.sum(distances <= tolerance))
            total += distances.size
        return count / max(total, 1)

    phi_m = _match_count(m)
    phi_m1 = _match_count(m + 1)
    if phi_m <= 0.0 or phi_m1 <= 0.0:
        return float("nan")
    return float(-np.log(phi_m1 / phi_m))


def signal_spectral_entropy(signal_values: np.ndarray, sample_rate_hz: float) -> float:
    frequencies, psd = sp_signal.welch(signal_values, fs=sample_rate_hz, nperseg=min(signal_values.size, 256))
    psd = psd[frequencies > 0]
    if psd.size == 0:
        return float("nan")
    probabilities = psd / (np.sum(psd) + EPS)
    entropy = -np.sum(probabilities * np.log2(probabilities + EPS))
    return float(entropy / np.log2(probabilities.size))


def extract_interval_features(ibi_seconds: np.ndarray) -> dict[str, float]:
    if ibi_seconds.size == 0:
        return {
            "ibi_count": 0.0,
            "mean_ibi_ms": float("nan"),
            "median_ibi_ms": float("nan"),
            "sdnn_ms": float("nan"),
            "rmssd_ms": float("nan"),
            "pnn50": float("nan"),
            "mean_hr_bpm": float("nan"),
            "std_hr_bpm": float("nan"),
            "cv_ibi": float("nan"),
            "sample_entropy": float("nan"),
        }

    ibi_ms = ibi_seconds * 1000.0
    diff_ms = np.diff(ibi_ms)
    heart_rate = 60.0 / np.maximum(ibi_seconds, EPS)

    sdnn = float(np.std(ibi_ms, ddof=1)) if ibi_ms.size > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(diff_ms ** 2))) if diff_ms.size > 0 else 0.0
    pnn50 = float(np.mean(np.abs(diff_ms) > 50.0)) if diff_ms.size > 0 else 0.0

    return {
        "ibi_count": float(ibi_seconds.size),
        "mean_ibi_ms": float(np.mean(ibi_ms)),
        "median_ibi_ms": float(np.median(ibi_ms)),
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "pnn50": pnn50,
        "mean_hr_bpm": float(np.mean(heart_rate)),
        "std_hr_bpm": float(np.std(heart_rate, ddof=1)) if heart_rate.size > 1 else 0.0,
        "cv_ibi": float(np.std(ibi_seconds, ddof=1) / (np.mean(ibi_seconds) + EPS)) if ibi_seconds.size > 1 else 0.0,
        "sample_entropy": sample_entropy(ibi_seconds),
    }


def evaluate_segment_quality(
    signal_values: np.ndarray,
    peaks: np.ndarray,
    acc_segment: Optional[np.ndarray],
    config: SignalPipelineConfig,
) -> dict[str, Any]:
    energy_ratio = estimate_heart_band_energy_ratio(signal_values, config)
    skewness = float(stats.skew(signal_values, bias=False))
    if not np.isfinite(skewness):
        skewness = 0.0
    template_corr = estimate_template_correlation(signal_values, peaks, config.sample_rate_hz)
    acc_variance = float(np.var(np.linalg.norm(acc_segment, axis=1))) if acc_segment is not None else float("nan")
    ibi_seconds = interbeat_intervals_seconds(peaks, config.sample_rate_hz)
    estimated_hr = float(60.0 / np.median(ibi_seconds)) if ibi_seconds.size > 0 else float("nan")

    reasons = []
    if peaks.size < config.quality.min_peak_count:
        reasons.append("too_few_peaks")
    if energy_ratio < config.quality.min_heart_band_energy_ratio:
        reasons.append("low_heart_band_energy")
    if np.isfinite(skewness) and abs(skewness) > config.quality.max_abs_skewness:
        reasons.append("skewness_out_of_range")
    if np.isfinite(template_corr) and template_corr < config.quality.min_template_correlation:
        reasons.append("low_template_correlation")
    if (
        acc_segment is not None
        and config.quality.max_acc_variance is not None
        and acc_variance > config.quality.max_acc_variance
    ):
        reasons.append("high_acc_variance")

    normalized_scores = [
        float(np.clip(energy_ratio / max(config.quality.min_heart_band_energy_ratio, EPS), 0.0, 1.0)),
        float(np.clip(1.0 - (abs(skewness) / max(config.quality.max_abs_skewness, EPS)), 0.0, 1.0)),
        0.0 if not np.isfinite(template_corr) else float(np.clip(template_corr / max(config.quality.min_template_correlation, EPS), 0.0, 1.0)),
    ]
    if config.quality.max_acc_variance is not None and np.isfinite(acc_variance):
        normalized_scores.append(float(np.clip(1.0 - (acc_variance / max(config.quality.max_acc_variance, EPS)), 0.0, 1.0)))

    accepted = len(reasons) == 0
    return {
        "peak_count": int(peaks.size),
        "heart_band_energy_ratio": float(energy_ratio),
        "signal_skewness": float(skewness),
        "template_correlation": float(template_corr) if np.isfinite(template_corr) else float("nan"),
        "acc_variance": acc_variance,
        "estimated_hr_bpm": estimated_hr,
        "quality_score": float(np.mean(normalized_scores)),
        "accepted": bool(accepted),
        "rejection_reason": "" if accepted else ";".join(reasons),
    }


def process_segment(
    signal_segment: np.ndarray,
    config: SignalPipelineConfig,
    acc_segment: Optional[np.ndarray] = None,
) -> ProcessedSegment:
    signal_values = interpolate_nan(signal_segment)
    if (
        acc_segment is not None
        and acc_segment.size > 0
        and config.segment.use_adaptive_motion_cancellation
    ):
        signal_values = adaptive_lms_cancel(signal_values, acc_segment)

    filtered_signal = bandpass_cheby2(signal_values, config)
    normalized_signal = zscore_normalize(filtered_signal) if config.segment.zscore_normalize else filtered_signal.copy()
    peaks = detect_beats(normalized_signal, config)
    ibi_seconds = interbeat_intervals_seconds(peaks, config.sample_rate_hz)

    quality_metrics = evaluate_segment_quality(normalized_signal, peaks, acc_segment, config)
    feature_metrics = extract_interval_features(ibi_seconds)
    feature_metrics["signal_spectral_entropy"] = signal_spectral_entropy(normalized_signal, config.sample_rate_hz)

    return ProcessedSegment(
        filtered_signal=filtered_signal,
        normalized_signal=normalized_signal,
        peaks=peaks,
        ibi_seconds=ibi_seconds,
        quality_metrics=quality_metrics,
        feature_metrics=feature_metrics,
    )


def process_dataframe(
    dataframe: pd.DataFrame,
    label: Optional[int],
    record_id: str,
    config: SignalPipelineConfig,
    window_label_table: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if config.signal_column not in dataframe.columns:
        raise KeyError(f"Missing signal column: {config.signal_column}")

    segment_samples = int(round(config.segment.length_seconds * config.sample_rate_hz))
    stride_samples = int(round(config.segment.stride_seconds * config.sample_rate_hz))
    if stride_samples <= 0:
        raise ValueError("stride_samples must be positive")

    signal_values = dataframe[config.signal_column].to_numpy(dtype=float)
    time_values = (
        dataframe["Time"].to_numpy(dtype=float)
        if "Time" in dataframe.columns
        else np.arange(signal_values.size, dtype=float) / config.sample_rate_hz
    )

    acc_segment_values: Optional[np.ndarray] = None
    if config.acc_columns and all(column in dataframe.columns for column in config.acc_columns):
        acc_segment_values = dataframe[list(config.acc_columns)].to_numpy(dtype=float)

    label_lookup: Optional[dict[int, dict[str, Any]]] = None
    if window_label_table is not None:
        label_lookup = (
            window_label_table.sort_values("segment_index")
            .set_index("segment_index")
            .to_dict(orient="index")
        )

    rows = []
    segments = []
    for start in range(0, signal_values.size - segment_samples + 1, stride_samples):
        end = start + segment_samples
        segment_index = int(start // stride_samples)
        label_source = "path"
        label_metadata: dict[str, Any] = {}
        if label_lookup is not None:
            label_entry = label_lookup.get(segment_index)
            if label_entry is None:
                continue
            current_label = int(label_entry["label"])
            label_source = str(label_entry.get("label_source", "window_label_csv"))
            label_metadata = {
                key: value
                for key, value in label_entry.items()
                if key not in {"label", "label_source", "use_for_training"}
            }
        elif label is not None:
            current_label = int(label)
        else:
            continue

        segment = signal_values[start:end]
        acc_segment = acc_segment_values[start:end] if acc_segment_values is not None else None
        processed = process_segment(segment, config, acc_segment=acc_segment)

        row = {
            "record_id": record_id,
            "label": int(current_label),
            "label_source": label_source,
            "signal_name": config.signal_name,
            "segment_index": segment_index,
            "start_sample": int(start),
            "end_sample": int(end),
            "start_time_sec": float(time_values[start]),
            "end_time_sec": float(time_values[end - 1]),
        }
        for meta_key, meta_value in label_metadata.items():
            row[f"label_{meta_key}"] = meta_value
        row.update(processed.quality_metrics)
        row.update(processed.feature_metrics)

        rows.append(row)
        segments.append(processed.normalized_signal.astype(np.float32))

    if not rows:
        return pd.DataFrame(), np.empty((0, segment_samples), dtype=np.float32)

    return pd.DataFrame(rows), np.stack(segments)


def infer_label_from_path(csv_path: Path) -> int:
    parent_name = csv_path.parent.name.lower()
    if "non_af" in parent_name:
        return 0
    if "af" in parent_name:
        return 1
    raise ValueError(f"Unable to infer AF label from path: {csv_path}")


def _coerce_training_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).astype(float) != 0.0
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def load_window_label_table(window_label_csv: Path) -> pd.DataFrame:
    label_df = pd.read_csv(window_label_csv)
    missing_columns = WINDOW_LABEL_REQUIRED_COLUMNS - set(label_df.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Window label CSV is missing required columns: {missing_display}. "
            "Expected at least record_id, segment_index, label."
        )

    label_df = label_df.copy()
    label_df["record_id"] = label_df["record_id"].astype(str)
    label_df["segment_index"] = label_df["segment_index"].astype(int)
    if "use_for_training" in label_df.columns:
        label_df = label_df.loc[_coerce_training_mask(label_df["use_for_training"])].copy()

    label_df = label_df.loc[label_df["label"].notna()].copy()
    label_df["label"] = label_df["label"].astype(int)
    if "label_source" not in label_df.columns:
        label_df["label_source"] = "window_label_csv"

    duplicated = label_df.duplicated(subset=["record_id", "segment_index"], keep=False)
    if duplicated.any():
        duplicate_rows = label_df.loc[duplicated, ["record_id", "segment_index"]].drop_duplicates()
        preview = duplicate_rows.head(5).to_dict(orient="records")
        raise ValueError(
            "Window label CSV contains duplicate record_id/segment_index pairs. "
            f"Examples: {preview}"
        )

    return label_df


def build_dataset_from_csvs(
    csv_paths: list[Path],
    config: SignalPipelineConfig,
    window_label_table: Optional[pd.DataFrame] = None,
    fallback_to_path_labels: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    all_rows = []
    all_segments = []
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
        label = infer_label_from_path(csv_path) if (record_window_labels is None or fallback_to_path_labels) else None

        record_rows, record_segments = process_dataframe(
            dataframe=dataframe,
            label=label,
            record_id=record_id,
            config=config,
            window_label_table=record_window_labels,
        )
        if not record_rows.empty:
            all_rows.append(record_rows)
            all_segments.append(record_segments)

    if not all_rows:
        segment_length = int(round(config.segment.length_seconds * config.sample_rate_hz))
        return pd.DataFrame(), np.empty((0, segment_length), dtype=np.float32)

    return pd.concat(all_rows, ignore_index=True), np.concatenate(all_segments, axis=0)


def save_dataset_bundle(
    summary_df: pd.DataFrame,
    segments: np.ndarray,
    output_dir: Path,
    config: SignalPipelineConfig,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{config.signal_name}_segment_summary.csv"
    accepted_path = output_dir / f"{config.signal_name}_accepted_segment_summary.csv"
    segments_path = output_dir / f"{config.signal_name}_segments.npz"
    accepted_segments_path = output_dir / f"{config.signal_name}_accepted_segments.npz"
    config_path = output_dir / f"{config.signal_name}_config.json"

    summary_df.to_csv(summary_path, index=False)
    if summary_df.empty or "accepted" not in summary_df.columns:
        accepted_mask = np.empty((0,), dtype=bool)
        summary_df.to_csv(accepted_path, index=False)
        labels = np.empty((0,), dtype=np.int8)
        quality_score = np.empty((0,), dtype=np.float32)
        start_time_sec = np.empty((0,), dtype=np.float32)
    else:
        summary_df.loc[summary_df["accepted"]].to_csv(accepted_path, index=False)
        accepted_mask = summary_df["accepted"].to_numpy(dtype=bool)
        labels = summary_df["label"].to_numpy(dtype=np.int8)
        quality_score = summary_df["quality_score"].to_numpy(dtype=np.float32)
        start_time_sec = summary_df["start_time_sec"].to_numpy(dtype=np.float32)

    np.savez_compressed(
        segments_path,
        segments=segments,
        labels=labels,
        accepted=accepted_mask,
        quality_score=quality_score,
        start_time_sec=start_time_sec,
    )
    np.savez_compressed(
        accepted_segments_path,
        segments=segments[accepted_mask],
        labels=summary_df.loc[accepted_mask, "label"].to_numpy(dtype=np.int8) if not summary_df.empty else labels,
        quality_score=summary_df.loc[accepted_mask, "quality_score"].to_numpy(dtype=np.float32)
        if not summary_df.empty
        else quality_score,
        start_time_sec=summary_df.loc[accepted_mask, "start_time_sec"].to_numpy(dtype=np.float32)
        if not summary_df.empty
        else start_time_sec,
    )
    save_config(config, config_path)

    return {
        "summary_csv": summary_path,
        "accepted_summary_csv": accepted_path,
        "segments_npz": segments_path,
        "accepted_segments_npz": accepted_segments_path,
        "config_json": config_path,
    }
