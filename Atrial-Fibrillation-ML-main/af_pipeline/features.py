from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "peak_count",
    "heart_band_energy_ratio",
    "signal_skewness",
    "template_correlation",
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

RHYTHM_CONTEXT_COLUMNS = [
    "mean_hr_bpm",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50",
    "cv_ibi",
    "sample_entropy",
    "signal_spectral_entropy",
]

QUALITY_COLUMNS = [
    "quality_score",
    "template_correlation",
    "heart_band_energy_ratio",
]

SQI_CONDITION_COLUMNS = [
    "quality_score",
    "template_correlation",
    "heart_band_energy_ratio",
    "signal_spectral_entropy",
]


@dataclass
class NormalizationStats:
    feature_medians: np.ndarray
    feature_means: np.ndarray
    feature_stds: np.ndarray


@dataclass
class FeatureStats:
    medians: np.ndarray
    means: np.ndarray
    stds: np.ndarray


@dataclass
class CombinedFeatureStats:
    means: np.ndarray
    stds: np.ndarray


def stats_to_jsonable(stats: FeatureStats) -> dict[str, list[float]]:
    return {
        "medians": stats.medians.tolist(),
        "means": stats.means.tolist(),
        "stds": stats.stds.tolist(),
    }


def fill_and_scale_features(
    summary_df: pd.DataFrame,
    split_masks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, NormalizationStats]:
    features = summary_df[FEATURE_COLUMNS].copy()
    train_features = features.loc[split_masks["train"]]

    medians = train_features.median(axis=0).to_numpy(dtype=np.float32)
    features = features.fillna(dict(zip(FEATURE_COLUMNS, medians)))

    train_filled = features.loc[split_masks["train"]]
    means = train_filled.mean(axis=0).to_numpy(dtype=np.float32)
    stds = train_filled.std(axis=0).replace(0.0, 1.0).to_numpy(dtype=np.float32)
    features = (features - means) / stds

    scaled_df = summary_df.copy()
    scaled_df[FEATURE_COLUMNS] = features
    return scaled_df, NormalizationStats(feature_medians=medians, feature_means=means, feature_stds=stds)


def fill_and_scale_columns(
    summary_df: pd.DataFrame,
    columns: list[str],
    split_masks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, FeatureStats]:
    features = summary_df[columns].copy()
    train_features = features.loc[split_masks["train"]]
    medians = train_features.median(axis=0).to_numpy(dtype=np.float32)
    features = features.fillna(dict(zip(columns, medians)))

    train_filled = features.loc[split_masks["train"]]
    means = train_filled.mean(axis=0).to_numpy(dtype=np.float32)
    stds = train_filled.std(axis=0).replace(0.0, 1.0).to_numpy(dtype=np.float32)
    features = (features - means) / stds

    scaled_df = summary_df.copy()
    scaled_df[columns] = features
    return scaled_df, FeatureStats(medians=medians, means=means, stds=stds)


def prepare_aux_targets(
    summary_df: pd.DataFrame,
    aux_target_columns: list[str],
    split_masks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, FeatureStats]:
    targets = summary_df[aux_target_columns].copy()
    train_targets = targets.loc[split_masks["train"]]
    medians = train_targets.median(axis=0).to_numpy(dtype=np.float32)
    mask = targets.notna().to_numpy(dtype=bool)
    targets = targets.fillna(dict(zip(aux_target_columns, medians)))

    train_filled = targets.loc[split_masks["train"]]
    means = train_filled.mean(axis=0).to_numpy(dtype=np.float32)
    stds = train_filled.std(axis=0).replace(0.0, 1.0).to_numpy(dtype=np.float32)
    targets = (targets - means) / stds

    normalized_df = summary_df.copy()
    normalized_df[aux_target_columns] = targets
    normalized_df[[f"{column}_valid" for column in aux_target_columns]] = mask
    return normalized_df, FeatureStats(medians=medians, means=means, stds=stds)


def make_multiscale_rhythm_features(
    summary_df: pd.DataFrame,
    group_ids: pd.Series,
    split_masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str], CombinedFeatureStats]:
    base_features = summary_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    context_source = summary_df[RHYTHM_CONTEXT_COLUMNS].to_numpy(dtype=np.float32)
    start_times = pd.to_numeric(summary_df["start_time_sec"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    group_array = group_ids.astype(str).to_numpy()

    extras = []
    extra_names = []
    for window_size, label in ((2, "60s"), (4, "120s")):
        mean_values = np.zeros((summary_df.shape[0], len(RHYTHM_CONTEXT_COLUMNS)), dtype=np.float32)
        std_values = np.zeros((summary_df.shape[0], len(RHYTHM_CONTEXT_COLUMNS)), dtype=np.float32)
        for _, positions in pd.Series(np.arange(summary_df.shape[0])).groupby(group_array):
            ordered = positions.to_numpy()
            ordered = ordered[np.argsort(start_times[ordered])]
            frame = pd.DataFrame(context_source[ordered], columns=RHYTHM_CONTEXT_COLUMNS)
            mean_block = frame.rolling(window=window_size, min_periods=1).mean().to_numpy(dtype=np.float32)
            std_block = frame.rolling(window=window_size, min_periods=1).std().fillna(0.0).to_numpy(dtype=np.float32)
            mean_values[ordered] = mean_block
            std_values[ordered] = std_block
        extras.extend([mean_values, std_values])
        extra_names.extend([f"{column}_mean_{label}" for column in RHYTHM_CONTEXT_COLUMNS])
        extra_names.extend([f"{column}_std_{label}" for column in RHYTHM_CONTEXT_COLUMNS])

    delta_120 = context_source - extras[2]
    extras.append(delta_120.astype(np.float32))
    extra_names.extend([f"{column}_delta_120s" for column in RHYTHM_CONTEXT_COLUMNS])

    combined = np.concatenate([base_features] + extras, axis=1).astype(np.float32)
    train_features = combined[split_masks["train"]]
    means = train_features.mean(axis=0).astype(np.float32)
    stds = train_features.std(axis=0).astype(np.float32)
    stds[stds == 0.0] = 1.0
    combined = (combined - means) / stds

    combined_names = list(FEATURE_COLUMNS) + extra_names
    return combined, combined_names, CombinedFeatureStats(means=means, stds=stds)


def build_quality_feature_matrix(summary_df: pd.DataFrame) -> np.ndarray:
    quality = summary_df[QUALITY_COLUMNS].copy()
    quality["quality_score"] = pd.to_numeric(quality["quality_score"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    quality["template_correlation"] = (
        pd.to_numeric(quality["template_correlation"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    )
    quality["heart_band_energy_ratio"] = (
        pd.to_numeric(quality["heart_band_energy_ratio"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    )
    return quality.to_numpy(dtype=np.float32)
