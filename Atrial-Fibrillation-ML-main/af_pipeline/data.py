from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from af_pipeline.runtime import log_stage


class PPGAugment:
    def __init__(self, signal_length: int, enable_time_warp: bool = True):
        self.signal_length = signal_length
        self.enable_time_warp = enable_time_warp

    def __call__(self, signal_values: np.ndarray) -> np.ndarray:
        x = signal_values.astype(np.float32).copy()

        if np.random.rand() < 0.9:
            x *= np.random.uniform(0.90, 1.10)

        if np.random.rand() < 0.8:
            x += np.random.normal(0.0, np.random.uniform(0.005, 0.03), size=x.shape).astype(np.float32)

        if np.random.rand() < 0.5:
            shift = np.random.randint(-32, 33)
            x = np.roll(x, shift)

        if np.random.rand() < 0.4:
            t = np.linspace(0.0, 1.0, x.size, dtype=np.float32)
            drift = np.sin(2.0 * np.pi * np.random.uniform(0.2, 1.2) * t + np.random.uniform(0, 2 * np.pi))
            x += drift.astype(np.float32) * np.random.uniform(0.01, 0.05)

        if np.random.rand() < 0.3:
            mask_len = np.random.randint(self.signal_length // 80, self.signal_length // 20)
            start = np.random.randint(0, self.signal_length - mask_len)
            x[start : start + mask_len] = float(np.mean(x))

        if self.enable_time_warp and np.random.rand() < 0.35:
            stretch = np.random.uniform(0.96, 1.04)
            idx = np.linspace(0, self.signal_length - 1, int(self.signal_length * stretch), dtype=np.float32)
            warped = np.interp(idx, np.arange(self.signal_length, dtype=np.float32), x)
            x = np.interp(
                np.linspace(0, warped.size - 1, self.signal_length, dtype=np.float32),
                np.arange(warped.size, dtype=np.float32),
                warped,
            ).astype(np.float32)

        return x


class PPGSegmentDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        records: np.ndarray,
        quality_scores: np.ndarray,
        augment: PPGAugment | None = None,
    ):
        self.signals = signals.astype(np.float32)
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.records = records
        self.quality_scores = quality_scores.astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        signal_values = self.signals[index]
        if self.augment is not None:
            signal_values = self.augment(signal_values)

        return {
            "waveform": torch.from_numpy(signal_values),
            "features": torch.from_numpy(self.features[index]),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
            "quality_score": torch.tensor(self.quality_scores[index], dtype=torch.float32),
            "record_id": self.records[index],
        }


class PhysioDataset(Dataset):
    def __init__(
        self,
        ppg_waveforms: np.ndarray,
        ppg_ibi: np.ndarray,
        student_features: np.ndarray,
        ecg_waveforms: np.ndarray,
        ecg_ibi: np.ndarray,
        resp_waveforms: np.ndarray,
        teacher_features: np.ndarray,
        aux_targets: np.ndarray,
        aux_mask: np.ndarray,
        labels: np.ndarray,
        quality_scores: np.ndarray,
        record_ids: np.ndarray,
        group_ids: np.ndarray,
        augment: PPGAugment | None = None,
    ):
        self.ppg_waveforms = ppg_waveforms.astype(np.float32)
        self.ppg_ibi = ppg_ibi.astype(np.float32)
        self.student_features = student_features.astype(np.float32)
        self.ecg_waveforms = ecg_waveforms.astype(np.float32)
        self.ecg_ibi = ecg_ibi.astype(np.float32)
        self.resp_waveforms = resp_waveforms.astype(np.float32)
        self.teacher_features = teacher_features.astype(np.float32)
        self.aux_targets = aux_targets.astype(np.float32)
        self.aux_mask = aux_mask.astype(bool)
        self.labels = labels.astype(np.float32)
        self.quality_scores = quality_scores.astype(np.float32)
        self.record_ids = record_ids
        self.group_ids = group_ids
        self.augment = augment

    def __len__(self) -> int:
        return self.ppg_waveforms.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        waveform = self.ppg_waveforms[index]
        if self.augment is not None:
            waveform = self.augment(waveform)
        return {
            "ppg_waveform": torch.from_numpy(waveform),
            "ppg_ibi": torch.from_numpy(self.ppg_ibi[index]),
            "student_features": torch.from_numpy(self.student_features[index]),
            "ecg_waveform": torch.from_numpy(self.ecg_waveforms[index]),
            "ecg_ibi": torch.from_numpy(self.ecg_ibi[index]),
            "resp_waveform": torch.from_numpy(self.resp_waveforms[index]),
            "teacher_features": torch.from_numpy(self.teacher_features[index]),
            "aux_targets": torch.from_numpy(self.aux_targets[index]),
            "aux_mask": torch.from_numpy(self.aux_mask[index]),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
            "quality_score": torch.tensor(self.quality_scores[index], dtype=torch.float32),
            "record_id": self.record_ids[index],
            "group_id": self.group_ids[index],
        }


def select_representative_segments(
    ordered_indices: np.ndarray,
    quality_scores: np.ndarray,
    max_segments: int,
) -> np.ndarray:
    if ordered_indices.size <= max_segments:
        return ordered_indices

    anchor_positions = np.linspace(0, ordered_indices.size - 1, num=max_segments)
    selected_positions = np.unique(np.round(anchor_positions).astype(np.int64))
    if selected_positions.size < max_segments:
        remaining_positions = np.setdiff1d(np.arange(ordered_indices.size), selected_positions, assume_unique=True)
        remaining_scores = quality_scores[ordered_indices[remaining_positions]]
        top_remaining = remaining_positions[np.argsort(remaining_scores)[::-1][: max_segments - selected_positions.size]]
        selected_positions = np.sort(np.concatenate([selected_positions, top_remaining]))
    selected_positions = np.sort(selected_positions[:max_segments])
    return ordered_indices[selected_positions]


def group_segments_into_bags(
    summary_df: pd.DataFrame,
    split_masks: dict[str, np.ndarray],
    group_ids: pd.Series,
    raw_quality_scores: np.ndarray,
    max_segments_per_record: int,
    eval_max_segments_per_record: int,
    max_groups_per_split: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    bags_by_split: dict[str, list[dict[str, Any]]] = {}
    start_times = pd.to_numeric(summary_df["start_time_sec"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    for split_name, mask in split_masks.items():
        split_frame = summary_df.loc[mask].copy()
        split_frame["_row_index"] = np.flatnonzero(mask)
        split_frame["_group_id"] = group_ids.loc[mask].astype(str).to_numpy()
        split_frame = split_frame.sort_values(["_group_id", "start_time_sec", "segment_index"], kind="mergesort")

        records: list[dict[str, Any]] = []
        for group_id, group in split_frame.groupby("_group_id", sort=False):
            ordered_indices = group["_row_index"].to_numpy(dtype=np.int64)
            quality_scores = raw_quality_scores[ordered_indices]
            max_segments = max_segments_per_record if split_name == "train" else eval_max_segments_per_record
            selected_indices = select_representative_segments(
                ordered_indices=ordered_indices,
                quality_scores=quality_scores,
                max_segments=max_segments,
            )
            selected_start_times = start_times[selected_indices]
            records.append(
                {
                    "group_id": str(group_id),
                    "record_id": str(group["record_id"].iloc[0]),
                    "label": float(group["label"].iloc[0]),
                    "indices": selected_indices[np.argsort(selected_start_times)],
                    "full_segment_count": int(group.shape[0]),
                    "quality_mean": float(quality_scores.mean()),
                }
            )

        if max_groups_per_split is not None and max_groups_per_split > 0 and len(records) > max_groups_per_split:
            records = records[:max_groups_per_split]
        bags_by_split[split_name] = records

    return bags_by_split


class RecordBagDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        combined_features: np.ndarray,
        quality_features: np.ndarray,
        bags: list[dict[str, Any]],
    ):
        self.signals = signals.astype(np.float32)
        self.combined_features = combined_features.astype(np.float32)
        self.quality_features = quality_features.astype(np.float32)
        self.bags = bags

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, index: int) -> dict[str, Any]:
        bag = self.bags[index]
        segment_indices = bag["indices"]
        return {
            "waveforms": self.signals[segment_indices],
            "rhythm_features": self.combined_features[segment_indices],
            "quality_features": self.quality_features[segment_indices],
            "label": bag["label"],
            "group_id": bag["group_id"],
            "record_id": bag["record_id"],
            "segment_count": len(segment_indices),
            "full_segment_count": bag["full_segment_count"],
        }


def collate_record_bags(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_segments = max(item["segment_count"] for item in batch)
    signal_length = batch[0]["waveforms"].shape[1]
    rhythm_feature_dim = batch[0]["rhythm_features"].shape[1]
    quality_feature_dim = batch[0]["quality_features"].shape[1]

    waveforms = torch.zeros(len(batch), max_segments, signal_length, dtype=torch.float32)
    rhythm_features = torch.zeros(len(batch), max_segments, rhythm_feature_dim, dtype=torch.float32)
    quality_features = torch.zeros(len(batch), max_segments, quality_feature_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_segments, dtype=torch.bool)
    labels = torch.zeros(len(batch), dtype=torch.float32)
    group_ids: list[str] = []
    record_ids: list[str] = []
    full_segment_counts = torch.zeros(len(batch), dtype=torch.int32)

    for batch_index, item in enumerate(batch):
        segment_count = item["segment_count"]
        waveforms[batch_index, :segment_count] = torch.from_numpy(item["waveforms"])
        rhythm_features[batch_index, :segment_count] = torch.from_numpy(item["rhythm_features"])
        quality_features[batch_index, :segment_count] = torch.from_numpy(item["quality_features"])
        mask[batch_index, :segment_count] = True
        labels[batch_index] = float(item["label"])
        group_ids.append(item["group_id"])
        record_ids.append(item["record_id"])
        full_segment_counts[batch_index] = int(item["full_segment_count"])

    return {
        "waveforms": waveforms,
        "rhythm_features": rhythm_features,
        "quality_features": quality_features,
        "mask": mask,
        "labels": labels,
        "group_ids": np.asarray(group_ids),
        "record_ids": np.asarray(record_ids),
        "full_segment_counts": full_segment_counts,
    }


def load_and_concat_signal_datasets(
    segments_paths: list[Path],
    summary_paths: list[Path],
) -> tuple[np.ndarray, pd.DataFrame]:
    if len(segments_paths) != len(summary_paths):
        raise ValueError("--segments-path and --summary-path must be provided the same number of times.")

    signal_blocks = []
    summary_blocks = []
    expected_signal_length = None

    for segments_path, summary_path in zip(segments_paths, summary_paths):
        log_stage(f"[load] reading NPZ: {segments_path}")
        segments_npz = np.load(segments_path)
        signals = segments_npz["segments"].astype(np.float32)
        log_stage(f"[load] NPZ ready: shape={signals.shape} dtype={signals.dtype}")
        log_stage(f"[load] reading CSV: {summary_path}")
        summary_df = pd.read_csv(summary_path)
        log_stage(f"[load] CSV ready: rows={summary_df.shape[0]} cols={summary_df.shape[1]}")
        if signals.shape[0] != summary_df.shape[0]:
            raise ValueError(
                f"Segments NPZ and summary CSV row counts do not match for {segments_path} and {summary_path}."
            )
        if expected_signal_length is None:
            expected_signal_length = signals.shape[1]
        elif signals.shape[1] != expected_signal_length:
            raise ValueError(
                "All input signal datasets must have the same segment length. "
                f"Expected {expected_signal_length}, got {signals.shape[1]} from {segments_path}."
            )
        signal_blocks.append(signals)
        summary_blocks.append(summary_df)

    merged_signals = np.concatenate(signal_blocks, axis=0)
    merged_summary = pd.concat(summary_blocks, ignore_index=True)
    log_stage(
        "[load] merged dataset ready: "
        f"signals_shape={merged_signals.shape} summary_rows={merged_summary.shape[0]}"
    )
    return merged_signals, merged_summary


def load_and_concat_multimodal_datasets(
    segments_paths: list[Path],
    summary_paths: list[Path],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    if len(segments_paths) != len(summary_paths):
        raise ValueError("--segments-path and --summary-path must be provided the same number of times.")

    array_blocks: dict[str, list[np.ndarray]] = {}
    summary_blocks = []
    expected_lengths: dict[str, tuple[int, ...]] = {}

    for segments_path, summary_path in zip(segments_paths, summary_paths):
        arrays = dict(np.load(segments_path))
        summary_df = pd.read_csv(summary_path)
        row_count = summary_df.shape[0]
        if arrays["ppg_segments"].shape[0] != row_count:
            raise ValueError(
                f"Accepted multimodal NPZ and summary CSV row counts do not match for {segments_path} and {summary_path}."
            )

        for key, values in arrays.items():
            if values.ndim >= 2 and values.shape[0] == row_count:
                trailing_shape = values.shape[1:]
                if key not in expected_lengths:
                    expected_lengths[key] = trailing_shape
                elif expected_lengths[key] != trailing_shape:
                    raise ValueError(
                        f"All multimodal datasets must agree on shape for {key}. "
                        f"Expected {expected_lengths[key]}, got {trailing_shape} from {segments_path}."
                    )
            array_blocks.setdefault(key, []).append(values)
        summary_blocks.append(summary_df)

    merged_arrays = {key: np.concatenate(value_list, axis=0) for key, value_list in array_blocks.items()}
    return merged_arrays, pd.concat(summary_blocks, ignore_index=True)

