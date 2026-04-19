from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ppg_hybrid_model import RhythmMorphologyFusionNet

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_amp(device: torch.device, disable_amp: bool = False) -> tuple[bool, str]:
    if disable_amp:
        return False, device.type if device.type in {"cuda", "cpu", "mps"} else "cpu"
    if device.type == "cuda":
        return True, "cuda"
    return False, "cpu"


def format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def load_init_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("student_model_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError(
            f"Initialization checkpoint {checkpoint_path} does not contain student_model_state_dict or model_state_dict."
        )
    load_result = model.load_state_dict(state_dict, strict=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def should_report_progress(current_step: int, total_steps: int, every_steps: int) -> bool:
    if current_step <= 1 or current_step >= total_steps:
        return True
    if every_steps > 0 and current_step % every_steps == 0:
        return True
    completed_percent = int((current_step * 100) / max(total_steps, 1))
    previous_percent = int(((current_step - 1) * 100) / max(total_steps, 1))
    return completed_percent != previous_percent and completed_percent % 10 == 0


@dataclass
class NormalizationStats:
    feature_medians: np.ndarray
    feature_means: np.ndarray
    feature_stds: np.ndarray


class PPGAugment:
    def __init__(self, signal_length: int):
        self.signal_length = signal_length

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

        if np.random.rand() < 0.35:
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


def stratified_record_split(
    summary_df: pd.DataFrame,
    val_record_count: int = 5,
    test_record_count: int = 5,
    seed: int = 42,
) -> dict[str, list[str]]:
    records = (
        summary_df.groupby("record_id", as_index=False)
        .agg(label=("label", "max"))
        .reset_index(drop=True)
    )
    pos_records = records.loc[records["label"] == 1, "record_id"].tolist()
    neg_records = records.loc[records["label"] == 0, "record_id"].tolist()

    rng = random.Random(seed)
    rng.shuffle(pos_records)
    rng.shuffle(neg_records)

    total_records = len(records)
    if total_records < 3:
        raise ValueError(
            "Patient-wise split requires at least 3 unique record_id values. "
            "Use --split-mode random_windows for single-record smoke tests."
        )

    if not pos_records or not neg_records:
        shuffled_records = records["record_id"].tolist()
        rng.shuffle(shuffled_records)
        test_records = shuffled_records[:test_record_count]
        val_records = shuffled_records[test_record_count : test_record_count + val_record_count]
        train_records = shuffled_records[test_record_count + val_record_count :]
        return {
            "train": sorted(train_records),
            "val": sorted(val_records),
            "test": sorted(test_records),
        }

    test_pos = round(test_record_count * len(pos_records) / total_records)
    test_neg = test_record_count - test_pos
    val_pos = round(val_record_count * len(pos_records) / total_records)
    val_neg = val_record_count - val_pos

    test_records = pos_records[:test_pos] + neg_records[:test_neg]
    val_records = pos_records[test_pos : test_pos + val_pos] + neg_records[test_neg : test_neg + val_neg]
    train_records = pos_records[test_pos + val_pos :] + neg_records[test_neg + val_neg :]

    return {
        "train": sorted(train_records),
        "val": sorted(val_records),
        "test": sorted(test_records),
    }


def fill_and_scale_features(summary_df: pd.DataFrame, split_masks: dict[str, np.ndarray]) -> tuple[pd.DataFrame, NormalizationStats]:
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


def create_split_masks(summary_df: pd.DataFrame, split_records: dict[str, list[str]]) -> dict[str, np.ndarray]:
    return {
        split_name: summary_df["record_id"].isin(records).to_numpy()
        for split_name, records in split_records.items()
    }


def create_random_window_split_masks(
    summary_df: pd.DataFrame,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("--val-fraction + --test-fraction must be smaller than 1.")

    labels = summary_df["label"].to_numpy(dtype=np.int64)
    indices = np.arange(labels.shape[0])
    rng = np.random.default_rng(seed)

    train_mask = np.zeros(labels.shape[0], dtype=bool)
    val_mask = np.zeros(labels.shape[0], dtype=bool)
    test_mask = np.zeros(labels.shape[0], dtype=bool)

    for class_value in np.unique(labels):
        class_indices = indices[labels == class_value]
        rng.shuffle(class_indices)
        n_items = class_indices.shape[0]
        n_test = max(1, int(round(n_items * test_fraction)))
        n_val = max(1, int(round(n_items * val_fraction)))
        if n_test + n_val >= n_items:
            n_test = max(1, n_items // 5)
            n_val = max(1, n_items // 5)
        n_train = n_items - n_test - n_val
        if n_train <= 0:
            raise ValueError(
                "Not enough windows to create train/val/test splits for all classes. "
                "Reduce --val-fraction/--test-fraction or add more data."
            )

        test_idx = class_indices[:n_test]
        val_idx = class_indices[n_test : n_test + n_val]
        train_idx = class_indices[n_test + n_val :]

        test_mask[test_idx] = True
        val_mask[val_idx] = True
        train_mask[train_idx] = True

    return {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    }


def _parse_fold_list(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        normalized = part.strip()
        if not normalized:
            continue
        values.append(int(normalized))
    if not values:
        raise ValueError("Fold list must contain at least one integer.")
    return values


def create_metadata_fold_split_masks(
    summary_df: pd.DataFrame,
    train_folds: list[int],
    val_folds: list[int],
    test_folds: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    if "strat_fold" not in summary_df.columns:
        raise ValueError("split_mode=metadata_folds requires a 'strat_fold' column in the summary CSV.")

    fold_values = pd.to_numeric(summary_df["strat_fold"], errors="raise").astype(int)
    split_masks = {
        "train": fold_values.isin(train_folds).to_numpy(),
        "val": fold_values.isin(val_folds).to_numpy(),
        "test": fold_values.isin(test_folds).to_numpy(),
    }
    split_records = {
        "train": [str(value) for value in train_folds],
        "val": [str(value) for value in val_folds],
        "test": [str(value) for value in test_folds],
    }
    return split_masks, split_records


def validate_split_masks(summary_df: pd.DataFrame, split_masks: dict[str, np.ndarray]) -> None:
    for split_name, mask in split_masks.items():
        count = int(mask.sum())
        if count == 0:
            raise ValueError(f"Split '{split_name}' is empty.")
        labels = summary_df.loc[mask, "label"].to_numpy(dtype=np.int64)
        if np.unique(labels).size < 2:
            raise ValueError(
                f"Split '{split_name}' contains only one class. "
                "Use a different split configuration or add more data."
            )


def supports_record_level_metrics(summary_df: pd.DataFrame) -> bool:
    return bool(summary_df.groupby("record_id")["label"].nunique().max() <= 1)


def infer_record_grouping(summary_df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    if supports_record_level_metrics(summary_df):
        return "record_id", summary_df["record_id"].astype(str)

    if {"record_id", "event_id"}.issubset(summary_df.columns):
        event_uniqueness = summary_df.groupby(["record_id", "event_id"])["label"].nunique().max()
        if bool(event_uniqueness <= 1):
            event_values = pd.to_numeric(summary_df["event_id"], errors="coerce").astype("Int64").astype(str)
            group_ids = summary_df["record_id"].astype(str) + "::event_" + event_values
            return "record_id+event_id", group_ids

    if "signal_file_name" in summary_df.columns:
        signal_uniqueness = summary_df.groupby("signal_file_name")["label"].nunique().max()
        if bool(signal_uniqueness <= 1):
            return "signal_file_name", summary_df["signal_file_name"].astype(str)

    return None, None


def safe_probability_metric(metric_name: str, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    if np.unique(y_true).size < 2:
        return float("nan")
    if metric_name == "auroc":
        return float(roc_auc_score(y_true, y_prob))
    if metric_name == "auprc":
        return float(average_precision_score(y_true, y_prob))
    raise ValueError(f"Unsupported metric: {metric_name}")


def mixup_batch(
    waveform: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    qualities: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if alpha <= 0.0:
        return waveform, features, labels, qualities

    lam = np.random.beta(alpha, alpha)
    permutation = torch.randperm(waveform.size(0), device=waveform.device)
    mixed_waveform = lam * waveform + (1.0 - lam) * waveform[permutation]
    mixed_features = lam * features + (1.0 - lam) * features[permutation]
    mixed_labels = lam * labels + (1.0 - lam) * labels[permutation]
    mixed_qualities = lam * qualities + (1.0 - lam) * qualities[permutation]
    return mixed_waveform, mixed_features, mixed_labels, mixed_qualities


class QualityAwareFocalLoss(nn.Module):
    def __init__(self, pos_weight: float, gamma: float = 1.5, label_smoothing: float = 0.02):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        quality_scores: torch.Tensor,
    ) -> torch.Tensor:
        quality_scores = quality_scores.clamp(0.0, 1.0)
        smooth_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(
            logits,
            smooth_targets,
            reduction="none",
            pos_weight=self.pos_weight.to(logits.device),
        )
        pt = torch.exp(-bce)
        focal = ((1.0 - pt) ** self.gamma) * bce
        sample_weights = 0.6 + 0.4 * quality_scores
        return (focal * sample_weights).mean()


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": safe_probability_metric("auroc", y_true, y_prob),
        "auprc": safe_probability_metric("auprc", y_true, y_prob),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str = "balanced_accuracy",
) -> float:
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    quantile_grid = np.quantile(y_prob, np.linspace(0.0, 1.0, 501))
    dense_grid = np.linspace(0.001, 0.999, 999)
    candidate_thresholds = np.unique(np.clip(np.concatenate([dense_grid, quantile_grid]), 0.0, 1.0))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidate_thresholds:
        y_pred = (y_prob >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        if objective == "f1":
            score = float(f1_score(y_true, y_pred, zero_division=0))
        elif objective == "balanced_accuracy":
            score = sensitivity + specificity
        else:
            raise ValueError(f"Unsupported threshold objective: {objective}")
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    tta_shifts: tuple[int, ...] = (0,),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []
    all_records = []

    with torch.no_grad():
        for batch in dataloader:
            waveform = batch["waveform"].to(device)
            features = batch["features"].to(device)

            probs = []
            for shift in tta_shifts:
                shifted = torch.roll(waveform, shifts=shift, dims=1) if shift != 0 else waveform
                logits = model(shifted, features)
                probs.append(torch.sigmoid(logits))
            mean_prob = torch.stack(probs, dim=0).mean(dim=0)
            mean_prob = torch.nan_to_num(mean_prob, nan=0.5, posinf=1.0, neginf=0.0)

            all_probs.append(mean_prob.cpu().numpy())
            all_labels.append(batch["label"].numpy())
            all_records.extend(batch["record_id"])

    return np.concatenate(all_labels), np.concatenate(all_probs), np.asarray(all_records)


def summarize_by_record(
    record_ids: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    quality_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame({"record_id": record_ids, "label": labels, "prob": probs})
    if quality_scores is None:
        return frame.groupby("record_id", as_index=False).agg(label=("label", "first"), prob=("prob", "mean"))

    frame["quality_score"] = np.asarray(quality_scores, dtype=np.float32)
    frame["quality_score"] = frame["quality_score"].fillna(0.5).clip(lower=0.0, upper=1.0)

    records = []
    for record_id, group in frame.groupby("record_id", sort=False):
        weights = group["quality_score"].to_numpy(dtype=np.float32)
        probabilities = group["prob"].to_numpy(dtype=np.float32)
        if np.allclose(weights.sum(), 0.0):
            aggregated_prob = float(np.mean(probabilities))
        else:
            aggregated_prob = float(np.average(probabilities, weights=weights))
        records.append(
            {
                "record_id": record_id,
                "label": int(group["label"].iloc[0]),
                "prob": aggregated_prob,
                "segment_count": int(group.shape[0]),
                "quality_mean": float(group["quality_score"].mean()),
            }
        )

    return pd.DataFrame.from_records(records)


def run_training_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    amp_device_type: str,
    mixup_alpha: float,
    epoch_index: int,
    total_epochs: int,
    training_start_time: float,
    progress_every_batches: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    scaler = torch.amp.GradScaler(enabled=amp_enabled)
    total_batches = len(dataloader)
    epoch_start_time = time.time()
    skipped_nonfinite_batches = 0

    for batch_index, batch in enumerate(dataloader, start=1):
        waveform = torch.nan_to_num(batch["waveform"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        features = torch.nan_to_num(batch["features"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        labels = torch.nan_to_num(batch["label"].to(device), nan=0.0, posinf=1.0, neginf=0.0)
        qualities = torch.nan_to_num(batch["quality_score"].to(device), nan=0.5, posinf=1.0, neginf=0.0)

        if mixup_alpha > 0.0 and waveform.size(0) > 1:
            waveform, features, labels, qualities = mixup_batch(
                waveform, features, labels, qualities, alpha=mixup_alpha
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=amp_device_type, enabled=amp_enabled):
            logits = model(waveform, features)
            if not torch.isfinite(logits).all():
                skipped_nonfinite_batches += 1
                if skipped_nonfinite_batches <= 5:
                    print(
                        f"[train] skipped batch {batch_index}/{total_batches} due to non-finite logits",
                        flush=True,
                    )
                continue
            loss = loss_fn(logits, labels, qualities)
        if not torch.isfinite(loss):
            skipped_nonfinite_batches += 1
            if skipped_nonfinite_batches <= 5:
                print(
                    f"[train] skipped batch {batch_index}/{total_batches} due to non-finite loss",
                    flush=True,
                )
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not math.isfinite(float(grad_norm)):
            skipped_nonfinite_batches += 1
            optimizer.zero_grad(set_to_none=True)
            if skipped_nonfinite_batches <= 5:
                print(
                    f"[train] skipped batch {batch_index}/{total_batches} due to non-finite grad norm",
                    flush=True,
                )
            continue
        scaler.step(optimizer)
        scaler.update()

        batch_size = waveform.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        if should_report_progress(batch_index, total_batches, progress_every_batches):
            epoch_elapsed = time.time() - epoch_start_time
            epoch_fraction = batch_index / max(total_batches, 1)
            epoch_eta = (epoch_elapsed / max(epoch_fraction, 1e-8)) - epoch_elapsed
            training_fraction = ((epoch_index - 1) + epoch_fraction) / max(total_epochs, 1)
            total_elapsed = time.time() - training_start_time
            total_eta = (total_elapsed / max(training_fraction, 1e-8)) - total_elapsed
            running_loss = total_loss / max(total_items, 1)
            print(
                f"train_progress epoch={epoch_index:02d}/{total_epochs:02d} "
                f"batch={batch_index}/{total_batches} "
                f"epoch_pct={epoch_fraction * 100:5.1f}% "
                f"total_pct={training_fraction * 100:5.1f}% "
                f"loss={running_loss:.4f} "
                f"epoch_elapsed={format_duration(epoch_elapsed)} "
                f"epoch_eta={format_duration(epoch_eta)} "
                f"total_eta={format_duration(total_eta)}",
                flush=True,
            )

    if skipped_nonfinite_batches:
        print(f"[train] skipped non-finite batches this epoch: {skipped_nonfinite_batches}", flush=True)
    if total_items == 0:
        raise RuntimeError("All training batches were skipped due to non-finite values.")
    return total_loss / max(total_items, 1)


def prepare_dataloaders(
    signals: np.ndarray,
    summary_df: pd.DataFrame,
    raw_quality_scores: np.ndarray,
    split_masks: dict[str, np.ndarray],
    batch_size: int,
    balanced_sampler: bool,
    disable_train_augment: bool,
) -> tuple[dict[str, DataLoader], dict[str, np.ndarray]]:
    arrays = {}
    loaders = {}
    record_column = "_record_group_id" if "_record_group_id" in summary_df.columns else "record_id"

    for split_name, mask in split_masks.items():
        split_signals = signals[mask]
        split_features = summary_df.loc[mask, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        split_labels = summary_df.loc[mask, "label"].to_numpy(dtype=np.float32)
        split_records = summary_df.loc[mask, record_column].astype(str).to_numpy()
        split_qualities = raw_quality_scores[mask].astype(np.float32)

        arrays[split_name] = split_labels
        dataset = PPGSegmentDataset(
            split_signals,
            split_features,
            split_labels,
            split_records,
            split_qualities,
            augment=(
                PPGAugment(split_signals.shape[1])
                if split_name == "train" and not disable_train_augment
                else None
            ),
        )
        sampler = None
        shuffle = split_name == "train"
        if split_name == "train" and balanced_sampler:
            class_counts = np.bincount(split_labels.astype(np.int64), minlength=2)
            class_weights = np.zeros(2, dtype=np.float32)
            for class_index, count in enumerate(class_counts):
                class_weights[class_index] = 1.0 / max(int(count), 1)
            sample_weights = class_weights[split_labels.astype(np.int64)]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
        )

    return loaders, arrays


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_stage(message: str) -> None:
    print(message, flush=True)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a patient-wise PPG-only AF classifier.")
    parser.add_argument(
        "--segments-path",
        type=Path,
        nargs="+",
        default=[Path("artifacts/signal_pipeline/ppg/ppg_accepted_segments.npz")],
        help="One or more accepted PPG segments NPZ paths",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        nargs="+",
        default=[Path("artifacts/signal_pipeline/ppg/ppg_accepted_segment_summary.csv")],
        help="One or more accepted PPG summary CSV paths",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/ppg_hybrid"),
        help="Directory for checkpoints and metrics",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mixup-alpha", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-records", type=int, default=5)
    parser.add_argument("--test-records", type=int, default=5)
    parser.add_argument(
        "--split-mode",
        choices=("patient", "random_windows", "metadata_folds"),
        default="patient",
        help="Use patient-wise split for real experiments, or random window split for single-record debug runs.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--train-folds",
        type=_parse_fold_list,
        default=_parse_fold_list("0,1,2,3,4,5,6,7"),
        help="Comma-separated metadata fold ids used for training when --split-mode metadata_folds.",
    )
    parser.add_argument(
        "--val-folds",
        type=_parse_fold_list,
        default=_parse_fold_list("8"),
        help="Comma-separated metadata fold ids used for validation when --split-mode metadata_folds.",
    )
    parser.add_argument(
        "--test-folds",
        type=_parse_fold_list,
        default=_parse_fold_list("9"),
        help="Comma-separated metadata fold ids used for testing when --split-mode metadata_folds.",
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--threshold-objective",
        choices=("balanced_accuracy", "f1"),
        default="balanced_accuracy",
        help="Validation objective used to select the decision threshold.",
    )
    parser.add_argument(
        "--balanced-sampler",
        action="store_true",
        help="Use a class-balanced sampler for the training dataloader.",
    )
    parser.add_argument(
        "--disable-train-augment",
        action="store_true",
        help="Disable waveform augmentation in the training dataset.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable automatic mixed precision for more numerically stable training.",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=20,
        help="Print in-epoch training progress every N batches.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional pretrained checkpoint used to initialize the PPG hybrid model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    amp_enabled, amp_device_type = choose_amp(device, disable_amp=args.disable_amp)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_stage(
        "[setup] starting training: "
        f"device={device.type} split_mode={args.split_mode} epochs={args.epochs} batch_size={args.batch_size} "
        f"amp_enabled={amp_enabled}"
    )
    signals, summary_df = load_and_concat_signal_datasets(args.segments_path, args.summary_path)
    if not np.isfinite(signals).all():
        replaced = int((~np.isfinite(signals)).sum())
        log_stage(f"[load] replacing non-finite signal values: count={replaced}")
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
    raw_quality_scores = pd.to_numeric(summary_df["quality_score"], errors="coerce").to_numpy(dtype=np.float32)
    raw_quality_scores = np.nan_to_num(raw_quality_scores, nan=0.5, posinf=1.0, neginf=0.0)
    raw_quality_scores = np.clip(raw_quality_scores, 0.0, 1.0)

    log_stage("[split] building train/val/test split")
    if args.split_mode == "patient":
        split_records = stratified_record_split(
            summary_df=summary_df,
            val_record_count=args.val_records,
            test_record_count=args.test_records,
            seed=args.seed,
        )
        split_masks = create_split_masks(summary_df, split_records)
    elif args.split_mode == "metadata_folds":
        split_masks, split_records = create_metadata_fold_split_masks(
            summary_df=summary_df,
            train_folds=args.train_folds,
            val_folds=args.val_folds,
            test_folds=args.test_folds,
        )
    else:
        split_records = {
            "train": sorted(summary_df["record_id"].drop_duplicates().tolist()),
            "val": sorted(summary_df["record_id"].drop_duplicates().tolist()),
            "test": sorted(summary_df["record_id"].drop_duplicates().tolist()),
        }
        split_masks = create_random_window_split_masks(
            summary_df=summary_df,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    log_stage(
        "[split] split records: "
        + json.dumps({key: len(value) for key, value in split_records.items()})
    )
    validate_split_masks(summary_df, split_masks)
    record_grouping, record_group_ids = infer_record_grouping(summary_df)
    record_level_enabled = args.split_mode != "random_windows" and record_group_ids is not None
    if record_level_enabled:
        summary_df = summary_df.copy()
        summary_df["_record_group_id"] = record_group_ids.to_numpy()
        log_stage(f"[split] record-level aggregation enabled using grouping={record_grouping}")
    else:
        record_grouping = None
    log_stage("[features] filling missing values and scaling handcrafted features")
    summary_df, normalization_stats = fill_and_scale_features(summary_df, split_masks)
    feature_array = summary_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(feature_array).all():
        replaced = int((~np.isfinite(feature_array)).sum())
        log_stage(f"[features] replacing non-finite scaled feature values: count={replaced}")
        summary_df.loc[:, FEATURE_COLUMNS] = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

    log_stage("[loader] building PyTorch dataloaders")
    loaders, label_arrays = prepare_dataloaders(
        signals,
        summary_df,
        raw_quality_scores,
        split_masks,
        batch_size=args.batch_size,
        balanced_sampler=args.balanced_sampler,
        disable_train_augment=args.disable_train_augment,
    )
    split_sizes = {split_name: int(mask.sum()) for split_name, mask in split_masks.items()}
    print(
        "dataset summary:",
        json.dumps(
            {
                "total_segments": int(summary_df.shape[0]),
                "record_count": int(summary_df["record_id"].nunique()),
                "split_mode": args.split_mode,
                "split_sizes": split_sizes,
            }
        ),
        flush=True,
    )

    train_pos = float(label_arrays["train"].sum())
    train_neg = float(label_arrays["train"].shape[0] - train_pos)
    pos_weight = train_neg / max(train_pos, 1.0)
    log_stage(
        "[model] dataset prepared: "
        f"train_pos={int(train_pos)} train_neg={int(train_neg)} pos_weight={pos_weight:.4f}"
    )

    model = RhythmMorphologyFusionNet(feature_dim=len(FEATURE_COLUMNS), signal_length=signals.shape[1]).to(device)
    initialization = None
    if args.init_checkpoint is not None:
        initialization = load_init_checkpoint(model, args.init_checkpoint)
        print("loaded init checkpoint:", json.dumps(initialization, indent=2), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = QualityAwareFocalLoss(pos_weight=pos_weight)

    best_state = None
    best_epoch = 0
    best_val_score = -math.inf
    best_threshold = 0.5
    patience_counter = 0
    history = []

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = run_training_epoch(
            model=model,
            dataloader=loaders["train"],
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            amp_enabled=amp_enabled,
            amp_device_type=amp_device_type,
            mixup_alpha=args.mixup_alpha,
            epoch_index=epoch,
            total_epochs=args.epochs,
            training_start_time=start_time,
            progress_every_batches=args.progress_every_batches,
        )
        scheduler.step()

        val_y, val_prob, val_records = evaluate_model(model, loaders["val"], device=device, tta_shifts=(0, -8, 8))
        val_quality_scores = raw_quality_scores[split_masks["val"]]

        if record_level_enabled:
            record_val = summarize_by_record(
                val_records,
                val_y,
                val_prob,
                quality_scores=val_quality_scores,
            )
            val_threshold = find_best_threshold(
                record_val["label"].to_numpy(dtype=np.int64),
                record_val["prob"].to_numpy(dtype=np.float32),
                objective=args.threshold_objective,
            )
        else:
            val_threshold = find_best_threshold(
                val_y,
                val_prob,
                objective=args.threshold_objective,
            )

        val_metrics = compute_metrics(val_y, val_prob, threshold=val_threshold)

        if record_level_enabled:
            record_val_metrics = compute_metrics(
                record_val["label"].to_numpy(dtype=np.int64),
                record_val["prob"].to_numpy(dtype=np.float32),
                threshold=val_threshold,
            )
            score = record_val_metrics["auroc"] + record_val_metrics["f1"]
        else:
            record_val_metrics = {}
            score = val_metrics["auroc"] + val_metrics["f1"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_threshold": val_threshold,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"val_record_{k}": v for k, v in record_val_metrics.items()},
            }
        )

        message = (
            f"epoch={epoch:02d} "
            f"loss={train_loss:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )
        if record_level_enabled:
            message += f" val_record_auroc={record_val_metrics['auroc']:.4f}"
        elapsed = time.time() - start_time
        avg_epoch_seconds = elapsed / epoch
        eta_seconds = avg_epoch_seconds * max(args.epochs - epoch, 0)
        message += (
            f" epoch_time={format_duration(time.time() - epoch_start)}"
            f" elapsed={format_duration(elapsed)}"
            f" eta={format_duration(eta_seconds)}"
        )
        print(message)

        if score > best_val_score:
            best_val_score = score
            best_epoch = epoch
            best_threshold = val_threshold
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)

    val_y, val_prob, val_records = evaluate_model(model, loaders["val"], device=device, tta_shifts=(0, -8, 8))
    test_y, test_prob, test_records = evaluate_model(model, loaders["test"], device=device, tta_shifts=(0, -8, 8))

    val_metrics = compute_metrics(val_y, val_prob, threshold=best_threshold)
    test_metrics = compute_metrics(test_y, test_prob, threshold=best_threshold)

    record_val_metrics = None
    record_test_metrics = None
    record_test = pd.DataFrame(columns=["record_id", "label", "prob", "segment_count", "quality_mean"])
    if record_level_enabled:
        val_quality_scores = raw_quality_scores[split_masks["val"]]
        test_quality_scores = raw_quality_scores[split_masks["test"]]
        record_val = summarize_by_record(
            val_records,
            val_y,
            val_prob,
            quality_scores=val_quality_scores,
        )
        record_test = summarize_by_record(
            test_records,
            test_y,
            test_prob,
            quality_scores=test_quality_scores,
        )
        record_val_metrics = compute_metrics(
            record_val["label"].to_numpy(dtype=np.int64),
            record_val["prob"].to_numpy(dtype=np.float32),
            threshold=best_threshold,
        )
        record_test_metrics = compute_metrics(
            record_test["label"].to_numpy(dtype=np.int64),
            record_test["prob"].to_numpy(dtype=np.float32),
            threshold=best_threshold,
        )

    experiment_summary = {
        "device": str(device),
        "split_mode": args.split_mode,
        "record_level_supported": record_level_enabled,
        "record_level_grouping": record_grouping,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_threshold": best_threshold,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        "initialization": initialization,
        "split_records": split_records,
        "threshold_objective": args.threshold_objective,
        "feature_columns": FEATURE_COLUMNS,
        "normalization": {
            "feature_medians": normalization_stats.feature_medians.tolist(),
            "feature_means": normalization_stats.feature_means.tolist(),
            "feature_stds": normalization_stats.feature_stds.tolist(),
        },
        "segment_level": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "record_level": None
        if not record_level_enabled
        else {
            "val": record_val_metrics,
            "test": record_test_metrics,
        },
        "runtime_seconds": time.time() - start_time,
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "best_threshold": best_threshold,
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
            "split_records": split_records,
            "normalization": {
                "feature_medians": normalization_stats.feature_medians,
                "feature_means": normalization_stats.feature_means,
                "feature_stds": normalization_stats.feature_stds,
            },
        },
        args.output_dir / "best_model.pt",
    )

    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    record_test.to_csv(args.output_dir / "test_record_predictions.csv", index=False)
    pd.DataFrame({"record_id": test_records, "label": test_y, "prob": test_prob}).to_csv(
        args.output_dir / "test_segment_predictions.csv",
        index=False,
    )
    save_json(experiment_summary, args.output_dir / "metrics.json")

    print("\nBest validation threshold:", round(best_threshold, 4))
    print("Segment-level test metrics:", json.dumps(test_metrics, indent=2))
    if record_level_enabled:
        print("Record-level test metrics:", json.dumps(record_test_metrics, indent=2))
    else:
        print("Record-level test metrics: skipped (record labels inconsistent or random-window debug split)")
    print("Saved artifacts to:", args.output_dir)


if __name__ == "__main__":
    main()
