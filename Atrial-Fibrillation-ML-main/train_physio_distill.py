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
from torch.utils.data import DataLoader, Dataset

from physio_distill_model import PhysiologyAwareDistillationNet

STUDENT_FEATURE_COLUMNS = [
    "ppg_peak_count",
    "ppg_heart_band_energy_ratio",
    "ppg_signal_skewness",
    "ppg_template_correlation",
    "ppg_estimated_hr_bpm",
    "ppg_quality_score",
    "ppg_ibi_count",
    "ppg_mean_ibi_ms",
    "ppg_median_ibi_ms",
    "ppg_sdnn_ms",
    "ppg_rmssd_ms",
    "ppg_pnn50",
    "ppg_mean_hr_bpm",
    "ppg_std_hr_bpm",
    "ppg_cv_ibi",
    "ppg_sample_entropy",
    "ppg_signal_spectral_entropy",
]

TEACHER_FEATURE_COLUMNS = [
    "resp_available",
    "ecg_peak_count",
    "ecg_heart_band_energy_ratio",
    "ecg_template_correlation",
    "ecg_estimated_hr_bpm",
    "ecg_ibi_count",
    "ecg_mean_ibi_ms",
    "ecg_sdnn_ms",
    "ecg_rmssd_ms",
    "ecg_sample_entropy",
    "timing_matched_peak_count",
    "timing_median_delay_ms",
    "timing_mean_delay_ms",
    "timing_std_delay_ms",
    "timing_ibi_mae_ms",
    "timing_ibi_corr",
    "resp_rate_bpm",
    "resp_spectral_entropy",
    "resp_ppg_amplitude_corr",
    "resp_ibi_corr",
]

AUX_TARGET_COLUMNS = [
    "timing_median_delay_ms",
    "timing_ibi_mae_ms",
    "timing_ibi_corr",
    "resp_rate_bpm",
    "resp_spectral_entropy",
    "resp_ppg_amplitude_corr",
    "resp_ibi_corr",
]

RAW_QUALITY_SCORE_COLUMN = "ppg_quality_score_raw"


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


def choose_amp(device: torch.device) -> tuple[bool, str]:
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


@dataclass
class FeatureStats:
    medians: np.ndarray
    means: np.ndarray
    stds: np.ndarray


def stats_to_jsonable(stats: FeatureStats) -> dict[str, list[float]]:
    return {
        "medians": stats.medians.tolist(),
        "means": stats.means.tolist(),
        "stds": stats.stds.tolist(),
    }


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

        return x


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
        records: np.ndarray,
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
        self.records = records
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


def create_split_masks(summary_df: pd.DataFrame, split_records: dict[str, list[str]]) -> dict[str, np.ndarray]:
    return {
        split_name: summary_df["record_id"].isin(records).to_numpy()
        for split_name, records in split_records.items()
    }


def fill_and_scale_columns(
    summary_df: pd.DataFrame,
    columns: list[str],
    split_masks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, FeatureStats]:
    features = summary_df[columns].copy()
    train_features = features.loc[split_masks["train"]]
    medians = np.nan_to_num(train_features.median(axis=0).to_numpy(dtype=np.float32), nan=0.0)
    features = features.fillna(dict(zip(columns, medians)))

    train_filled = features.loc[split_masks["train"]]
    means = np.nan_to_num(train_filled.mean(axis=0).to_numpy(dtype=np.float32), nan=0.0)
    stds = np.nan_to_num(train_filled.std(axis=0).replace(0.0, 1.0).to_numpy(dtype=np.float32), nan=1.0)
    features = (features - means) / stds

    scaled_df = summary_df.copy()
    scaled_df[columns] = features
    return scaled_df, FeatureStats(medians=medians, means=means, stds=stds)


def prepare_aux_targets(
    summary_df: pd.DataFrame,
    split_masks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, FeatureStats]:
    targets = summary_df[AUX_TARGET_COLUMNS].copy()
    train_targets = targets.loc[split_masks["train"]]
    medians = np.nan_to_num(train_targets.median(axis=0).to_numpy(dtype=np.float32), nan=0.0)
    mask = targets.notna().to_numpy(dtype=bool)
    targets = targets.fillna(dict(zip(AUX_TARGET_COLUMNS, medians)))

    train_filled = targets.loc[split_masks["train"]]
    means = np.nan_to_num(train_filled.mean(axis=0).to_numpy(dtype=np.float32), nan=0.0)
    stds = np.nan_to_num(train_filled.std(axis=0).replace(0.0, 1.0).to_numpy(dtype=np.float32), nan=1.0)
    targets = (targets - means) / stds

    normalized_df = summary_df.copy()
    normalized_df[AUX_TARGET_COLUMNS] = targets
    normalized_df[[f"{column}_valid" for column in AUX_TARGET_COLUMNS]] = mask
    return normalized_df, FeatureStats(medians=medians, means=means, stds=stds)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
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
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    candidate_thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidate_thresholds:
        y_pred = (y_prob >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        score = sensitivity + specificity
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def summarize_by_record(record_ids: np.ndarray, labels: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"record_id": record_ids, "label": labels, "prob": probs})
    return frame.groupby("record_id", as_index=False).agg(label=("label", "first"), prob=("prob", "mean"))


class DistillationLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float,
        alpha_teacher: float = 0.35,
        alpha_distill: float = 0.20,
        alpha_aux: float = 0.15,
        gamma: float = 1.5,
        label_smoothing: float = 0.02,
    ):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.alpha_teacher = alpha_teacher
        self.alpha_distill = alpha_distill
        self.alpha_aux = alpha_aux
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.aux_loss = nn.SmoothL1Loss(reduction="none")

    def _student_classification_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        quality_scores: torch.Tensor,
    ) -> torch.Tensor:
        smooth_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(
            logits,
            smooth_targets,
            reduction="none",
            pos_weight=self.pos_weight.to(logits.device),
        )
        pt = torch.exp(-bce)
        focal = ((1.0 - pt) ** self.gamma) * bce
        sample_weights = (0.6 + 0.4 * quality_scores).clamp(min=0.2, max=1.0)
        return (focal * sample_weights).mean()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        quality_scores: torch.Tensor,
        aux_targets: torch.Tensor,
        aux_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        student_cls = self._student_classification_loss(outputs["student_logit"], labels, quality_scores)
        teacher_cls = F.binary_cross_entropy_with_logits(
            outputs["teacher_logit"],
            labels,
            pos_weight=self.pos_weight.to(outputs["teacher_logit"].device),
        )

        student_embedding = F.normalize(outputs["student_embedding"], dim=1)
        teacher_embedding = F.normalize(outputs["teacher_embedding"].detach(), dim=1)
        distill = 1.0 - F.cosine_similarity(student_embedding, teacher_embedding, dim=1).mean()

        aux_residual = self.aux_loss(outputs["aux_prediction"], aux_targets)
        aux_mask = aux_mask.to(aux_residual.dtype)
        aux = (aux_residual * aux_mask).sum() / aux_mask.sum().clamp(min=1.0)

        total = student_cls + self.alpha_teacher * teacher_cls + self.alpha_distill * distill + self.alpha_aux * aux
        loss_parts = {
            "student_cls": float(student_cls.detach().item()),
            "teacher_cls": float(teacher_cls.detach().item()),
            "distill": float(distill.detach().item()),
            "aux": float(aux.detach().item()),
        }
        return total, loss_parts


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
            ppg_waveform = batch["ppg_waveform"].to(device)
            ppg_ibi = batch["ppg_ibi"].to(device)
            student_features = batch["student_features"].to(device)

            probs = []
            for shift in tta_shifts:
                shifted = torch.roll(ppg_waveform, shifts=shift, dims=1) if shift != 0 else ppg_waveform
                logits, _, _ = model.forward_student(
                    ppg_waveform=shifted,
                    ppg_ibi=ppg_ibi,
                    student_features=student_features,
                )
                probs.append(torch.sigmoid(logits))
            mean_prob = torch.stack(probs, dim=0).mean(dim=0)

            all_probs.append(mean_prob.cpu().numpy())
            all_labels.append(batch["label"].numpy())
            all_records.extend(batch["record_id"])

    return np.concatenate(all_labels), np.concatenate(all_probs), np.asarray(all_records)


def run_training_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: DistillationLoss,
    device: torch.device,
    amp_enabled: bool,
    amp_device_type: str,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_items = 0
    total_parts = {"student_cls": 0.0, "teacher_cls": 0.0, "distill": 0.0, "aux": 0.0}
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)

        ppg_waveform = batch["ppg_waveform"].to(device)
        ppg_ibi = batch["ppg_ibi"].to(device)
        student_features = batch["student_features"].to(device)
        ecg_waveform = batch["ecg_waveform"].to(device)
        ecg_ibi = batch["ecg_ibi"].to(device)
        resp_waveform = batch["resp_waveform"].to(device)
        teacher_features = batch["teacher_features"].to(device)
        aux_targets = batch["aux_targets"].to(device)
        aux_mask = batch["aux_mask"].to(device)
        labels = batch["label"].to(device)
        quality_scores = batch["quality_score"].to(device)

        with torch.amp.autocast(device_type=amp_device_type, enabled=amp_enabled):
            outputs = model(
                ppg_waveform=ppg_waveform,
                ppg_ibi=ppg_ibi,
                student_features=student_features,
                ecg_waveform=ecg_waveform,
                ecg_ibi=ecg_ibi,
                resp_waveform=resp_waveform,
                teacher_features=teacher_features,
            )
            loss, parts = loss_fn(
                outputs=outputs,
                labels=labels,
                quality_scores=quality_scores,
                aux_targets=aux_targets,
                aux_mask=aux_mask,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = ppg_waveform.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        for key, value in parts.items():
            total_parts[key] += value * batch_size

    averaged_parts = {key: value / max(total_items, 1) for key, value in total_parts.items()}
    return total_loss / max(total_items, 1), averaged_parts


def prepare_dataloaders(
    arrays: dict[str, np.ndarray],
    summary_df: pd.DataFrame,
    split_masks: dict[str, np.ndarray],
    batch_size: int,
) -> tuple[dict[str, DataLoader], dict[str, np.ndarray]]:
    loaders = {}
    label_arrays = {}

    aux_valid_columns = [f"{column}_valid" for column in AUX_TARGET_COLUMNS]
    quality_column = RAW_QUALITY_SCORE_COLUMN if RAW_QUALITY_SCORE_COLUMN in summary_df.columns else "ppg_quality_score"

    for split_name, mask in split_masks.items():
        quality_scores = summary_df.loc[mask, quality_column].to_numpy(dtype=np.float32)
        quality_scores = np.nan_to_num(quality_scores, nan=0.5, posinf=1.0, neginf=0.0)
        quality_scores = np.clip(quality_scores, 0.0, 1.0)
        dataset = PhysioDataset(
            ppg_waveforms=arrays["ppg_segments"][mask],
            ppg_ibi=arrays["ppg_ibi_sequences"][mask],
            student_features=summary_df.loc[mask, STUDENT_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
            ecg_waveforms=arrays["ecg_segments"][mask],
            ecg_ibi=arrays["ecg_ibi_sequences"][mask],
            resp_waveforms=arrays["resp_segments"][mask],
            teacher_features=summary_df.loc[mask, TEACHER_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
            aux_targets=summary_df.loc[mask, AUX_TARGET_COLUMNS].to_numpy(dtype=np.float32),
            aux_mask=summary_df.loc[mask, aux_valid_columns].to_numpy(dtype=bool),
            labels=summary_df.loc[mask, "label"].to_numpy(dtype=np.float32),
            quality_scores=quality_scores,
            records=summary_df.loc[mask, "record_id"].to_numpy(),
            augment=PPGAugment(arrays["ppg_segments"].shape[1]) if split_name == "train" else None,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=0,
        )
        label_arrays[split_name] = summary_df.loc[mask, "label"].to_numpy(dtype=np.float32)

    return loaders, label_arrays


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a physiology-aware multimodal distillation model with PPG-only inference."
    )
    parser.add_argument(
        "--segments-path",
        type=Path,
        nargs="+",
        default=[Path("artifacts/physio_distill/physio_multimodal_accepted_segments.npz")],
        help="One or more accepted multimodal NPZ paths",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        nargs="+",
        default=[Path("artifacts/physio_distill/physio_multimodal_accepted_segment_summary.csv")],
        help="One or more accepted multimodal summary CSV paths",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/physio_distill"),
        help="Directory for checkpoints and metrics",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-records", type=int, default=5)
    parser.add_argument("--test-records", type=int, default=5)
    parser.add_argument("--patience", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    amp_enabled, amp_device_type = choose_amp(device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    arrays, summary_df = load_and_concat_multimodal_datasets(args.segments_path, args.summary_path)

    split_records = stratified_record_split(
        summary_df=summary_df,
        val_record_count=args.val_records,
        test_record_count=args.test_records,
        seed=args.seed,
    )
    split_masks = create_split_masks(summary_df, split_records)
    if RAW_QUALITY_SCORE_COLUMN not in summary_df.columns and "ppg_quality_score" in summary_df.columns:
        summary_df[RAW_QUALITY_SCORE_COLUMN] = summary_df["ppg_quality_score"].astype(np.float32)

    summary_df, student_stats = fill_and_scale_columns(summary_df, STUDENT_FEATURE_COLUMNS, split_masks)
    summary_df, teacher_stats = fill_and_scale_columns(summary_df, TEACHER_FEATURE_COLUMNS, split_masks)
    summary_df, aux_stats = prepare_aux_targets(summary_df, split_masks)

    loaders, label_arrays = prepare_dataloaders(arrays, summary_df, split_masks, batch_size=args.batch_size)
    split_sizes = {split_name: int(mask.sum()) for split_name, mask in split_masks.items()}
    print(
        "dataset summary:",
        json.dumps(
            {
                "total_segments": int(summary_df.shape[0]),
                "record_count": int(summary_df["record_id"].nunique()),
                "split_sizes": split_sizes,
            }
        ),
    )

    train_pos = float(label_arrays["train"].sum())
    train_neg = float(label_arrays["train"].shape[0] - train_pos)
    pos_weight = train_neg / max(train_pos, 1.0)

    model = PhysiologyAwareDistillationNet(
        student_feature_dim=len(STUDENT_FEATURE_COLUMNS),
        teacher_feature_dim=len(TEACHER_FEATURE_COLUMNS),
        aux_target_dim=len(AUX_TARGET_COLUMNS),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = DistillationLoss(pos_weight=pos_weight)

    best_state = None
    best_epoch = 0
    best_val_score = -math.inf
    best_threshold = 0.5
    patience_counter = 0
    history = []

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, loss_parts = run_training_epoch(
            model=model,
            dataloader=loaders["train"],
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            amp_enabled=amp_enabled,
            amp_device_type=amp_device_type,
        )
        scheduler.step()

        val_y, val_prob, val_records = evaluate_model(model, loaders["val"], device=device, tta_shifts=(0, -8, 8))
        val_threshold = find_best_threshold(val_y, val_prob)
        val_metrics = compute_metrics(val_y, val_prob, threshold=val_threshold)

        record_val = summarize_by_record(val_records, val_y, val_prob)
        record_val_metrics = compute_metrics(
            record_val["label"].to_numpy(dtype=np.int64),
            record_val["prob"].to_numpy(dtype=np.float32),
            threshold=val_threshold,
        )

        score = record_val_metrics["auroc"] + record_val_metrics["f1"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_threshold": val_threshold,
                **loss_parts,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"val_record_{k}": v for k, v in record_val_metrics.items()},
            }
        )

        elapsed = time.time() - start_time
        avg_epoch_seconds = elapsed / epoch
        eta_seconds = avg_epoch_seconds * max(args.epochs - epoch, 0)
        print(
            f"epoch={epoch:02d} "
            f"loss={train_loss:.4f} "
            f"student_cls={loss_parts['student_cls']:.4f} "
            f"distill={loss_parts['distill']:.4f} "
            f"aux={loss_parts['aux']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} "
            f"val_record_auroc={record_val_metrics['auroc']:.4f} "
            f"epoch_time={format_duration(time.time() - epoch_start)} "
            f"elapsed={format_duration(elapsed)} "
            f"eta={format_duration(eta_seconds)}"
        )

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

    record_val = summarize_by_record(val_records, val_y, val_prob)
    record_test = summarize_by_record(test_records, test_y, test_prob)
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
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_threshold": best_threshold,
        "split_records": split_records,
        "student_feature_columns": STUDENT_FEATURE_COLUMNS,
        "teacher_feature_columns": TEACHER_FEATURE_COLUMNS,
        "aux_target_columns": AUX_TARGET_COLUMNS,
        "student_normalization": stats_to_jsonable(student_stats),
        "teacher_normalization": stats_to_jsonable(teacher_stats),
        "aux_normalization": stats_to_jsonable(aux_stats),
        "segment_level": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "record_level": {
            "val": record_val_metrics,
            "test": record_test_metrics,
        },
        "runtime_seconds": time.time() - start_time,
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_threshold": best_threshold,
            "split_records": split_records,
            "student_feature_columns": STUDENT_FEATURE_COLUMNS,
            "teacher_feature_columns": TEACHER_FEATURE_COLUMNS,
            "aux_target_columns": AUX_TARGET_COLUMNS,
            "student_normalization": stats_to_jsonable(student_stats),
            "teacher_normalization": stats_to_jsonable(teacher_stats),
            "aux_normalization": stats_to_jsonable(aux_stats),
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
    print("Record-level test metrics:", json.dumps(record_test_metrics, indent=2))
    print("Saved artifacts to:", args.output_dir)


if __name__ == "__main__":
    main()
