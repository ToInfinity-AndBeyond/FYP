from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from physio_distill_model import ECGTeacherNet
from ppg_hybrid_model import RhythmMorphologyFusionNet
from train_physio_distill import (
    choose_amp,
    compute_metrics,
    create_split_masks,
    fill_and_scale_columns,
    format_duration,
    get_device,
    load_and_concat_multimodal_datasets,
    save_json,
    set_seed,
    stats_to_jsonable,
)

PPG_FEATURE_COLUMNS = [
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

ECG_FEATURE_COLUMNS = [
    "ecg_peak_count",
    "ecg_heart_band_energy_ratio",
    "ecg_signal_skewness",
    "ecg_template_correlation",
    "ecg_estimated_hr_bpm",
    "ecg_quality_score",
    "ecg_ibi_count",
    "ecg_mean_ibi_ms",
    "ecg_median_ibi_ms",
    "ecg_sdnn_ms",
    "ecg_rmssd_ms",
    "ecg_pnn50",
    "ecg_mean_hr_bpm",
    "ecg_std_hr_bpm",
    "ecg_cv_ibi",
    "ecg_sample_entropy",
    "ecg_signal_spectral_entropy",
]


class PPGDistillDataset(Dataset):
    def __init__(
        self,
        ppg_waveforms: np.ndarray,
        ppg_features: np.ndarray,
        ppg_quality_scores: np.ndarray,
        ecg_waveforms: np.ndarray,
        ecg_ibi: np.ndarray,
        ecg_features: np.ndarray,
        labels: np.ndarray,
        subject_ids: np.ndarray,
    ):
        self.ppg_waveforms = ppg_waveforms.astype(np.float32)
        self.ppg_features = ppg_features.astype(np.float32)
        self.ppg_quality_scores = ppg_quality_scores.astype(np.float32)
        self.ecg_waveforms = ecg_waveforms.astype(np.float32)
        self.ecg_ibi = ecg_ibi.astype(np.float32)
        self.ecg_features = ecg_features.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        return self.ppg_waveforms.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "ppg_waveform": torch.from_numpy(self.ppg_waveforms[index]),
            "ppg_features": torch.from_numpy(self.ppg_features[index]),
            "ppg_quality_score": torch.tensor(self.ppg_quality_scores[index], dtype=torch.float32),
            "ecg_waveform": torch.from_numpy(self.ecg_waveforms[index]),
            "ecg_ibi": torch.from_numpy(self.ecg_ibi[index]),
            "ecg_features": torch.from_numpy(self.ecg_features[index]),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
            "subject_id": self.subject_ids[index],
        }


class QualityAwareDistillLoss(nn.Module):
    def __init__(self, pos_weight: float, alpha_kd: float = 0.7, temperature: float = 2.0, gamma: float = 1.5, label_smoothing: float = 0.02):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.alpha_kd = alpha_kd
        self.temperature = temperature
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def _student_cls(self, logits: torch.Tensor, targets: torch.Tensor, quality_scores: torch.Tensor) -> torch.Tensor:
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

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        quality_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        student_cls = self._student_cls(student_logits, labels, quality_scores)
        teacher_targets = torch.sigmoid(teacher_logits.detach() / self.temperature)
        kd = F.binary_cross_entropy_with_logits(student_logits / self.temperature, teacher_targets)
        total = student_cls + self.alpha_kd * (self.temperature ** 2) * kd
        return total, {
            "student_cls": float(student_cls.detach().item()),
            "kd": float(kd.detach().item()),
        }


def create_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(labels.astype(np.int64), minlength=2)
    class_weights = np.zeros(2, dtype=np.float32)
    for class_index, count in enumerate(class_counts):
        class_weights[class_index] = 1.0 / max(int(count), 1)
    sample_weights = class_weights[labels.astype(np.int64)]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def apply_saved_stats(summary_df: pd.DataFrame, columns: list[str], stats_payload: dict[str, list[float]]) -> pd.DataFrame:
    features = summary_df[columns].copy()
    medians = np.asarray(stats_payload["medians"], dtype=np.float32)
    means = np.asarray(stats_payload["means"], dtype=np.float32)
    stds = np.asarray(stats_payload["stds"], dtype=np.float32)
    features = features.fillna(dict(zip(columns, medians.tolist())))
    features = (features - means) / stds
    scaled_df = summary_df.copy()
    scaled_df[columns] = features
    return scaled_df


def evaluate_student(
    model: RhythmMorphologyFusionNet,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            ppg_waveform = batch["ppg_waveform"].to(device)
            ppg_features = batch["ppg_features"].to(device)
            logits = model(ppg_waveform, ppg_features)
            probs = torch.sigmoid(logits)
            probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["label"].numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)


def run_training_epoch(
    student_model: RhythmMorphologyFusionNet,
    teacher_model: ECGTeacherNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: QualityAwareDistillLoss,
    device: torch.device,
    amp_enabled: bool,
    amp_device_type: str,
) -> tuple[float, dict[str, float]]:
    student_model.train()
    teacher_model.eval()
    total_loss = 0.0
    total_items = 0
    part_totals = {"student_cls": 0.0, "kd": 0.0}
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        ppg_waveform = torch.nan_to_num(batch["ppg_waveform"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        ppg_features = torch.nan_to_num(batch["ppg_features"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        ppg_quality = torch.nan_to_num(batch["ppg_quality_score"].to(device), nan=0.5, posinf=1.0, neginf=0.0)
        ecg_waveform = torch.nan_to_num(batch["ecg_waveform"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        ecg_ibi = torch.nan_to_num(batch["ecg_ibi"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        ecg_features = torch.nan_to_num(batch["ecg_features"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        labels = torch.nan_to_num(batch["label"].to(device), nan=0.0, posinf=1.0, neginf=0.0)

        with torch.no_grad():
            teacher_logits, _ = teacher_model(ecg_waveform=ecg_waveform, ecg_ibi=ecg_ibi, ecg_features=ecg_features)

        with torch.amp.autocast(device_type=amp_device_type, enabled=amp_enabled):
            student_logits = student_model(ppg_waveform, ppg_features)
            loss, parts = loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                quality_scores=ppg_quality,
            )

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        if not math.isfinite(float(grad_norm)):
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.step(optimizer)
        scaler.update()

        batch_size = ppg_waveform.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        for key, value in parts.items():
            part_totals[key] += value * batch_size

    if total_items == 0:
        raise RuntimeError("All student distillation batches were skipped due to non-finite values.")
    return total_loss / total_items, {key: value / total_items for key, value in part_totals.items()}


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    candidate_thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidate_thresholds:
        metrics = compute_metrics(y_true, y_prob, float(threshold))
        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_threshold = float(threshold)
    return best_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill a MIMIC-compatible PPG student from a fixed Zenodo ECG teacher.")
    parser.add_argument("--segments-path", type=Path, nargs="+", required=True)
    parser.add_argument("--summary-path", type=Path, nargs="+", required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--alpha-kd", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    amp_enabled, amp_device_type = choose_amp(device, disable_amp=args.disable_amp)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays, summary_df = load_and_concat_multimodal_datasets(args.segments_path, args.summary_path)

    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu")
    split_groups = teacher_checkpoint["split_groups"]
    split_masks = create_split_masks(summary_df, split_groups, group_column="subject_id")
    summary_df, ppg_stats = fill_and_scale_columns(summary_df, PPG_FEATURE_COLUMNS, split_masks)
    summary_df = apply_saved_stats(summary_df, ECG_FEATURE_COLUMNS, teacher_checkpoint["normalization"])

    loaders = {}
    label_arrays = {}
    for split_name, mask in split_masks.items():
        dataset = PPGDistillDataset(
            ppg_waveforms=arrays["ppg_segments"][mask],
            ppg_features=summary_df.loc[mask, PPG_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
            ppg_quality_scores=np.clip(
                np.nan_to_num(summary_df.loc[mask, "ppg_quality_score"].to_numpy(dtype=np.float32), nan=0.5),
                0.0,
                1.0,
            ),
            ecg_waveforms=arrays["ecg_segments"][mask],
            ecg_ibi=arrays["ecg_ibi_sequences"][mask],
            ecg_features=summary_df.loc[mask, ECG_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
            labels=summary_df.loc[mask, "label"].to_numpy(dtype=np.float32),
            subject_ids=summary_df.loc[mask, "subject_id"].astype(str).to_numpy(),
        )
        sampler = None
        shuffle = split_name == "train"
        split_labels = summary_df.loc[mask, "label"].to_numpy(dtype=np.float32)
        if split_name == "train" and args.balanced_sampler:
            sampler = create_balanced_sampler(split_labels)
            shuffle = False
        loaders[split_name] = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, num_workers=0)
        label_arrays[split_name] = split_labels

    split_sizes = {split_name: int(mask.sum()) for split_name, mask in split_masks.items()}
    print(
        "dataset summary:",
        json.dumps(
            {
                "total_segments": int(summary_df.shape[0]),
                "subject_count": int(summary_df["subject_id"].astype(str).nunique()),
                "split_subjects": {split_name: len(groups) for split_name, groups in split_groups.items()},
                "split_sizes": split_sizes,
                "teacher_checkpoint": str(args.teacher_checkpoint),
            }
        ),
        flush=True,
    )

    train_pos = float(label_arrays["train"].sum())
    train_neg = float(label_arrays["train"].shape[0] - train_pos)
    pos_weight = train_neg / max(train_pos, 1.0)

    teacher_model = ECGTeacherNet(feature_dim=len(ECG_FEATURE_COLUMNS)).to(device)
    teacher_model.load_state_dict(teacher_checkpoint["model_state_dict"])
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad = False

    student_model = RhythmMorphologyFusionNet(feature_dim=len(PPG_FEATURE_COLUMNS), signal_length=arrays["ppg_segments"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = QualityAwareDistillLoss(pos_weight=pos_weight, alpha_kd=args.alpha_kd, temperature=args.temperature)

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
            student_model=student_model,
            teacher_model=teacher_model,
            dataloader=loaders["train"],
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            amp_enabled=amp_enabled,
            amp_device_type=amp_device_type,
        )
        scheduler.step()

        val_y, val_prob = evaluate_student(student_model, loaders["val"], device=device)
        val_threshold = find_best_threshold(val_y, val_prob)
        val_metrics = compute_metrics(val_y, val_prob, threshold=val_threshold)
        score = val_metrics["auroc"] + val_metrics["f1"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_threshold": val_threshold,
                **loss_parts,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

        elapsed = time.time() - start_time
        avg_epoch_seconds = elapsed / epoch
        eta_seconds = avg_epoch_seconds * max(args.epochs - epoch, 0)
        print(
            f"epoch={epoch:02d} loss={train_loss:.4f} student_cls={loss_parts['student_cls']:.4f} "
            f"kd={loss_parts['kd']:.4f} val_auroc={val_metrics['auroc']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"epoch_time={format_duration(time.time() - epoch_start)} elapsed={format_duration(elapsed)} "
            f"eta={format_duration(eta_seconds)}",
            flush=True,
        )

        if score > best_val_score:
            best_val_score = score
            best_epoch = epoch
            best_threshold = val_threshold
            best_state = copy.deepcopy(student_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("PPG student distillation did not produce a valid model state.")

    student_model.load_state_dict(best_state)
    val_y, val_prob = evaluate_student(student_model, loaders["val"], device=device)
    test_y, test_prob = evaluate_student(student_model, loaders["test"], device=device)
    val_metrics = compute_metrics(val_y, val_prob, threshold=best_threshold)
    test_metrics = compute_metrics(test_y, test_prob, threshold=best_threshold)

    experiment_summary = {
        "device": str(device),
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_threshold": best_threshold,
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "split_group_column": "subject_id",
        "split_groups": split_groups,
        "feature_columns": PPG_FEATURE_COLUMNS,
        "normalization": stats_to_jsonable(ppg_stats),
        "segment_level": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "runtime_seconds": time.time() - start_time,
    }

    torch.save(
        {
            "student_model_state_dict": student_model.state_dict(),
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "split_group_column": "subject_id",
            "split_groups": split_groups,
            "feature_columns": PPG_FEATURE_COLUMNS,
            "normalization": stats_to_jsonable(ppg_stats),
            "best_threshold": best_threshold,
        },
        args.output_dir / "best_model.pt",
    )
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    pd.DataFrame({"label": test_y, "prob": test_prob}).to_csv(args.output_dir / "test_segment_predictions.csv", index=False)
    save_json(experiment_summary, args.output_dir / "metrics.json")

    print("\nBest validation threshold:", round(best_threshold, 4), flush=True)
    print("Segment-level test metrics:", json.dumps(test_metrics, indent=2), flush=True)
    print("Saved artifacts to:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
