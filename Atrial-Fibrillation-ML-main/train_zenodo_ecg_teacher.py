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
from train_physio_distill import (
    choose_amp,
    compute_metrics,
    create_split_masks,
    fill_and_scale_columns,
    find_best_threshold,
    format_duration,
    get_device,
    load_and_concat_multimodal_datasets,
    save_json,
    set_seed,
    stats_to_jsonable,
    stratified_group_split,
)

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


class ECGTeacherDataset(Dataset):
    def __init__(
        self,
        ecg_waveforms: np.ndarray,
        ecg_ibi: np.ndarray,
        ecg_features: np.ndarray,
        labels: np.ndarray,
        quality_scores: np.ndarray,
        subject_ids: np.ndarray,
    ):
        self.ecg_waveforms = ecg_waveforms.astype(np.float32)
        self.ecg_ibi = ecg_ibi.astype(np.float32)
        self.ecg_features = ecg_features.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.quality_scores = quality_scores.astype(np.float32)
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        return self.ecg_waveforms.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "ecg_waveform": torch.from_numpy(self.ecg_waveforms[index]),
            "ecg_ibi": torch.from_numpy(self.ecg_ibi[index]),
            "ecg_features": torch.from_numpy(self.ecg_features[index]),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
            "quality_score": torch.tensor(self.quality_scores[index], dtype=torch.float32),
            "subject_id": self.subject_ids[index],
        }


class QualityAwareBinaryLoss(nn.Module):
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


def evaluate_model(
    model: ECGTeacherNet,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            ecg_waveform = batch["ecg_waveform"].to(device)
            ecg_ibi = batch["ecg_ibi"].to(device)
            ecg_features = batch["ecg_features"].to(device)
            logits, _ = model(ecg_waveform=ecg_waveform, ecg_ibi=ecg_ibi, ecg_features=ecg_features)
            probs = torch.sigmoid(logits)
            probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["label"].numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)


def run_training_epoch(
    model: ECGTeacherNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: QualityAwareBinaryLoss,
    device: torch.device,
    amp_enabled: bool,
    amp_device_type: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        ecg_waveform = torch.nan_to_num(batch["ecg_waveform"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        ecg_ibi = torch.nan_to_num(batch["ecg_ibi"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        ecg_features = torch.nan_to_num(batch["ecg_features"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        labels = torch.nan_to_num(batch["label"].to(device), nan=0.0, posinf=1.0, neginf=0.0)
        quality_scores = torch.nan_to_num(batch["quality_score"].to(device), nan=0.5, posinf=1.0, neginf=0.0)

        with torch.amp.autocast(device_type=amp_device_type, enabled=amp_enabled):
            logits, _ = model(ecg_waveform=ecg_waveform, ecg_ibi=ecg_ibi, ecg_features=ecg_features)
            loss = loss_fn(logits, labels, quality_scores)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not math.isfinite(float(grad_norm)):
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.step(optimizer)
        scaler.update()

        batch_size = ecg_waveform.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    if total_items == 0:
        raise RuntimeError("All ECG teacher batches were skipped due to non-finite values.")
    return total_loss / total_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ECG-only teacher on Zenodo multimodal bundles.")
    parser.add_argument("--segments-path", type=Path, nargs="+", required=True)
    parser.add_argument("--summary-path", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-subjects", type=int, default=5)
    parser.add_argument("--test-subjects", type=int, default=5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--threshold-objective",
        choices=("balanced_accuracy", "f1"),
        default="f1",
    )
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--ssl-checkpoint", type=Path, default=None)
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

    split_groups = stratified_group_split(
        summary_df=summary_df,
        group_column="subject_id",
        val_group_count=args.val_subjects,
        test_group_count=args.test_subjects,
        seed=args.seed,
    )
    split_masks = create_split_masks(summary_df, split_groups, group_column="subject_id")
    summary_df, ecg_stats = fill_and_scale_columns(summary_df, ECG_FEATURE_COLUMNS, split_masks)

    loaders = {}
    label_arrays = {}
    for split_name, mask in split_masks.items():
        dataset = ECGTeacherDataset(
            ecg_waveforms=arrays["ecg_segments"][mask],
            ecg_ibi=arrays["ecg_ibi_sequences"][mask],
            ecg_features=summary_df.loc[mask, ECG_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
            labels=summary_df.loc[mask, "label"].to_numpy(dtype=np.float32),
            quality_scores=np.clip(
                np.nan_to_num(summary_df.loc[mask, "ecg_quality_score"].to_numpy(dtype=np.float32), nan=0.5),
                0.0,
                1.0,
            ),
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
            }
        ),
        flush=True,
    )

    train_pos = float(label_arrays["train"].sum())
    train_neg = float(label_arrays["train"].shape[0] - train_pos)
    pos_weight = train_neg / max(train_pos, 1.0)

    model = ECGTeacherNet(feature_dim=len(ECG_FEATURE_COLUMNS)).to(device)
    ssl_initialization = None
    if args.ssl_checkpoint is not None:
        checkpoint = torch.load(args.ssl_checkpoint, map_location="cpu")
        ecg_state = checkpoint.get("ecg_encoder_state_dict")
        if ecg_state is not None:
            load_result = model.ecg_encoder.load_state_dict(ecg_state, strict=False)
            ssl_initialization = {
                "checkpoint_path": str(args.ssl_checkpoint),
                "missing_keys": list(load_result.missing_keys),
                "unexpected_keys": list(load_result.unexpected_keys),
            }
            print("loaded SSL ECG encoder:", json.dumps(ssl_initialization, indent=2), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = QualityAwareBinaryLoss(pos_weight=pos_weight)

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
        )
        scheduler.step()

        val_y, val_prob = evaluate_model(model, loaders["val"], device=device)
        val_threshold = find_best_threshold(val_y, val_prob, objective=args.threshold_objective)
        val_metrics = compute_metrics(val_y, val_prob, threshold=val_threshold)
        score = val_metrics["auroc"] + val_metrics["f1"]

        history.append({"epoch": epoch, "train_loss": train_loss, "val_threshold": val_threshold, **{f"val_{k}": v for k, v in val_metrics.items()}})

        elapsed = time.time() - start_time
        avg_epoch_seconds = elapsed / epoch
        eta_seconds = avg_epoch_seconds * max(args.epochs - epoch, 0)
        print(
            f"epoch={epoch:02d} loss={train_loss:.4f} val_auroc={val_metrics['auroc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} epoch_time={format_duration(time.time() - epoch_start)} "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta_seconds)}",
            flush=True,
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
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("ECG teacher training did not produce a valid model state.")

    model.load_state_dict(best_state)
    val_y, val_prob = evaluate_model(model, loaders["val"], device=device)
    test_y, test_prob = evaluate_model(model, loaders["test"], device=device)
    val_metrics = compute_metrics(val_y, val_prob, threshold=best_threshold)
    test_metrics = compute_metrics(test_y, test_prob, threshold=best_threshold)

    experiment_summary = {
        "device": str(device),
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_threshold": best_threshold,
        "threshold_objective": args.threshold_objective,
        "split_group_column": "subject_id",
        "split_groups": split_groups,
        "feature_columns": ECG_FEATURE_COLUMNS,
        "normalization": stats_to_jsonable(ecg_stats),
        "ssl_checkpoint": str(args.ssl_checkpoint) if args.ssl_checkpoint is not None else None,
        "ssl_initialization": ssl_initialization,
        "segment_level": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "runtime_seconds": time.time() - start_time,
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "split_group_column": "subject_id",
            "split_groups": split_groups,
            "feature_columns": ECG_FEATURE_COLUMNS,
            "normalization": stats_to_jsonable(ecg_stats),
            "best_threshold": best_threshold,
            "ssl_checkpoint": str(args.ssl_checkpoint) if args.ssl_checkpoint is not None else None,
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
