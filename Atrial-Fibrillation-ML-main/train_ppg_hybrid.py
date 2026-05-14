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
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from af_pipeline.data import PPGAugment, PPGSegmentDataset, load_and_concat_signal_datasets
from af_pipeline.features import FEATURE_COLUMNS, SQI_CONDITION_COLUMNS, fill_and_scale_features
from af_pipeline.losses import AsymmetricLoss, QualityAwareFocalLoss, mixup_batch
from af_pipeline.runtime import (
    choose_amp,
    compute_metrics,
    find_best_threshold,
    format_duration,
    get_device,
    load_init_checkpoint,
    log_stage,
    safe_probability_metric,
    save_json,
    set_seed,
    should_report_progress,
)
from af_pipeline.splits import (
    _parse_fold_list,
    create_metadata_fold_split_masks,
    create_random_window_split_masks,
    create_split_masks,
    infer_record_grouping,
    stratified_record_split,
    validate_split_masks,
)
from ppg_beatformer_model import QualityAwareBeatFormer
from ppg_hybrid_model import RhythmMorphologyFusionNet


PREDICTION_METADATA_COLUMNS = [
    "record_id",
    "event_id",
    "subject_id",
    "segment_index",
    "start_time_sec",
    "end_time_sec",
    "quality_score",
    "template_correlation",
    "heart_band_energy_ratio",
    "estimated_hr_bpm",
    "mean_hr_bpm",
    "std_hr_bpm",
    "sample_entropy",
    "signal_file_name",
    "patient",
    "folder_path",
    "strat_fold",
]


def feature_indices(column_names: list[str], selected_columns: list[str]) -> list[int]:
    missing_columns = [column for column in selected_columns if column not in column_names]
    if missing_columns:
        raise ValueError(f"Selected SQI condition columns are not in feature columns: {missing_columns}")
    return [column_names.index(column) for column in selected_columns]


def active_branches_for_variant(model_variant: str) -> tuple[str, ...]:
    if model_variant == "full_fusion":
        return ("time", "spectral", "feature")
    if model_variant == "waveform_only":
        return ("time",)
    if model_variant == "spectral_only":
        return ("spectral",)
    if model_variant == "feature_only":
        return ("feature",)
    raise ValueError(f"Unknown model variant: {model_variant}")


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


def build_threshold_sweep(
    labels: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.nan_to_num(np.asarray(probs, dtype=np.float32), nan=0.5, posinf=1.0, neginf=0.0)
    if thresholds is None:
        thresholds = np.unique(
            np.clip(
                np.concatenate(
                    [
                        np.linspace(0.001, 0.999, 999),
                        np.quantile(probs, np.linspace(0.0, 1.0, 501)),
                    ]
                ),
                0.0,
                1.0,
            )
        )

    order = np.argsort(probs)
    sorted_probs = probs[order]
    sorted_labels = labels[order]
    positive_prefix = np.concatenate([[0], np.cumsum(sorted_labels == 1)])
    total_positive = int(positive_prefix[-1])
    total_count = int(labels.size)

    rows = []
    for threshold in thresholds:
        below_count = int(np.searchsorted(sorted_probs, threshold, side="left"))
        fn = int(positive_prefix[below_count])
        tp = total_positive - fn
        tn = below_count - fn
        fp = (total_count - below_count) - tp

        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        f1 = (2.0 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) else 0.0

        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision),
                "f1": float(f1),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "predicted_positive": int(total_count - below_count),
                "predicted_negative": int(below_count),
            }
        )

    return pd.DataFrame.from_records(rows)


def best_threshold_row(sweep_df: pd.DataFrame, metric: str = "f1") -> dict[str, Any]:
    if sweep_df.empty:
        return {}
    best_index = sweep_df[metric].idxmax()
    row = sweep_df.loc[best_index].to_dict()
    return {key: value.item() if hasattr(value, "item") else value for key, value in row.items()}


def build_segment_prediction_frame(
    analysis_summary_df: pd.DataFrame,
    split_mask: np.ndarray,
    grouped_record_ids: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    raw_quality_scores: np.ndarray,
) -> pd.DataFrame:
    available_columns = [column for column in PREDICTION_METADATA_COLUMNS if column in analysis_summary_df.columns]
    frame = analysis_summary_df.loc[split_mask, available_columns].reset_index(drop=True).copy()
    raw_record_ids = analysis_summary_df.loc[split_mask, "record_id"].astype(str).reset_index(drop=True)
    frame["record_id"] = raw_record_ids
    frame["group_id"] = np.asarray(grouped_record_ids, dtype=str)
    frame["label"] = np.asarray(labels, dtype=np.int64)
    frame["prob"] = np.asarray(probs, dtype=np.float32)
    frame["quality_score_runtime"] = raw_quality_scores[split_mask].astype(np.float32)
    frame["predicted_label_at_0_5"] = (frame["prob"] >= 0.5).astype(np.int64)
    return frame


def attach_group_metadata(record_frame: pd.DataFrame, segment_frame: pd.DataFrame) -> pd.DataFrame:
    if record_frame.empty:
        return record_frame

    enriched = record_frame.copy()
    enriched["group_id"] = enriched["record_id"].astype(str)

    metadata_spec: dict[str, tuple[str, str]] = {"raw_record_id": ("record_id", "first")}
    for column in ("subject_id", "event_id", "signal_file_name", "patient", "folder_path", "strat_fold"):
        if column in segment_frame.columns:
            metadata_spec[column] = (column, "first")
    if "quality_score_runtime" in segment_frame.columns:
        metadata_spec["segment_quality_mean"] = ("quality_score_runtime", "mean")

    grouped_metadata = segment_frame.groupby("group_id", as_index=False).agg(**metadata_spec)
    return enriched.merge(grouped_metadata, on="group_id", how="left")


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
    sampler_positive_fraction: float,
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
            target_fractions = np.asarray(
                [1.0 - sampler_positive_fraction, sampler_positive_fraction],
                dtype=np.float32,
            )
            class_weights = np.zeros(2, dtype=np.float32)
            for class_index, count in enumerate(class_counts):
                class_weights[class_index] = target_fractions[class_index] / max(int(count), 1)
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
        "--sampler-positive-fraction",
        type=float,
        default=0.5,
        help=(
            "Target AF fraction sampled by --balanced-sampler. "
            "Use 0.5 for 1:1, 0.333333 for 1:2, or 0.25 for 1:3 AF:non-AF."
        ),
    )
    parser.add_argument(
        "--pos-weight-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the computed BCE positive-class weight.",
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
    parser.add_argument(
        "--fusion-mode",
        choices=("standard", "sqi_conditioned"),
        default="sqi_conditioned",
        help=(
            "Fusion gate variant. standard uses only branch embeddings; sqi_conditioned also feeds selected "
            "SQI features into the gate controller."
        ),
    )
    parser.add_argument(
        "--model-architecture",
        choices=("hybrid", "qa_beatformer"),
        default="hybrid",
        help="Top-level model architecture to train.",
    )
    parser.add_argument(
        "--model-variant",
        choices=("full_fusion", "waveform_only", "spectral_only", "feature_only"),
        default="full_fusion",
        help="Hybrid branch ablation variant to train. Ignored for --model-architecture qa_beatformer.",
    )
    parser.add_argument("--beatformer-max-beats", type=int, default=64)
    parser.add_argument("--beatformer-beat-length", type=int, default=128)
    parser.add_argument("--beatformer-d-model", type=int, default=128)
    parser.add_argument("--beatformer-layers", type=int, default=2)
    parser.add_argument("--beatformer-heads", type=int, default=4)
    parser.add_argument("--beatformer-dropout", type=float, default=0.1)
    parser.add_argument("--loss-type", choices=("quality_focal", "asymmetric"), default="quality_focal")
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0)
    parser.add_argument("--asl-gamma-pos", type=float, default=1.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--asl-soft-f1-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.sampler_positive_fraction < 1.0:
        raise ValueError("--sampler-positive-fraction must be between 0 and 1.")
    if args.pos_weight_scale <= 0.0:
        raise ValueError("--pos-weight-scale must be greater than 0.")

    set_seed(args.seed)
    device = get_device()
    amp_enabled, amp_device_type = choose_amp(device, disable_amp=args.disable_amp)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_stage(
        "[setup] starting training: "
        f"device={device.type} split_mode={args.split_mode} epochs={args.epochs} batch_size={args.batch_size} "
        f"amp_enabled={amp_enabled} model_architecture={args.model_architecture} model_variant={args.model_variant}"
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
    analysis_summary_df = summary_df.copy()
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
        sampler_positive_fraction=args.sampler_positive_fraction,
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
    effective_train_pos = train_pos
    effective_train_neg = train_neg
    pos_weight_source = "raw_train_distribution"
    if args.balanced_sampler and train_pos > 0.0 and train_neg > 0.0:
        effective_train_pos = label_arrays["train"].shape[0] * args.sampler_positive_fraction
        effective_train_neg = label_arrays["train"].shape[0] * (1.0 - args.sampler_positive_fraction)
        pos_weight_source = "balanced_sampler"
    pos_weight = (effective_train_neg / max(effective_train_pos, 1.0)) * args.pos_weight_scale
    log_stage(
        "[model] dataset prepared: "
        f"train_pos={int(train_pos)} train_neg={int(train_neg)} "
        f"effective_train_pos={int(effective_train_pos)} effective_train_neg={int(effective_train_neg)} "
        f"sampler_positive_fraction={args.sampler_positive_fraction:.4f} "
        f"pos_weight={pos_weight:.4f} pos_weight_scale={args.pos_weight_scale:.4f} "
        f"pos_weight_source={pos_weight_source}"
    )

    if args.model_architecture == "hybrid":
        active_branches = active_branches_for_variant(args.model_variant)
        uses_feature_branch = "feature" in active_branches
        sqi_condition_columns = (
            list(SQI_CONDITION_COLUMNS) if args.fusion_mode == "sqi_conditioned" and uses_feature_branch else []
        )
        sqi_condition_indices = feature_indices(FEATURE_COLUMNS, sqi_condition_columns)
        model_config = {
            "model_architecture": args.model_architecture,
            "feature_dim": len(FEATURE_COLUMNS),
            "signal_length": int(signals.shape[1]),
            "model_variant": args.model_variant,
            "active_branches": list(active_branches),
            "fusion_mode": args.fusion_mode,
            "sqi_condition_columns": sqi_condition_columns,
            "sqi_condition_indices": sqi_condition_indices,
        }
        log_stage(
            "[model] "
            f"architecture={args.model_architecture} variant={args.model_variant} "
            f"active_branches={','.join(active_branches)} fusion_mode={args.fusion_mode} "
            f"sqi_condition_columns={','.join(sqi_condition_columns) if sqi_condition_columns else 'none'}"
        )

        model = RhythmMorphologyFusionNet(
            feature_dim=len(FEATURE_COLUMNS),
            signal_length=signals.shape[1],
            sqi_condition_indices=sqi_condition_indices,
            active_branches=active_branches,
        ).to(device)
    else:
        model_config = {
            "model_architecture": args.model_architecture,
            "feature_dim": len(FEATURE_COLUMNS),
            "signal_length": int(signals.shape[1]),
            "max_beats": args.beatformer_max_beats,
            "beat_length": args.beatformer_beat_length,
            "d_model": args.beatformer_d_model,
            "num_layers": args.beatformer_layers,
            "num_heads": args.beatformer_heads,
            "dropout": args.beatformer_dropout,
        }
        log_stage(
            "[model] "
            f"architecture={args.model_architecture} max_beats={args.beatformer_max_beats} "
            f"beat_length={args.beatformer_beat_length} d_model={args.beatformer_d_model} "
            f"layers={args.beatformer_layers} heads={args.beatformer_heads}"
        )
        model = QualityAwareBeatFormer(
            feature_dim=len(FEATURE_COLUMNS),
            signal_length=signals.shape[1],
            max_beats=args.beatformer_max_beats,
            beat_length=args.beatformer_beat_length,
            d_model=args.beatformer_d_model,
            num_layers=args.beatformer_layers,
            num_heads=args.beatformer_heads,
            dropout=args.beatformer_dropout,
        ).to(device)
    initialization = None
    if args.init_checkpoint is not None:
        initialization = load_init_checkpoint(model, args.init_checkpoint)
        print("loaded init checkpoint:", json.dumps(initialization, indent=2), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.loss_type == "asymmetric":
        loss_fn = AsymmetricLoss(
            gamma_neg=args.asl_gamma_neg,
            gamma_pos=args.asl_gamma_pos,
            clip=args.asl_clip,
            alpha_soft_f1=args.asl_soft_f1_weight,
        )
    else:
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
            f"val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_sensitivity={val_metrics['sensitivity']:.4f} "
            f"val_specificity={val_metrics['specificity']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )
        if record_level_enabled:
            message += (
                f" val_record_accuracy={record_val_metrics['accuracy']:.4f}"
                f" val_record_auroc={record_val_metrics['auroc']:.4f}"
                f" val_record_f1={record_val_metrics['f1']:.4f}"
            )
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
    record_val = pd.DataFrame(columns=["record_id", "label", "prob", "segment_count", "quality_mean"])
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

    segment_val_predictions = build_segment_prediction_frame(
        analysis_summary_df=analysis_summary_df,
        split_mask=split_masks["val"],
        grouped_record_ids=val_records,
        labels=val_y,
        probs=val_prob,
        raw_quality_scores=raw_quality_scores,
    )
    segment_test_predictions = build_segment_prediction_frame(
        analysis_summary_df=analysis_summary_df,
        split_mask=split_masks["test"],
        grouped_record_ids=test_records,
        labels=test_y,
        probs=test_prob,
        raw_quality_scores=raw_quality_scores,
    )
    record_val = attach_group_metadata(record_val, segment_val_predictions)
    record_test = attach_group_metadata(record_test, segment_test_predictions)
    segment_val_sweep = build_threshold_sweep(val_y, val_prob)
    segment_test_sweep = build_threshold_sweep(test_y, test_prob)
    record_val_sweep = (
        build_threshold_sweep(
            record_val["label"].to_numpy(dtype=np.int64),
            record_val["prob"].to_numpy(dtype=np.float32),
        )
        if record_level_enabled
        else pd.DataFrame()
    )
    record_test_sweep = (
        build_threshold_sweep(
            record_test["label"].to_numpy(dtype=np.int64),
            record_test["prob"].to_numpy(dtype=np.float32),
        )
        if record_level_enabled
        else pd.DataFrame()
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
        "loss_type": args.loss_type,
        "loss_config": {
            "asl_gamma_neg": args.asl_gamma_neg,
            "asl_gamma_pos": args.asl_gamma_pos,
            "asl_clip": args.asl_clip,
            "asl_soft_f1_weight": args.asl_soft_f1_weight,
        },
        "balanced_sampler": args.balanced_sampler,
        "sampler_positive_fraction": args.sampler_positive_fraction,
        "pos_weight_scale": args.pos_weight_scale,
        "model_config": model_config,
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
        "threshold_sweep_best_f1": {
            "segment_val": best_threshold_row(segment_val_sweep, metric="f1"),
            "segment_test_analysis_only": best_threshold_row(segment_test_sweep, metric="f1"),
            "record_val": best_threshold_row(record_val_sweep, metric="f1") if record_level_enabled else None,
            "record_test_analysis_only": (
                best_threshold_row(record_test_sweep, metric="f1") if record_level_enabled else None
            ),
        },
        "runtime_seconds": time.time() - start_time,
    }

    segment_val_predictions["predicted_label_at_best_threshold"] = (
        segment_val_predictions["prob"] >= best_threshold
    ).astype(np.int64)
    segment_test_predictions["predicted_label_at_best_threshold"] = (
        segment_test_predictions["prob"] >= best_threshold
    ).astype(np.int64)
    if not record_val.empty:
        record_val["predicted_label_at_best_threshold"] = (record_val["prob"] >= best_threshold).astype(np.int64)
    if not record_test.empty:
        record_test["predicted_label_at_best_threshold"] = (record_test["prob"] >= best_threshold).astype(np.int64)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
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
    segment_val_predictions.to_csv(args.output_dir / "val_segment_predictions.csv", index=False)
    record_val.to_csv(args.output_dir / "val_record_predictions.csv", index=False)
    segment_val_sweep.to_csv(args.output_dir / "val_segment_threshold_sweep.csv", index=False)
    segment_test_sweep.to_csv(args.output_dir / "test_segment_threshold_sweep.csv", index=False)
    record_val_sweep.to_csv(args.output_dir / "val_record_threshold_sweep.csv", index=False)
    record_test_sweep.to_csv(args.output_dir / "test_record_threshold_sweep.csv", index=False)
    record_test.to_csv(args.output_dir / "test_record_predictions.csv", index=False)
    segment_test_predictions.to_csv(args.output_dir / "test_segment_predictions.csv", index=False)
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
