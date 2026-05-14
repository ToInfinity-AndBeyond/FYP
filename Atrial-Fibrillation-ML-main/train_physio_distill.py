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

from af_pipeline.data import PPGAugment, PhysioDataset, load_and_concat_multimodal_datasets
from af_pipeline.features import fill_and_scale_columns, prepare_aux_targets, stats_to_jsonable
from af_pipeline.losses import DistillationLoss
from af_pipeline.runtime import (
    choose_amp,
    compute_metrics,
    find_best_threshold,
    format_duration,
    get_device,
    save_json,
    set_seed,
)
from af_pipeline.splits import (
    _parse_fold_list,
    choose_split_group_column,
    create_metadata_fold_split_masks,
    create_split_masks,
    stratified_group_split,
    validate_split_masks,
)
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
def summarize_by_group(group_ids: np.ndarray, labels: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"group_id": group_ids, "label": labels, "prob": probs})
    return frame.groupby("group_id", as_index=False).agg(label=("label", "first"), prob=("prob", "mean"))
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    tta_shifts: tuple[int, ...] = (0,),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []
    all_records = []
    all_groups = []

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
            mean_prob = torch.nan_to_num(mean_prob, nan=0.5, posinf=1.0, neginf=0.0)

            all_probs.append(mean_prob.cpu().numpy())
            all_labels.append(batch["label"].numpy())
            all_records.extend(batch["record_id"])
            all_groups.extend(batch["group_id"])

    return (
        np.concatenate(all_labels),
        np.concatenate(all_probs),
        np.asarray(all_records),
        np.asarray(all_groups),
    )


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
    skipped_nonfinite_batches = 0

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)

        ppg_waveform = torch.nan_to_num(batch["ppg_waveform"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        ppg_ibi = torch.nan_to_num(batch["ppg_ibi"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        student_features = torch.nan_to_num(batch["student_features"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        ecg_waveform = torch.nan_to_num(batch["ecg_waveform"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        ecg_ibi = torch.nan_to_num(batch["ecg_ibi"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        resp_waveform = torch.nan_to_num(batch["resp_waveform"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        teacher_features = torch.nan_to_num(batch["teacher_features"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        aux_targets = torch.nan_to_num(batch["aux_targets"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        aux_mask = batch["aux_mask"].to(device)
        labels = torch.nan_to_num(batch["label"].to(device), nan=0.0, posinf=1.0, neginf=0.0)
        quality_scores = torch.nan_to_num(batch["quality_score"].to(device), nan=0.5, posinf=1.0, neginf=0.0)

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
        if not torch.isfinite(loss):
            skipped_nonfinite_batches += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not math.isfinite(float(grad_norm)):
            skipped_nonfinite_batches += 1
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue
        scaler.step(optimizer)
        scaler.update()

        batch_size = ppg_waveform.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        for key, value in parts.items():
            total_parts[key] += value * batch_size

    if skipped_nonfinite_batches:
        print(f"[train] skipped non-finite batches this epoch: {skipped_nonfinite_batches}", flush=True)
    if total_items == 0:
        raise RuntimeError("All training batches were skipped due to non-finite values.")
    averaged_parts = {key: value / max(total_items, 1) for key, value in total_parts.items()}
    return total_loss / max(total_items, 1), averaged_parts


def prepare_dataloaders(
    arrays: dict[str, np.ndarray],
    summary_df: pd.DataFrame,
    split_masks: dict[str, np.ndarray],
    batch_size: int,
    balanced_sampler: bool,
    disable_train_augment: bool,
    group_column: str,
) -> tuple[dict[str, DataLoader], dict[str, np.ndarray]]:
    loaders = {}
    label_arrays = {}

    aux_valid_columns = [f"{column}_valid" for column in AUX_TARGET_COLUMNS]

    for split_name, mask in split_masks.items():
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
            quality_scores=np.clip(
                np.nan_to_num(
                    summary_df.loc[mask, "ppg_quality_score"].to_numpy(dtype=np.float32),
                    nan=0.5,
                    posinf=1.0,
                    neginf=0.0,
                ),
                0.0,
                1.0,
            ),
            record_ids=summary_df.loc[mask, "record_id"].to_numpy(),
            group_ids=summary_df.loc[mask, group_column].astype(str).to_numpy(),
            augment=(
                PPGAugment(arrays["ppg_segments"].shape[1], enable_time_warp=False)
                if split_name == "train" and not disable_train_augment
                else None
            ),
        )
        sampler = None
        shuffle = split_name == "train"
        if split_name == "train" and balanced_sampler:
            split_labels = summary_df.loc[mask, "label"].to_numpy(dtype=np.int64)
            class_counts = np.bincount(split_labels, minlength=2)
            class_weights = np.zeros(2, dtype=np.float32)
            for class_index, count in enumerate(class_counts):
                class_weights[class_index] = 1.0 / max(int(count), 1)
            sample_weights = class_weights[split_labels]
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
        label_arrays[split_name] = summary_df.loc[mask, "label"].to_numpy(dtype=np.float32)

    return loaders, label_arrays


def load_ssl_pretrained_encoders(model: PhysiologyAwareDistillationNet, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    loaded: dict[str, Any] = {"checkpoint_path": str(checkpoint_path)}

    ppg_state = checkpoint.get("ppg_encoder_state_dict")
    ecg_state = checkpoint.get("ecg_encoder_state_dict")
    if ppg_state is None and ecg_state is None:
        raise ValueError(
            f"SSL checkpoint {checkpoint_path} does not contain ppg_encoder_state_dict/ecg_encoder_state_dict."
        )

    if ppg_state is not None:
        student_load = model.student_morph.load_state_dict(ppg_state, strict=False)
        loaded["student_morph_missing"] = list(student_load.missing_keys)
        loaded["student_morph_unexpected"] = list(student_load.unexpected_keys)
    if ecg_state is not None:
        teacher_load = model.teacher_ecg.load_state_dict(ecg_state, strict=False)
        loaded["teacher_ecg_missing"] = list(teacher_load.missing_keys)
        loaded["teacher_ecg_unexpected"] = list(teacher_load.unexpected_keys)

    return loaded
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
    parser.add_argument(
        "--split-mode",
        choices=("group", "metadata_folds"),
        default="group",
        help="Use a fresh stratified group split or the strat_fold metadata from the summary CSV.",
    )
    parser.add_argument(
        "--train-folds",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated folds for training when --split-mode metadata_folds is used.",
    )
    parser.add_argument(
        "--val-folds",
        default="8",
        help="Comma-separated folds for validation when --split-mode metadata_folds is used.",
    )
    parser.add_argument(
        "--test-folds",
        default="9",
        help="Comma-separated folds for testing when --split-mode metadata_folds is used.",
    )
    parser.add_argument(
        "--split-group-column",
        default="auto",
        help="Grouping column used for train/val/test splitting. Use 'auto' to infer the safest option.",
    )
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--threshold-objective",
        choices=("balanced_accuracy", "f1"),
        default="f1",
        help="Validation objective used to pick the classification threshold.",
    )
    parser.add_argument(
        "--balanced-sampler",
        action="store_true",
        help="Use a class-balanced sampler for train batches.",
    )
    parser.add_argument(
        "--disable-train-augment",
        action="store_true",
        help="Disable PPG augmentation during training.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable automatic mixed precision for more stable training.",
    )
    parser.add_argument(
        "--ssl-checkpoint",
        type=Path,
        default=None,
        help="Optional ECG-PPG SSL checkpoint used to initialize the student PPG and teacher ECG encoders.",
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

    arrays, summary_df = load_and_concat_multimodal_datasets(args.segments_path, args.summary_path)

    split_group_column = choose_split_group_column(summary_df, args.split_group_column)
    if args.split_mode == "metadata_folds":
        split_masks, split_groups = create_metadata_fold_split_masks(
            summary_df=summary_df,
            train_folds=_parse_fold_list(args.train_folds),
            val_folds=_parse_fold_list(args.val_folds),
            test_folds=_parse_fold_list(args.test_folds),
        )
        validate_split_masks(summary_df, split_masks)
    else:
        split_groups = stratified_group_split(
            summary_df=summary_df,
            group_column=split_group_column,
            val_group_count=args.val_records,
            test_group_count=args.test_records,
            seed=args.seed,
        )
        split_masks = create_split_masks(summary_df, split_groups, group_column=split_group_column)
    grouped_metrics_supported = bool((summary_df.groupby(split_group_column)["label"].nunique() <= 1).all())

    summary_df, student_stats = fill_and_scale_columns(summary_df, STUDENT_FEATURE_COLUMNS, split_masks)
    summary_df, teacher_stats = fill_and_scale_columns(summary_df, TEACHER_FEATURE_COLUMNS, split_masks)
    summary_df, aux_stats = prepare_aux_targets(summary_df, AUX_TARGET_COLUMNS, split_masks)

    loaders, label_arrays = prepare_dataloaders(
        arrays,
        summary_df,
        split_masks,
        batch_size=args.batch_size,
        balanced_sampler=args.balanced_sampler,
        disable_train_augment=args.disable_train_augment,
        group_column=split_group_column,
    )
    split_sizes = {split_name: int(mask.sum()) for split_name, mask in split_masks.items()}
    print(
        "dataset summary:",
        json.dumps(
            {
                "total_segments": int(summary_df.shape[0]),
                "record_count": int(summary_df["record_id"].nunique()),
                "split_group_column": split_group_column,
                "group_count": int(summary_df[split_group_column].astype(str).nunique()),
                "group_split_sizes": {split_name: len(groups) for split_name, groups in split_groups.items()},
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
    ssl_initialization = None
    if args.ssl_checkpoint is not None:
        ssl_initialization = load_ssl_pretrained_encoders(model, args.ssl_checkpoint)
        print("loaded SSL encoders:", json.dumps(ssl_initialization, indent=2), flush=True)
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

        val_y, val_prob, val_records, val_groups = evaluate_model(
            model,
            loaders["val"],
            device=device,
            tta_shifts=(0, -8, 8),
        )
        val_threshold = find_best_threshold(
            val_y,
            val_prob,
            objective=args.threshold_objective,
        )
        val_metrics = compute_metrics(val_y, val_prob, threshold=val_threshold)

        grouped_val_metrics: dict[str, float] = {}
        if grouped_metrics_supported:
            grouped_val = summarize_by_group(val_groups, val_y, val_prob)
            grouped_val_metrics = compute_metrics(
                grouped_val["label"].to_numpy(dtype=np.int64),
                grouped_val["prob"].to_numpy(dtype=np.float32),
                threshold=val_threshold,
            )
            score = grouped_val_metrics["auroc"] + grouped_val_metrics["f1"]
        else:
            score = val_metrics["auroc"] + val_metrics["f1"]

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_threshold": val_threshold,
            **loss_parts,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        if grouped_val_metrics:
            history_row.update({f"val_group_{k}": v for k, v in grouped_val_metrics.items()})
        history.append(history_row)

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
            f"{('val_group_auroc=' + format(grouped_val_metrics['auroc'], '.4f')) if grouped_val_metrics else 'val_group_auroc=skipped'} "
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

    val_y, val_prob, val_records, val_groups = evaluate_model(
        model,
        loaders["val"],
        device=device,
        tta_shifts=(0, -8, 8),
    )
    test_y, test_prob, test_records, test_groups = evaluate_model(
        model,
        loaders["test"],
        device=device,
        tta_shifts=(0, -8, 8),
    )

    val_metrics = compute_metrics(val_y, val_prob, threshold=best_threshold)
    test_metrics = compute_metrics(test_y, test_prob, threshold=best_threshold)

    grouped_val_metrics: dict[str, float] | None = None
    grouped_test_metrics: dict[str, float] | None = None
    grouped_test_predictions: pd.DataFrame | None = None
    if grouped_metrics_supported:
        grouped_val = summarize_by_group(val_groups, val_y, val_prob)
        grouped_test = summarize_by_group(test_groups, test_y, test_prob)
        grouped_val_metrics = compute_metrics(
            grouped_val["label"].to_numpy(dtype=np.int64),
            grouped_val["prob"].to_numpy(dtype=np.float32),
            threshold=best_threshold,
        )
        grouped_test_metrics = compute_metrics(
            grouped_test["label"].to_numpy(dtype=np.int64),
            grouped_test["prob"].to_numpy(dtype=np.float32),
            threshold=best_threshold,
        )
        grouped_test_predictions = grouped_test

    experiment_summary = {
        "device": str(device),
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_threshold": best_threshold,
        "threshold_objective": args.threshold_objective,
        "split_group_column": split_group_column,
        "split_groups": split_groups,
        "student_feature_columns": STUDENT_FEATURE_COLUMNS,
        "teacher_feature_columns": TEACHER_FEATURE_COLUMNS,
        "aux_target_columns": AUX_TARGET_COLUMNS,
        "ssl_checkpoint": str(args.ssl_checkpoint) if args.ssl_checkpoint is not None else None,
        "ssl_initialization": ssl_initialization,
        "student_normalization": stats_to_jsonable(student_stats),
        "teacher_normalization": stats_to_jsonable(teacher_stats),
        "aux_normalization": stats_to_jsonable(aux_stats),
        "segment_level": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "group_level": (
            {
                "val": grouped_val_metrics,
                "test": grouped_test_metrics,
            }
            if grouped_metrics_supported
            else {
                "skipped": True,
                "reason": f"{split_group_column} contains mixed window labels, so group-mean metrics are not meaningful.",
            }
        ),
        "runtime_seconds": time.time() - start_time,
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_threshold": best_threshold,
            "split_groups": split_groups,
            "split_group_column": split_group_column,
            "student_feature_columns": STUDENT_FEATURE_COLUMNS,
            "teacher_feature_columns": TEACHER_FEATURE_COLUMNS,
            "aux_target_columns": AUX_TARGET_COLUMNS,
            "ssl_checkpoint": str(args.ssl_checkpoint) if args.ssl_checkpoint is not None else None,
            "student_normalization": stats_to_jsonable(student_stats),
            "teacher_normalization": stats_to_jsonable(teacher_stats),
            "aux_normalization": stats_to_jsonable(aux_stats),
        },
        args.output_dir / "best_model.pt",
    )

    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    if grouped_test_predictions is not None:
        grouped_test_predictions.to_csv(args.output_dir / "test_group_predictions.csv", index=False)
    pd.DataFrame({"record_id": test_records, "label": test_y, "prob": test_prob}).to_csv(
        args.output_dir / "test_segment_predictions.csv",
        index=False,
    )
    save_json(experiment_summary, args.output_dir / "metrics.json")

    print("\nBest validation threshold:", round(best_threshold, 4))
    print("Segment-level test metrics:", json.dumps(test_metrics, indent=2))
    if grouped_test_metrics is not None:
        print("Group-level test metrics:", json.dumps(grouped_test_metrics, indent=2))
    else:
        print(
            "Group-level test metrics: skipped "
            f"({split_group_column} contains mixed window labels)"
        )
    print("Saved artifacts to:", args.output_dir)


if __name__ == "__main__":
    main()
