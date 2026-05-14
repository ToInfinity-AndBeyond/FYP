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

from af_pipeline.data import (
    RecordBagDataset,
    collate_record_bags,
    group_segments_into_bags,
    load_and_concat_signal_datasets,
)
from af_pipeline.features import (
    QUALITY_COLUMNS,
    build_quality_feature_matrix,
    fill_and_scale_features,
    make_multiscale_rhythm_features,
)
from af_pipeline.losses import RecordLevelHardMiningLoss
from af_pipeline.runtime import (
    choose_amp,
    compute_metrics,
    find_best_threshold,
    format_duration,
    get_device,
    log_stage,
    safe_probability_metric,
    save_json,
    set_seed,
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
from ppg_record_hierarchical_model import HierarchicalRecordPPGNet
from ppg_record_mil_model import RecordMILPPGNet


def logit_transform(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def quality_adjusted_probabilities(
    probabilities: np.ndarray,
    quality_confidence: np.ndarray,
    strength: float,
) -> np.ndarray:
    logits = logit_transform(probabilities)
    adjusted_logits = logits + strength * (np.asarray(quality_confidence, dtype=np.float32) - 0.5)
    return 1.0 / (1.0 + np.exp(-adjusted_logits))
def run_training_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    amp_device_type: str,
    epoch_index: int,
    total_epochs: int,
    training_start_time: float,
    progress_every_batches: int,
) -> float:
    model.train()
    scaler = torch.amp.GradScaler(enabled=amp_enabled)
    total_loss = 0.0
    total_items = 0
    epoch_start = time.time()
    total_batches = len(dataloader)

    for batch_index, batch in enumerate(dataloader, start=1):
        waveforms = batch["waveforms"].to(device)
        rhythm_features = batch["rhythm_features"].to(device)
        quality_features = batch["quality_features"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=amp_device_type, enabled=amp_enabled):
            outputs = model(waveforms, rhythm_features, quality_features, mask)
            loss = loss_fn(
                outputs["record_logits"],
                outputs["segment_logits"],
                labels,
                quality_features,
                mask,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = waveforms.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        if batch_index == 1 or batch_index == total_batches or (
            progress_every_batches > 0 and batch_index % progress_every_batches == 0
        ):
            epoch_fraction = batch_index / max(total_batches, 1)
            training_fraction = ((epoch_index - 1) + epoch_fraction) / max(total_epochs, 1)
            epoch_elapsed = time.time() - epoch_start
            total_elapsed = time.time() - training_start_time
            epoch_eta = (epoch_elapsed / max(epoch_fraction, 1e-8)) - epoch_elapsed
            total_eta = (total_elapsed / max(training_fraction, 1e-8)) - total_elapsed
            print(
                f"train_progress epoch={epoch_index:02d}/{total_epochs:02d} "
                f"batch={batch_index}/{total_batches} "
                f"epoch_pct={epoch_fraction * 100:5.1f}% "
                f"total_pct={training_fraction * 100:5.1f}% "
                f"loss={total_loss / max(total_items, 1):.4f} "
                f"epoch_elapsed={format_duration(epoch_elapsed)} "
                f"epoch_eta={format_duration(epoch_eta)} "
                f"total_eta={format_duration(total_eta)}",
                flush=True,
            )

    if total_items == 0:
        raise RuntimeError("Training did not process any record bags.")
    return total_loss / max(total_items, 1)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    quality_inference_strength: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    record_rows = []
    segment_rows = []

    with torch.no_grad():
        for batch in dataloader:
            waveforms = batch["waveforms"].to(device)
            rhythm_features = batch["rhythm_features"].to(device)
            quality_features = batch["quality_features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(waveforms, rhythm_features, quality_features, mask)
            record_prob = torch.sigmoid(outputs["record_logits"]).cpu().numpy()
            pooled_quality = outputs["pooled_quality"].cpu().numpy()
            quality_confidence = pooled_quality.mean(axis=1)
            adjusted_prob = quality_adjusted_probabilities(
                probabilities=record_prob,
                quality_confidence=quality_confidence,
                strength=quality_inference_strength,
            )

            for idx, group_id in enumerate(batch["group_ids"]):
                record_rows.append(
                    {
                        "group_id": str(group_id),
                        "record_id": str(batch["record_ids"][idx]),
                        "label": int(labels[idx].item()),
                        "prob": float(record_prob[idx]),
                        "adjusted_prob": float(adjusted_prob[idx]),
                        "quality_confidence": float(quality_confidence[idx]),
                        "full_segment_count": int(batch["full_segment_counts"][idx].item()),
                    }
                )

            segment_prob = torch.sigmoid(outputs["segment_logits"]).cpu().numpy()
            mask_np = batch["mask"].cpu().numpy()
            quality_np = batch["quality_features"].cpu().numpy()
            for bag_index, group_id in enumerate(batch["group_ids"]):
                valid_count = int(mask_np[bag_index].sum())
                label_value = int(labels[bag_index].item())
                for segment_index in range(valid_count):
                    probability = float(segment_prob[bag_index, segment_index])
                    quality_conf = float(quality_np[bag_index, segment_index, 0])
                    adjusted_segment_prob = float(
                        quality_adjusted_probabilities(
                            probabilities=np.asarray([probability], dtype=np.float32),
                            quality_confidence=np.asarray([quality_conf], dtype=np.float32),
                            strength=quality_inference_strength,
                        )[0]
                    )
                    segment_rows.append(
                        {
                            "group_id": str(group_id),
                            "record_id": str(batch["record_ids"][bag_index]),
                            "label": label_value,
                            "prob": probability,
                            "adjusted_prob": adjusted_segment_prob,
                            "quality_score": quality_conf,
                        }
                    )

    return pd.DataFrame.from_records(record_rows), pd.DataFrame.from_records(segment_rows)
def get_score_column_name(decision_score: str) -> str:
    if decision_score == "prob":
        return "prob"
    if decision_score == "adjusted_prob":
        return "adjusted_prob"
    raise ValueError(f"Unsupported decision score: {decision_score}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a record-level PPG MIL AF classifier.")
    parser.add_argument(
        "--segments-path",
        type=Path,
        nargs="+",
        required=True,
        help="One or more accepted PPG segments NPZ paths",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        nargs="+",
        required=True,
        help="One or more accepted PPG summary CSV paths",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-records", type=int, default=5)
    parser.add_argument("--test-records", type=int, default=5)
    parser.add_argument(
        "--split-mode",
        choices=("patient", "random_windows", "metadata_folds"),
        default="metadata_folds",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--train-folds", type=_parse_fold_list, default=_parse_fold_list("0,1,2,3,4,5,6,7"))
    parser.add_argument("--val-folds", type=_parse_fold_list, default=_parse_fold_list("8"))
    parser.add_argument("--test-folds", type=_parse_fold_list, default=_parse_fold_list("9"))
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--threshold-objective", choices=("balanced_accuracy", "f1"), default="f1")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--progress-every-batches", type=int, default=20)
    parser.add_argument("--max-segments-per-record", type=int, default=32)
    parser.add_argument("--eval-max-segments-per-record", type=int, default=64)
    parser.add_argument("--quality-inference-strength", type=float, default=0.75)
    parser.add_argument("--decision-score", choices=("prob", "adjusted_prob"), default="prob")
    parser.add_argument("--segment-aux-weight", type=float, default=0.1)
    parser.add_argument("--hard-negative-scale", type=float, default=1.75)
    parser.add_argument("--hard-positive-scale", type=float, default=0.75)
    parser.add_argument("--segment-quality-power", type=float, default=2.0)
    parser.add_argument("--attention-quality-floor", type=float, default=0.35)
    parser.add_argument("--attention-quality-power", type=float, default=2.0)
    parser.add_argument("--max-groups-per-split", type=int, default=0)
    parser.add_argument(
        "--model-type",
        choices=("mil", "hierarchical"),
        default="mil",
        help="Record-level architecture to train.",
    )
    parser.add_argument("--token-dim", type=int, default=192)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
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
        "[setup] starting record-level MIL training: "
        f"device={device.type} split_mode={args.split_mode} epochs={args.epochs} "
        f"batch_size={args.batch_size} amp_enabled={amp_enabled}"
    )
    signals, summary_df = load_and_concat_signal_datasets(args.segments_path, args.summary_path)
    raw_quality_scores = pd.to_numeric(summary_df["quality_score"], errors="coerce").fillna(0.5).clip(0.0, 1.0).to_numpy(
        dtype=np.float32
    )

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

    validate_split_masks(summary_df, split_masks)
    record_grouping, record_group_ids = infer_record_grouping(summary_df)
    if record_group_ids is None:
        raise ValueError("Record-level MIL requires consistent labels per record group, but no safe grouping was found.")
    summary_df = summary_df.copy()
    summary_df["_record_group_id"] = record_group_ids.astype(str).to_numpy()
    log_stage(f"[split] record-level grouping={record_grouping}")

    summary_df, feature_stats = fill_and_scale_features(summary_df, split_masks)
    combined_features, combined_feature_names, combined_feature_stats = make_multiscale_rhythm_features(
        summary_df=summary_df,
        group_ids=summary_df["_record_group_id"],
        split_masks=split_masks,
    )
    quality_matrix = build_quality_feature_matrix(summary_df)

    bags_by_split = group_segments_into_bags(
        summary_df=summary_df,
        split_masks=split_masks,
        group_ids=summary_df["_record_group_id"],
        raw_quality_scores=raw_quality_scores,
        max_segments_per_record=args.max_segments_per_record,
        eval_max_segments_per_record=args.eval_max_segments_per_record,
        max_groups_per_split=args.max_groups_per_split if args.max_groups_per_split > 0 else None,
    )

    log_stage(
        "[bags] split groups: "
        + json.dumps({split_name: len(bags) for split_name, bags in bags_by_split.items()})
    )

    datasets = {
        split_name: RecordBagDataset(
            signals=signals,
            combined_features=combined_features,
            quality_features=quality_matrix,
            bags=bags,
        )
        for split_name, bags in bags_by_split.items()
    }

    loaders = {}
    for split_name, dataset in datasets.items():
        sampler = None
        shuffle = split_name == "train"
        if split_name == "train" and args.balanced_sampler:
            labels = np.asarray([bag["label"] for bag in bags_by_split["train"]], dtype=np.int64)
            class_counts = np.bincount(labels, minlength=2)
            class_weights = np.zeros(2, dtype=np.float32)
            for class_index, count in enumerate(class_counts):
                class_weights[class_index] = 1.0 / max(int(count), 1)
            sample_weights = class_weights[labels]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            collate_fn=collate_record_bags,
        )

    train_labels = np.asarray([bag["label"] for bag in bags_by_split["train"]], dtype=np.float32)
    train_pos = float(train_labels.sum())
    train_neg = float(train_labels.shape[0] - train_pos)
    pos_weight = train_neg / max(train_pos, 1.0)
    print(
        "dataset summary:",
        json.dumps(
            {
                "total_segments": int(summary_df.shape[0]),
                "record_grouping": record_grouping,
                "record_groups": {split_name: len(bags) for split_name, bags in bags_by_split.items()},
                "train_pos_records": int(train_pos),
                "train_neg_records": int(train_neg),
                "split_mode": args.split_mode,
            }
        ),
        flush=True,
    )

    if args.model_type == "hierarchical":
        model = HierarchicalRecordPPGNet(
            rhythm_feature_dim=combined_features.shape[1],
            quality_feature_dim=quality_matrix.shape[1],
            token_dim=args.token_dim,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads,
            quality_floor=args.attention_quality_floor,
            quality_power=args.attention_quality_power,
        ).to(device)
    else:
        model = RecordMILPPGNet(
            rhythm_feature_dim=combined_features.shape[1],
            quality_feature_dim=quality_matrix.shape[1],
            token_dim=args.token_dim,
            quality_floor=args.attention_quality_floor,
            quality_power=args.attention_quality_power,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = RecordLevelHardMiningLoss(
        pos_weight=pos_weight,
        segment_aux_weight=args.segment_aux_weight,
        hard_negative_scale=args.hard_negative_scale,
        hard_positive_scale=args.hard_positive_scale,
        segment_quality_power=args.segment_quality_power,
    )
    decision_score_column = get_score_column_name(args.decision_score)

    best_state = None
    best_epoch = 0
    best_threshold = 0.5
    best_score = -math.inf
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
            epoch_index=epoch,
            total_epochs=args.epochs,
            training_start_time=start_time,
            progress_every_batches=args.progress_every_batches,
        )
        scheduler.step()

        record_val, segment_val = evaluate_model(
            model=model,
            dataloader=loaders["val"],
            device=device,
            quality_inference_strength=args.quality_inference_strength,
        )
        val_threshold = find_best_threshold(
            record_val["label"].to_numpy(dtype=np.int64),
            record_val[decision_score_column].to_numpy(dtype=np.float32),
            objective=args.threshold_objective,
        )
        val_metrics = compute_metrics(
            record_val["label"].to_numpy(dtype=np.int64),
            record_val[decision_score_column].to_numpy(dtype=np.float32),
            threshold=val_threshold,
        )
        val_segment_metrics = compute_metrics(
            segment_val["label"].to_numpy(dtype=np.int64),
            segment_val[decision_score_column].to_numpy(dtype=np.float32),
            threshold=val_threshold,
        )
        score = val_metrics["f1"] + val_metrics["auprc"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_threshold": val_threshold,
                **{f"val_record_{key}": value for key, value in val_metrics.items()},
                **{f"val_segment_{key}": value for key, value in val_segment_metrics.items()},
            }
        )

        elapsed = time.time() - start_time
        avg_epoch_seconds = elapsed / epoch
        eta_seconds = avg_epoch_seconds * max(args.epochs - epoch, 0)
        print(
            f"epoch={epoch:02d} loss={train_loss:.4f} "
            f"val_record_auroc={val_metrics['auroc']:.4f} val_record_f1={val_metrics['f1']:.4f} "
            f"val_segment_auroc={val_segment_metrics['auroc']:.4f} val_segment_f1={val_segment_metrics['f1']:.4f} "
            f"epoch_time={format_duration(time.time() - epoch_start)} "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta_seconds)}",
            flush=True,
        )

        if score > best_score:
            best_score = score
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
        raise RuntimeError("Record-level MIL training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    record_val, segment_val = evaluate_model(
        model=model,
        dataloader=loaders["val"],
        device=device,
        quality_inference_strength=args.quality_inference_strength,
    )
    record_test, segment_test = evaluate_model(
        model=model,
        dataloader=loaders["test"],
        device=device,
        quality_inference_strength=args.quality_inference_strength,
    )

    record_val_metrics = compute_metrics(
        record_val["label"].to_numpy(dtype=np.int64),
        record_val[decision_score_column].to_numpy(dtype=np.float32),
        threshold=best_threshold,
    )
    record_test_metrics = compute_metrics(
        record_test["label"].to_numpy(dtype=np.int64),
        record_test[decision_score_column].to_numpy(dtype=np.float32),
        threshold=best_threshold,
    )
    segment_val_metrics = compute_metrics(
        segment_val["label"].to_numpy(dtype=np.int64),
        segment_val[decision_score_column].to_numpy(dtype=np.float32),
        threshold=best_threshold,
    )
    segment_test_metrics = compute_metrics(
        segment_test["label"].to_numpy(dtype=np.int64),
        segment_test[decision_score_column].to_numpy(dtype=np.float32),
        threshold=best_threshold,
    )

    metrics = {
        "device": str(device),
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_threshold": best_threshold,
        "record_grouping": record_grouping,
        "quality_inference_strength": args.quality_inference_strength,
        "decision_score": args.decision_score,
        "segment_aux_weight": args.segment_aux_weight,
        "hard_negative_scale": args.hard_negative_scale,
        "hard_positive_scale": args.hard_positive_scale,
        "segment_quality_power": args.segment_quality_power,
        "attention_quality_floor": args.attention_quality_floor,
        "attention_quality_power": args.attention_quality_power,
        "model_type": args.model_type,
        "token_dim": args.token_dim,
        "transformer_layers": args.transformer_layers,
        "transformer_heads": args.transformer_heads,
        "record_level": {
            "val": record_val_metrics,
            "test": record_test_metrics,
        },
        "segment_level": {
            "val": segment_val_metrics,
            "test": segment_test_metrics,
        },
        "runtime_seconds": time.time() - start_time,
        "combined_feature_columns": combined_feature_names,
        "combined_feature_stats": {
            "means": combined_feature_stats.means.tolist(),
            "stds": combined_feature_stats.stds.tolist(),
        },
        "base_feature_stats": {
            "means": feature_stats.feature_means.tolist(),
            "stds": feature_stats.feature_stds.tolist(),
            "medians": feature_stats.feature_medians.tolist(),
        },
        "group_counts": {split_name: len(bags) for split_name, bags in bags_by_split.items()},
        "avg_selected_segments_per_record": {
            split_name: float(np.mean([len(bag["indices"]) for bag in bags])) if bags else 0.0
            for split_name, bags in bags_by_split.items()
        },
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_threshold": best_threshold,
            "record_grouping": record_grouping,
            "quality_inference_strength": args.quality_inference_strength,
            "decision_score": args.decision_score,
            "segment_aux_weight": args.segment_aux_weight,
            "hard_negative_scale": args.hard_negative_scale,
            "hard_positive_scale": args.hard_positive_scale,
            "segment_quality_power": args.segment_quality_power,
            "attention_quality_floor": args.attention_quality_floor,
            "attention_quality_power": args.attention_quality_power,
            "model_type": args.model_type,
            "token_dim": args.token_dim,
            "transformer_layers": args.transformer_layers,
            "transformer_heads": args.transformer_heads,
            "combined_feature_columns": combined_feature_names,
            "combined_feature_stats": {
                "means": combined_feature_stats.means,
                "stds": combined_feature_stats.stds,
            },
            "quality_columns": QUALITY_COLUMNS,
        },
        args.output_dir / "best_model.pt",
    )
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    record_val.to_csv(args.output_dir / "val_record_predictions.csv", index=False)
    record_test.to_csv(args.output_dir / "test_record_predictions.csv", index=False)
    segment_val.to_csv(args.output_dir / "val_segment_predictions.csv", index=False)
    segment_test.to_csv(args.output_dir / "test_segment_predictions.csv", index=False)
    save_json(metrics, args.output_dir / "metrics.json")

    print("\nBest validation threshold:", round(best_threshold, 4), flush=True)
    print("Record-level test metrics:", json.dumps(record_test_metrics, indent=2), flush=True)
    print("Segment-level test metrics:", json.dumps(segment_test_metrics, indent=2), flush=True)
    print("Saved artifacts to:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
