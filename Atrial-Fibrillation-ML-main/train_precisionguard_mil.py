from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from af_pipeline.runtime import compute_metrics, find_best_threshold, save_json, set_seed
from train_ppg_hybrid import build_threshold_sweep, best_threshold_row


BASE_SEGMENT_COLUMNS = [
    "record_id",
    "group_id",
    "label",
    "segment_index",
    "start_time_sec",
    "end_time_sec",
    "quality_score",
    "quality_score_runtime",
    "template_correlation",
    "heart_band_energy_ratio",
    "estimated_hr_bpm",
    "mean_hr_bpm",
    "std_hr_bpm",
    "sample_entropy",
    "strat_fold",
    "subject_id",
    "event_id",
    "signal_file_name",
    "patient",
    "folder_path",
]

QUALITY_FEATURES = [
    "quality_score",
    "quality_score_runtime",
    "template_correlation",
    "heart_band_energy_ratio",
]


def parse_prediction_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Prediction spec must be name=path, got: {text}")
    name, path = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Prediction spec has an empty name: {text}")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PrecisionGuard-MIL record-level verifier from PPG-only segment predictions."
    )
    parser.add_argument("--train-prediction", action="append", type=parse_prediction_spec, required=True)
    parser.add_argument("--val-prediction", action="append", type=parse_prediction_spec, required=True)
    parser.add_argument("--test-prediction", action="append", type=parse_prediction_spec, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hard-negative-threshold", type=float, default=0.5)
    parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def segment_key_columns(frame: pd.DataFrame) -> list[str]:
    preferred = ["group_id", "segment_index", "start_time_sec", "end_time_sec", "label"]
    available = [column for column in preferred if column in frame.columns]
    if "group_id" not in available:
        available = ["record_id", "segment_index", "start_time_sec", "end_time_sec", "label"]
    return available


def load_prediction_frame(prediction_specs: list[tuple[str, Path]], split_name: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    key_columns: list[str] | None = None
    probability_columns = []

    for model_name, path in prediction_specs:
        frame = pd.read_csv(path)
        if "prob" not in frame.columns:
            raise ValueError(f"{path} does not contain a 'prob' column.")
        frame = frame.copy()
        frame[f"p_{model_name}"] = pd.to_numeric(frame["prob"], errors="coerce").astype(np.float32)

        if merged is None:
            base_columns = [column for column in BASE_SEGMENT_COLUMNS if column in frame.columns]
            merged = frame[base_columns + [f"p_{model_name}"]].reset_index(drop=True).copy()
            key_columns = segment_key_columns(merged)
        else:
            assert key_columns is not None
            join_frame = frame[key_columns + [f"p_{model_name}"]].reset_index(drop=True).copy()
            if len(join_frame) == len(merged) and merged[key_columns].reset_index(drop=True).equals(join_frame[key_columns]):
                merged[f"p_{model_name}"] = join_frame[f"p_{model_name}"].to_numpy(dtype=np.float32)
            else:
                merged = merged.merge(join_frame, on=key_columns, how="inner")
        probability_columns.append(f"p_{model_name}")

    if merged is None:
        raise ValueError(f"No prediction files were supplied for split={split_name}.")
    if merged.empty:
        raise ValueError(f"Merged prediction frame for split={split_name} is empty.")

    merged["split"] = split_name
    prob_values = merged[probability_columns].to_numpy(dtype=np.float32)
    merged["p_mean"] = np.nanmean(prob_values, axis=1)
    merged["p_max"] = np.nanmax(prob_values, axis=1)
    merged["p_min"] = np.nanmin(prob_values, axis=1)
    merged["p_std"] = np.nanstd(prob_values, axis=1)
    if "start_time_sec" in merged.columns:
        start = pd.to_numeric(merged["start_time_sec"], errors="coerce").fillna(0.0)
        max_start = max(float(start.max()), 1.0)
        merged["segment_position"] = (start / max_start).astype(np.float32)
    else:
        merged["segment_position"] = 0.0
    return merged


def record_aggregate_features(group: pd.DataFrame, probability_columns: list[str]) -> dict[str, float]:
    row: dict[str, float] = {}
    p_mean = group["p_mean"].to_numpy(dtype=np.float32)
    p_max = group["p_max"].to_numpy(dtype=np.float32)
    quality = group.get("quality_score_runtime", group.get("quality_score", pd.Series(0.5, index=group.index)))
    quality_values = pd.to_numeric(quality, errors="coerce").fillna(0.5).clip(0.0, 1.0).to_numpy(dtype=np.float32)

    def topk_mean(values: np.ndarray, k: int) -> float:
        if values.size == 0:
            return 0.0
        k = min(k, values.size)
        return float(np.mean(np.partition(values, -k)[-k:]))

    row["segment_count_log"] = float(np.log1p(group.shape[0]))
    row["p_mean_record"] = float(np.mean(p_mean))
    row["p_max_record"] = float(np.max(p_max))
    row["p_top3_mean"] = topk_mean(p_mean, 3)
    row["p_top5_mean"] = topk_mean(p_mean, 5)
    row["p_top10_mean"] = topk_mean(p_mean, 10)
    row["p_fraction_above_0_5"] = float(np.mean(p_mean >= 0.5))
    row["p_fraction_above_0_7"] = float(np.mean(p_mean >= 0.7))
    row["p_std_record"] = float(np.std(p_mean))
    row["quality_mean_record"] = float(np.mean(quality_values))
    row["quality_top_prob_mean"] = float(np.mean(quality_values[p_mean >= np.quantile(p_mean, 0.75)]))
    for column in probability_columns:
        values = group[column].to_numpy(dtype=np.float32)
        row[f"{column}_top5_mean"] = topk_mean(values, 5)
        row[f"{column}_max"] = float(np.max(values))
    return row


def build_record_bags(frame: pd.DataFrame, feature_columns: list[str], probability_columns: list[str]) -> list[dict[str, Any]]:
    records = []
    group_column = "group_id" if "group_id" in frame.columns else "record_id"
    sort_columns = [column for column in ("start_time_sec", "segment_index") if column in frame.columns]
    for group_id, group in frame.groupby(group_column, sort=False):
        if sort_columns:
            group = group.sort_values(sort_columns)
        features = group[feature_columns].to_numpy(dtype=np.float32)
        label = int(group["label"].iloc[0])
        aggregate = record_aggregate_features(group, probability_columns)
        records.append(
            {
                "group_id": str(group_id),
                "label": label,
                "features": features,
                "aggregate": aggregate,
                "segment_count": int(group.shape[0]),
                "baseline_score": aggregate["p_top5_mean"],
            }
        )
    return records


class RecordBagDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        aggregate_columns: list[str],
        hard_negative_threshold: float,
        hard_negative_weight: float,
    ):
        self.records = records
        self.aggregate_columns = aggregate_columns
        self.hard_negative_threshold = hard_negative_threshold
        self.hard_negative_weight = hard_negative_weight

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        label = float(record["label"])
        is_hard_negative = label == 0.0 and float(record["baseline_score"]) >= self.hard_negative_threshold
        weight = self.hard_negative_weight if is_hard_negative else 1.0
        aggregate = np.asarray(
            [record["aggregate"][column] for column in self.aggregate_columns],
            dtype=np.float32,
        )
        return {
            "group_id": record["group_id"],
            "features": record["features"].astype(np.float32),
            "aggregate": aggregate,
            "label": np.float32(label),
            "weight": np.float32(weight),
            "baseline_score": np.float32(record["baseline_score"]),
            "is_hard_negative": bool(is_hard_negative),
        }


def collate_bags(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_segments = max(item["features"].shape[0] for item in batch)
    feature_dim = batch[0]["features"].shape[1]
    aggregate_dim = batch[0]["aggregate"].shape[0]
    features = torch.zeros(batch_size, max_segments, feature_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_segments, dtype=torch.bool)
    aggregate = torch.zeros(batch_size, aggregate_dim, dtype=torch.float32)
    labels = torch.zeros(batch_size, dtype=torch.float32)
    weights = torch.ones(batch_size, dtype=torch.float32)
    baseline_scores = torch.zeros(batch_size, dtype=torch.float32)
    hard_negative = torch.zeros(batch_size, dtype=torch.bool)
    group_ids = []

    for i, item in enumerate(batch):
        segment_count = item["features"].shape[0]
        features[i, :segment_count] = torch.from_numpy(item["features"])
        mask[i, :segment_count] = True
        aggregate[i] = torch.from_numpy(item["aggregate"])
        labels[i] = float(item["label"])
        weights[i] = float(item["weight"])
        baseline_scores[i] = float(item["baseline_score"])
        hard_negative[i] = bool(item["is_hard_negative"])
        group_ids.append(item["group_id"])

    return {
        "features": features,
        "mask": mask,
        "aggregate": aggregate,
        "labels": labels,
        "weights": weights,
        "baseline_scores": baseline_scores,
        "hard_negative": hard_negative,
        "group_ids": group_ids,
    }


class PrecisionGuardMIL(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        aggregate_dim: int,
        quality_indices: list[int],
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.quality_indices = quality_indices
        self.segment_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        quality_dim = max(len(quality_indices), 1)
        self.quality_gate = nn.Sequential(
            nn.Linear(quality_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.aggregate_encoder = nn.Sequential(
            nn.Linear(aggregate_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor, aggregate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.segment_encoder(features)
        attention_logits = self.attention(encoded).squeeze(-1)
        if self.quality_indices:
            quality_inputs = features[:, :, self.quality_indices]
        else:
            quality_inputs = torch.ones(features.shape[:2] + (1,), dtype=features.dtype, device=features.device)
        quality_bias = torch.log(torch.sigmoid(self.quality_gate(quality_inputs)).squeeze(-1).clamp_min(1e-4))
        attention_logits = (attention_logits + quality_bias).masked_fill(~mask, -1.0e4)
        attention_weights = torch.softmax(attention_logits, dim=1)
        record_embedding = torch.sum(encoded * attention_weights.unsqueeze(-1), dim=1)
        aggregate_embedding = self.aggregate_encoder(aggregate)
        logits = self.classifier(torch.cat([record_embedding, aggregate_embedding], dim=1)).squeeze(1)
        return logits, attention_weights


def normalize_features(
    train_records: list[dict[str, Any]],
    other_records: list[list[dict[str, Any]]],
    feature_dim: int,
    aggregate_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_segments = np.concatenate([record["features"] for record in train_records], axis=0)
    feature_mean = np.nanmean(train_segments, axis=0).astype(np.float32)
    feature_std = np.nanstd(train_segments, axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)

    train_aggregates = np.asarray(
        [[record["aggregate"][column] for column in aggregate_columns] for record in train_records],
        dtype=np.float32,
    )
    aggregate_mean = np.nanmean(train_aggregates, axis=0).astype(np.float32)
    aggregate_std = np.nanstd(train_aggregates, axis=0).astype(np.float32)
    aggregate_std = np.where(aggregate_std < 1e-6, 1.0, aggregate_std).astype(np.float32)

    all_groups = [train_records] + other_records
    for records in all_groups:
        for record in records:
            record["features"] = np.nan_to_num((record["features"] - feature_mean) / feature_std, nan=0.0)
            for column_index, column in enumerate(aggregate_columns):
                record["aggregate"][column] = float(
                    np.nan_to_num(
                        (record["aggregate"][column] - aggregate_mean[column_index]) / aggregate_std[column_index],
                        nan=0.0,
                    )
                )

    if feature_mean.shape[0] != feature_dim:
        raise ValueError("Unexpected feature normalization shape mismatch.")
    return feature_mean, feature_std, aggregate_mean, aggregate_std


def make_loader(
    records: list[dict[str, Any]],
    aggregate_columns: list[str],
    batch_size: int,
    hard_negative_threshold: float,
    hard_negative_weight: float,
    balanced_sampler: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = RecordBagDataset(
        records=records,
        aggregate_columns=aggregate_columns,
        hard_negative_threshold=hard_negative_threshold,
        hard_negative_weight=hard_negative_weight,
    )
    sampler = None
    if balanced_sampler:
        labels = np.asarray([record["label"] for record in records], dtype=np.int64)
        counts = np.bincount(labels, minlength=2)
        class_weights = np.asarray([1.0 / max(counts[0], 1), 1.0 / max(counts[1], 1)], dtype=np.float32)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, collate_fn=collate_bags)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        aggregate = batch["aggregate"].to(device)
        labels = batch["labels"].to(device)
        weights = batch["weights"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(features, mask, aggregate)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = (loss * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += float(loss.item()) * labels.numel()
        total_items += labels.numel()
    return total_loss / max(total_items, 1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            aggregate = batch["aggregate"].to(device)
            logits, attention = model(features, mask, aggregate)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch["labels"].cpu().numpy().astype(np.int64)
            baseline = batch["baseline_scores"].cpu().numpy()
            hard_negative = batch["hard_negative"].cpu().numpy()
            for i, group_id in enumerate(batch["group_ids"]):
                valid_attention = attention[i, : int(mask[i].sum().item())].detach().cpu().numpy()
                rows.append(
                    {
                        "group_id": group_id,
                        "label": int(labels[i]),
                        "prob": float(probs[i]),
                        "baseline_score": float(baseline[i]),
                        "is_hard_negative": bool(hard_negative[i]),
                        "max_attention": float(np.max(valid_attention)) if valid_attention.size else 0.0,
                        "attention_entropy": float(-(valid_attention * np.log(valid_attention + 1e-8)).sum())
                        if valid_attention.size
                        else 0.0,
                    }
                )
    return pd.DataFrame.from_records(rows)


def metrics_from_predictions(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    return compute_metrics(
        frame["label"].to_numpy(dtype=np.int64),
        frame["prob"].to_numpy(dtype=np.float32),
        threshold=threshold,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_frame = load_prediction_frame(args.train_prediction, "train")
    val_frame = load_prediction_frame(args.val_prediction, "val")
    test_frame = load_prediction_frame(args.test_prediction, "test")

    probability_columns = sorted([column for column in train_frame.columns if column.startswith("p_") and column not in {"p_mean", "p_max", "p_min", "p_std"}])
    feature_columns = probability_columns + [
        "p_mean",
        "p_max",
        "p_min",
        "p_std",
        "quality_score",
        "quality_score_runtime",
        "template_correlation",
        "heart_band_energy_ratio",
        "estimated_hr_bpm",
        "mean_hr_bpm",
        "std_hr_bpm",
        "sample_entropy",
        "segment_position",
    ]
    feature_columns = [column for column in feature_columns if column in train_frame.columns]
    for frame in (train_frame, val_frame, test_frame):
        for column in feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = frame[column].replace([np.inf, -np.inf], np.nan).fillna(frame[column].median())
            frame[column] = frame[column].fillna(0.0).astype(np.float32)

    train_records = build_record_bags(train_frame, feature_columns, probability_columns)
    val_records = build_record_bags(val_frame, feature_columns, probability_columns)
    test_records = build_record_bags(test_frame, feature_columns, probability_columns)
    aggregate_columns = sorted(train_records[0]["aggregate"].keys())
    normalize_features(train_records, [val_records, test_records], len(feature_columns), aggregate_columns)

    hard_negative_count = sum(
        1 for record in train_records if int(record["label"]) == 0 and float(record["baseline_score"]) >= args.hard_negative_threshold
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quality_indices = [feature_columns.index(column) for column in QUALITY_FEATURES if column in feature_columns]

    train_loader = make_loader(
        train_records,
        aggregate_columns,
        args.batch_size,
        args.hard_negative_threshold,
        args.hard_negative_weight,
        args.balanced_sampler,
        shuffle=True,
    )
    val_loader = make_loader(
        val_records,
        aggregate_columns,
        args.batch_size,
        args.hard_negative_threshold,
        args.hard_negative_weight,
        balanced_sampler=False,
        shuffle=False,
    )
    test_loader = make_loader(
        test_records,
        aggregate_columns,
        args.batch_size,
        args.hard_negative_threshold,
        args.hard_negative_weight,
        balanced_sampler=False,
        shuffle=False,
    )

    model = PrecisionGuardMIL(
        feature_dim=len(feature_columns),
        aggregate_dim=len(aggregate_columns),
        quality_indices=quality_indices,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_score = -math.inf
    best_epoch = 0
    best_threshold = 0.5
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        val_predictions = evaluate(model, val_loader, device)
        threshold = find_best_threshold(
            val_predictions["label"].to_numpy(dtype=np.int64),
            val_predictions["prob"].to_numpy(dtype=np.float32),
            objective="f1",
        )
        val_metrics = metrics_from_predictions(val_predictions, threshold)
        score = val_metrics["f1"] + 0.25 * val_metrics["auprc"]
        history.append({"epoch": epoch, "loss": loss, "threshold": threshold, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(
            f"epoch={epoch:03d} loss={loss:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} val_sensitivity={val_metrics['sensitivity']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} val_auprc={val_metrics['auprc']:.4f} threshold={threshold:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("No valid PrecisionGuard-MIL state was trained.")
    model.load_state_dict(best_state)
    val_predictions = evaluate(model, val_loader, device)
    test_predictions = evaluate(model, test_loader, device)
    val_metrics = metrics_from_predictions(val_predictions, best_threshold)
    test_metrics = metrics_from_predictions(test_predictions, best_threshold)

    baseline_val_threshold = find_best_threshold(
        val_predictions["label"].to_numpy(dtype=np.int64),
        val_predictions["baseline_score"].to_numpy(dtype=np.float32),
        objective="f1",
    )
    baseline_val = compute_metrics(
        val_predictions["label"].to_numpy(dtype=np.int64),
        val_predictions["baseline_score"].to_numpy(dtype=np.float32),
        threshold=baseline_val_threshold,
    )
    baseline_test = compute_metrics(
        test_predictions["label"].to_numpy(dtype=np.int64),
        test_predictions["baseline_score"].to_numpy(dtype=np.float32),
        threshold=baseline_val_threshold,
    )

    val_sweep = build_threshold_sweep(
        val_predictions["label"].to_numpy(dtype=np.int64),
        val_predictions["prob"].to_numpy(dtype=np.float32),
    )
    test_sweep = build_threshold_sweep(
        test_predictions["label"].to_numpy(dtype=np.int64),
        test_predictions["prob"].to_numpy(dtype=np.float32),
    )
    metrics = {
        "model": "PrecisionGuardMIL",
        "best_epoch": best_epoch,
        "best_val_threshold": best_threshold,
        "feature_columns": feature_columns,
        "aggregate_columns": aggregate_columns,
        "probability_columns": probability_columns,
        "hard_negative_threshold": args.hard_negative_threshold,
        "hard_negative_weight": args.hard_negative_weight,
        "hard_negative_count": hard_negative_count,
        "record_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "val": val_metrics,
        "test": test_metrics,
        "baseline_top5_mean": {
            "val_threshold": baseline_val_threshold,
            "val": baseline_val,
            "test": baseline_test,
        },
        "threshold_sweep_best_f1": {
            "val": best_threshold_row(val_sweep, metric="f1"),
            "test_analysis_only": best_threshold_row(test_sweep, metric="f1"),
        },
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "aggregate_columns": aggregate_columns,
            "quality_indices": quality_indices,
            "best_threshold": best_threshold,
        },
        args.output_dir / "best_model.pt",
    )
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    val_predictions.to_csv(args.output_dir / "val_record_predictions.csv", index=False)
    test_predictions.to_csv(args.output_dir / "test_record_predictions.csv", index=False)
    val_sweep.to_csv(args.output_dir / "val_threshold_sweep.csv", index=False)
    test_sweep.to_csv(args.output_dir / "test_threshold_sweep.csv", index=False)
    save_json(metrics, args.output_dir / "metrics.json")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
