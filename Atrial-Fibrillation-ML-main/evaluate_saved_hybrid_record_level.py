from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ppg_hybrid_model import RhythmMorphologyFusionNet
from train_ppg_hybrid import (
    FEATURE_COLUMNS,
    PPGSegmentDataset,
    compute_metrics,
    create_metadata_fold_split_masks,
    evaluate_model,
    find_best_threshold,
    get_device,
    infer_record_grouping,
    load_and_concat_signal_datasets,
    summarize_by_record,
)


def _parse_fold_list(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        normalized = part.strip()
        if normalized:
            values.append(int(normalized))
    if not values:
        raise ValueError("Fold list must contain at least one integer.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved hybrid model with record-level threshold optimization.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--segments-path", type=Path, nargs="+", required=True)
    parser.add_argument("--summary-path", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-folds", type=_parse_fold_list, default=_parse_fold_list("8"))
    parser.add_argument("--test-folds", type=_parse_fold_list, default=_parse_fold_list("9"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold-objective", choices=("balanced_accuracy", "f1"), default="f1")
    return parser.parse_args()


def apply_saved_normalization(summary_df: pd.DataFrame, checkpoint: dict[str, object]) -> pd.DataFrame:
    normalized = summary_df.copy()
    medians = np.asarray(checkpoint["normalization"]["feature_medians"], dtype=np.float32)
    means = np.asarray(checkpoint["normalization"]["feature_means"], dtype=np.float32)
    stds = np.asarray(checkpoint["normalization"]["feature_stds"], dtype=np.float32)
    stds = np.where(stds == 0.0, 1.0, stds)

    feature_frame = normalized[FEATURE_COLUMNS].copy()
    feature_frame = feature_frame.fillna(dict(zip(FEATURE_COLUMNS, medians.tolist())))
    feature_values = feature_frame.to_numpy(dtype=np.float32)
    feature_values = (feature_values - means) / stds
    feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)
    for column_index, column_name in enumerate(FEATURE_COLUMNS):
        normalized[column_name] = feature_values[:, column_index].astype(np.float32)
    return normalized


def build_loader(
    signals: np.ndarray,
    summary_df: pd.DataFrame,
    raw_quality_scores: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
) -> DataLoader:
    record_column = "_record_group_id" if "_record_group_id" in summary_df.columns else "record_id"
    dataset = PPGSegmentDataset(
        signals=signals[mask],
        features=summary_df.loc[mask, FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        labels=summary_df.loc[mask, "label"].to_numpy(dtype=np.float32),
        records=summary_df.loc[mask, record_column].astype(str).to_numpy(),
        quality_scores=raw_quality_scores[mask].astype(np.float32),
        augment=None,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    signals, summary_df = load_and_concat_signal_datasets(args.segments_path, args.summary_path)
    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    raw_quality_scores = pd.to_numeric(summary_df["quality_score"], errors="coerce").to_numpy(dtype=np.float32)
    raw_quality_scores = np.nan_to_num(raw_quality_scores, nan=0.5, posinf=1.0, neginf=0.0)
    raw_quality_scores = np.clip(raw_quality_scores, 0.0, 1.0)

    split_masks, _ = create_metadata_fold_split_masks(
        summary_df=summary_df,
        train_folds=[],
        val_folds=args.val_folds,
        test_folds=args.test_folds,
    )
    summary_df = apply_saved_normalization(summary_df, checkpoint)

    record_grouping, record_group_ids = infer_record_grouping(summary_df)
    if record_group_ids is None:
        raise ValueError("Record-level metrics are not supported because no consistent grouping key was found.")
    summary_df = summary_df.copy()
    summary_df["_record_group_id"] = record_group_ids.to_numpy()

    val_loader = build_loader(signals, summary_df, raw_quality_scores, split_masks["val"], args.batch_size)
    test_loader = build_loader(signals, summary_df, raw_quality_scores, split_masks["test"], args.batch_size)

    device = get_device()
    model = RhythmMorphologyFusionNet(
        feature_dim=len(checkpoint["feature_columns"]),
        signal_length=signals.shape[1],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_y, val_prob, val_records = evaluate_model(model, val_loader, device=device, tta_shifts=(0, -8, 8))
    test_y, test_prob, test_records = evaluate_model(model, test_loader, device=device, tta_shifts=(0, -8, 8))

    record_val = summarize_by_record(
        val_records,
        val_y,
        val_prob,
        quality_scores=raw_quality_scores[split_masks["val"]],
    )
    record_test = summarize_by_record(
        test_records,
        test_y,
        test_prob,
        quality_scores=raw_quality_scores[split_masks["test"]],
    )

    best_record_threshold = find_best_threshold(
        record_val["label"].to_numpy(dtype=np.int64),
        record_val["prob"].to_numpy(dtype=np.float32),
        objective=args.threshold_objective,
    )

    record_val_metrics = compute_metrics(
        record_val["label"].to_numpy(dtype=np.int64),
        record_val["prob"].to_numpy(dtype=np.float32),
        threshold=best_record_threshold,
    )
    record_test_metrics = compute_metrics(
        record_test["label"].to_numpy(dtype=np.int64),
        record_test["prob"].to_numpy(dtype=np.float32),
        threshold=best_record_threshold,
    )
    segment_val_metrics = compute_metrics(val_y, val_prob, threshold=best_record_threshold)
    segment_test_metrics = compute_metrics(test_y, test_prob, threshold=best_record_threshold)

    metrics = {
        "threshold_objective": args.threshold_objective,
        "best_record_threshold": best_record_threshold,
        "record_level_grouping": record_grouping,
        "segment_level": {"val": segment_val_metrics, "test": segment_test_metrics},
        "record_level": {"val": record_val_metrics, "test": record_test_metrics},
    }

    record_val.to_csv(args.output_dir / "val_record_predictions_weighted.csv", index=False)
    record_test.to_csv(args.output_dir / "test_record_predictions_weighted.csv", index=False)
    (args.output_dir / "record_level_metrics_weighted.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
