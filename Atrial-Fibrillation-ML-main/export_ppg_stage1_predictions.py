from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from af_pipeline.data import PPGSegmentDataset, load_and_concat_signal_datasets
from af_pipeline.features import FEATURE_COLUMNS
from af_pipeline.splits import create_metadata_fold_split_masks, infer_record_grouping
from ppg_beatformer_model import QualityAwareBeatFormer
from ppg_hybrid_model import RhythmMorphologyFusionNet
from train_ppg_hybrid import build_segment_prediction_frame, evaluate_model


def parse_fold_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Fold list must not be empty.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export train/val/test segment predictions from a saved PPG model.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--segments-path", type=Path, nargs="+", required=True)
    parser.add_argument("--summary-path", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-folds", type=parse_fold_list, default=parse_fold_list("0,1,2,3,4,5,6,7"))
    parser.add_argument("--val-folds", type=parse_fold_list, default=parse_fold_list("8"))
    parser.add_argument("--test-folds", type=parse_fold_list, default=parse_fold_list("9"))
    return parser.parse_args()


def apply_saved_normalization(summary_df: pd.DataFrame, checkpoint: dict[str, object]) -> pd.DataFrame:
    normalized = summary_df.copy()
    medians = np.asarray(checkpoint["normalization"]["feature_medians"], dtype=np.float32)
    means = np.asarray(checkpoint["normalization"]["feature_means"], dtype=np.float32)
    stds = np.asarray(checkpoint["normalization"]["feature_stds"], dtype=np.float32)
    stds = np.where(stds < 1e-6, 1.0, stds)
    features = normalized[FEATURE_COLUMNS].copy().fillna(dict(zip(FEATURE_COLUMNS, medians.tolist())))
    values = (features.to_numpy(dtype=np.float32) - means) / stds
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    for index, column in enumerate(FEATURE_COLUMNS):
        normalized[column] = values[:, index].astype(np.float32)
    return normalized


def build_model(checkpoint: dict[str, object], signal_length: int) -> torch.nn.Module:
    config = checkpoint.get("model_config", {})
    if not isinstance(config, dict):
        config = {}
    architecture = config.get("model_architecture", "hybrid")
    if architecture == "qa_beatformer":
        return QualityAwareBeatFormer(
            feature_dim=int(config.get("feature_dim", len(FEATURE_COLUMNS))),
            signal_length=int(config.get("signal_length", signal_length)),
            max_beats=int(config.get("max_beats", 64)),
            beat_length=int(config.get("beat_length", 128)),
            d_model=int(config.get("d_model", 128)),
            num_layers=int(config.get("num_layers", 2)),
            num_heads=int(config.get("num_heads", 4)),
            dropout=float(config.get("dropout", 0.1)),
        )
    return RhythmMorphologyFusionNet(
        feature_dim=int(config.get("feature_dim", len(FEATURE_COLUMNS))),
        signal_length=int(config.get("signal_length", signal_length)),
        sqi_condition_indices=list(config.get("sqi_condition_indices", []) or []),
        active_branches=tuple(config.get("active_branches", ("time", "spectral", "feature"))),
    )


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
    signals, raw_summary_df = load_and_concat_signal_datasets(args.segments_path, args.summary_path)
    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    raw_quality_scores = pd.to_numeric(raw_summary_df["quality_score"], errors="coerce").to_numpy(dtype=np.float32)
    raw_quality_scores = np.nan_to_num(raw_quality_scores, nan=0.5, posinf=1.0, neginf=0.0)
    raw_quality_scores = np.clip(raw_quality_scores, 0.0, 1.0)

    split_masks, _ = create_metadata_fold_split_masks(
        raw_summary_df,
        train_folds=args.train_folds,
        val_folds=args.val_folds,
        test_folds=args.test_folds,
    )
    record_grouping, record_group_ids = infer_record_grouping(raw_summary_df)
    if record_group_ids is None:
        raise ValueError("Could not infer stable record grouping for exported predictions.")

    summary_df = apply_saved_normalization(raw_summary_df, checkpoint)
    summary_df = summary_df.copy()
    summary_df["_record_group_id"] = record_group_ids.to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint, signal_length=signals.shape[1]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metadata = {
        "checkpoint_path": str(args.checkpoint_path),
        "record_grouping": record_grouping,
        "splits": {name: int(mask.sum()) for name, mask in split_masks.items()},
    }
    for split_name, mask in split_masks.items():
        loader = build_loader(signals, summary_df, raw_quality_scores, mask, args.batch_size)
        labels, probs, grouped_records = evaluate_model(model, loader, device=device, tta_shifts=(0, -8, 8))
        frame = build_segment_prediction_frame(
            analysis_summary_df=raw_summary_df,
            split_mask=mask,
            grouped_record_ids=grouped_records,
            labels=labels,
            probs=probs,
            raw_quality_scores=raw_quality_scores,
        )
        frame.to_csv(args.output_dir / f"{split_name}_segment_predictions.csv", index=False)
        print(f"saved {split_name}: rows={frame.shape[0]} path={args.output_dir / f'{split_name}_segment_predictions.csv'}", flush=True)
    (args.output_dir / "export_metadata.json").write_text(__import__("json").dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
