#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from final_paper_evidence import (
    FEATURE_COLUMNS,
    METRIC_COLUMNS,
    WeightedLogisticRegression,
    aggregate_records,
    balanced_sample_weight,
    best_f1_threshold,
    fold_summary_path,
    median_impute_and_scale,
    metrics_at_threshold,
    prefix_from_root,
    split_from_fold,
)


RHYTHM_FEATURES = [
    "peak_count",
    "estimated_hr_bpm",
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
]

QUALITY_FEATURES = [
    "heart_band_energy_ratio",
    "signal_skewness",
    "template_correlation",
    "quality_score",
    "signal_spectral_entropy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SQI/filter/feature ablations with feature-only logistic models.")
    parser.add_argument(
        "--bundle-roots",
        type=Path,
        nargs="+",
        default=[
            Path("/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p00_by_fold"),
            Path("/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p01_by_fold"),
            Path("/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p02_by_fold"),
        ],
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("Atrial-Fibrillation-ML-main/analysis/final_paper_sqi_feature_ablation"),
    )
    parser.add_argument("--maxiter", type=int, default=160)
    return parser.parse_args()


def make_group_id(frame: pd.DataFrame) -> pd.Series:
    event = frame["event_id"].fillna(-1).astype(int).astype(str) if "event_id" in frame.columns else "0"
    return frame["record_id"].astype(str) + "::event_" + event


def load_summary(bundle_roots: list[Path], accepted: bool) -> pd.DataFrame:
    usecols = sorted(
        set(
            FEATURE_COLUMNS
            + [
                "label",
                "record_id",
                "event_id",
                "subject_id",
                "quality_score",
                "folder_path",
                "accepted",
            ]
        )
    )
    frames = []
    for root in bundle_roots:
        prefix = prefix_from_root(root)
        for fold in range(10):
            path = fold_summary_path(root, fold, accepted=accepted)
            if not path.exists():
                continue
            print(f"[load] accepted={accepted} {path}", flush=True)
            frame = pd.read_csv(path, usecols=lambda col: col in usecols)
            frame["prefix"] = prefix
            frame["fold"] = fold
            frame["split"] = split_from_fold(fold)
            frame["group_id"] = make_group_id(frame)
            if "accepted" not in frame.columns:
                frame["accepted"] = True
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No {'accepted' if accepted else 'raw'} summaries found.")
    return pd.concat(frames, ignore_index=True)


def record_aggregate(segment_frame: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "quality_weighted":
        input_frame = segment_frame.rename(columns={"quality_score": "quality_score_runtime"})
        return aggregate_records(input_frame, method="quality_weighted_mean")
    if method == "mean":
        return aggregate_records(segment_frame, method="mean")
    raise ValueError(f"Unsupported aggregation method: {method}")


def train_and_evaluate(
    *,
    name: str,
    data: pd.DataFrame,
    feature_columns: list[str],
    aggregation: str,
    maxiter: int,
    analysis_dir: Path,
) -> dict[str, Any]:
    train = data.loc[data["split"] == "train"].copy()
    val = data.loc[data["split"] == "validation"].copy()
    test = data.loc[data["split"] == "test"].copy()

    x_train = train[feature_columns].to_numpy(dtype=np.float64)
    x_val = val[feature_columns].to_numpy(dtype=np.float64)
    x_test = test[feature_columns].to_numpy(dtype=np.float64)
    x_train, x_val, x_test = median_impute_and_scale(x_train, x_val, x_test)

    y_train = train["label"].to_numpy(dtype=np.int64)
    model = WeightedLogisticRegression(l2=1e-4, maxiter=maxiter)
    model.fit(x_train, y_train, sample_weight=balanced_sample_weight(y_train))

    val_seg = val[["record_id", "event_id", "subject_id", "folder_path", "group_id", "label", "quality_score"]].copy()
    test_seg = test[["record_id", "event_id", "subject_id", "folder_path", "group_id", "label", "quality_score"]].copy()
    val_seg["prob"] = model.predict_proba(x_val)[:, 1]
    test_seg["prob"] = model.predict_proba(x_test)[:, 1]

    val_rec = record_aggregate(val_seg, aggregation)
    test_rec = record_aggregate(test_seg, aggregation)
    threshold = best_f1_threshold(val_rec["label"].to_numpy(), val_rec["prob"].to_numpy())
    result = {
        "ablation": name,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "aggregation": aggregation,
        "threshold": threshold,
        "segment_test": metrics_at_threshold(test_seg["label"].to_numpy(), test_seg["prob"].to_numpy(), threshold),
        "record_test": metrics_at_threshold(test_rec["label"].to_numpy(), test_rec["prob"].to_numpy(), threshold),
        "counts": {
            "train_segments": int(train.shape[0]),
            "val_segments": int(val.shape[0]),
            "test_segments": int(test.shape[0]),
            "val_records": int(val_rec.shape[0]),
            "test_records": int(test_rec.shape[0]),
        },
    }
    val_rec.to_csv(analysis_dir / f"{name}_val_record_predictions.csv", index=False)
    test_rec.to_csv(analysis_dir / f"{name}_test_record_predictions.csv", index=False)
    return result


def main() -> int:
    args = parse_args()
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    raw = load_summary(args.bundle_roots, accepted=False)
    accepted = load_summary(args.bundle_roots, accepted=True)

    experiments = [
        {
            "name": "no_sqi_filter_rhythm_features_mean",
            "data": raw,
            "features": RHYTHM_FEATURES,
            "aggregation": "mean",
            "sqi_filtering": False,
            "sqi_features": False,
            "quality_weighted_aggregation": False,
        },
        {
            "name": "no_sqi_filter_all_features_mean",
            "data": raw,
            "features": FEATURE_COLUMNS,
            "aggregation": "mean",
            "sqi_filtering": False,
            "sqi_features": True,
            "quality_weighted_aggregation": False,
        },
        {
            "name": "sqi_filter_rhythm_features_mean",
            "data": accepted,
            "features": RHYTHM_FEATURES,
            "aggregation": "mean",
            "sqi_filtering": True,
            "sqi_features": False,
            "quality_weighted_aggregation": False,
        },
        {
            "name": "sqi_filter_all_features_mean",
            "data": accepted,
            "features": FEATURE_COLUMNS,
            "aggregation": "mean",
            "sqi_filtering": True,
            "sqi_features": True,
            "quality_weighted_aggregation": False,
        },
        {
            "name": "sqi_filter_all_features_quality_weighted",
            "data": accepted,
            "features": FEATURE_COLUMNS,
            "aggregation": "quality_weighted",
            "sqi_filtering": True,
            "sqi_features": True,
            "quality_weighted_aggregation": True,
        },
        {
            "name": "sqi_filter_quality_features_quality_weighted",
            "data": accepted,
            "features": QUALITY_FEATURES,
            "aggregation": "quality_weighted",
            "sqi_filtering": True,
            "sqi_features": True,
            "quality_weighted_aggregation": True,
        },
    ]

    results = []
    for experiment in experiments:
        print(f"[ablation] {experiment['name']}", flush=True)
        result = train_and_evaluate(
            name=experiment["name"],
            data=experiment["data"],
            feature_columns=experiment["features"],
            aggregation=experiment["aggregation"],
            maxiter=args.maxiter,
            analysis_dir=args.analysis_dir,
        )
        result.update(
            {
                "sqi_filtering": experiment["sqi_filtering"],
                "sqi_features": experiment["sqi_features"],
                "quality_weighted_aggregation": experiment["quality_weighted_aggregation"],
            }
        )
        results.append(result)

    rows = []
    for result in results:
        row = {
            "ablation": result["ablation"],
            "sqi_filtering": result["sqi_filtering"],
            "sqi_features": result["sqi_features"],
            "quality_weighted_aggregation": result["quality_weighted_aggregation"],
            "feature_count": result["feature_count"],
            "aggregation": result["aggregation"],
            "threshold": result["threshold"],
            **{f"record_test_{metric}": result["record_test"][metric] for metric in METRIC_COLUMNS},
            **{f"segment_test_{metric}": result["segment_test"][metric] for metric in METRIC_COLUMNS},
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path = args.analysis_dir / "sqi_feature_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    (args.analysis_dir / "sqi_feature_ablation_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
