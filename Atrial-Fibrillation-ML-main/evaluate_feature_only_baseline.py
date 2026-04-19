#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "peak_count",
    "heart_band_energy_ratio",
    "signal_skewness",
    "template_correlation",
    "estimated_hr_bpm",
    "quality_score",
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
    "signal_spectral_entropy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--train-folds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--val-folds", default="8")
    parser.add_argument("--test-folds", default="9")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def parse_fold_list(spec: str) -> list[int]:
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def load_fold(bundle_root: Path, fold: int) -> pd.DataFrame:
    path = bundle_root / f"fold_{fold}" / "ppg" / "ppg_accepted_segment_summary.csv"
    return pd.read_csv(path, usecols=FEATURE_COLUMNS + ["label"])


def load_folds(bundle_root: Path, folds: list[int]) -> pd.DataFrame:
    return pd.concat([load_fold(bundle_root, fold) for fold in folds], ignore_index=True)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, sensitivity, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        y_pred = (y_prob >= threshold).astype(np.int32)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def main() -> int:
    args = parse_args()
    bundle_root = Path(args.bundle_root)
    train_folds = parse_fold_list(args.train_folds)
    val_folds = parse_fold_list(args.val_folds)
    test_folds = parse_fold_list(args.test_folds)

    train_df = load_folds(bundle_root, train_folds)
    val_df = load_folds(bundle_root, val_folds)
    test_df = load_folds(bundle_root, test_folds)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x_train = imputer.fit_transform(train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    x_val = imputer.transform(val_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    x_test = imputer.transform(test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))

    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    y_train = train_df["label"].to_numpy(dtype=np.int32)
    y_val = val_df["label"].to_numpy(dtype=np.int32)
    y_test = test_df["label"].to_numpy(dtype=np.int32)

    model = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    model.fit(x_train, y_train)

    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    best_threshold = find_best_threshold(y_val, val_prob)

    result = {
        "model": "logistic_regression_feature_only",
        "feature_columns": FEATURE_COLUMNS,
        "train_folds": train_folds,
        "val_folds": val_folds,
        "test_folds": test_folds,
        "best_val_threshold": best_threshold,
        "val": compute_metrics(y_val, val_prob, best_threshold),
        "test": compute_metrics(y_test, test_prob, best_threshold),
        "counts": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_positive_rate": float(y_train.mean()),
            "val_positive_rate": float(y_val.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
    }

    output = json.dumps(result, indent=2)
    print(output)
    if args.output_json:
        Path(args.output_json).write_text(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
