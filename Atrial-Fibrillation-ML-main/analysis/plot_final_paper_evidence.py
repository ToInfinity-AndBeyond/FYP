#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final-paper record-level ROC/PR/calibration figures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/vol/bitbucket/mc1920/mimic_ext_p00_p01_p02_sqi_v2_ppg_1to2_20260426_194112"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("Atrial-Fibrillation-ML-main/figures"),
    )
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def roc_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_prob, kind="mergesort")
    y = y_true[order].astype(np.int64)
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = np.concatenate([[0.0], tp / positives, [1.0]])
    fpr = np.concatenate([[0.0], fp / negatives, [1.0]])
    return fpr, tpr


def pr_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_prob, kind="mergesort")
    y = y_true[order].astype(np.int64)
    positives = max(int((y == 1).sum()), 1)
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    precision = np.concatenate([[precision[0] if precision.size else 1.0], precision, [0.0]])
    recall = np.concatenate([[0.0], recall, [1.0]])
    return recall, precision


def calibration_points(y_true: np.ndarray, y_prob: np.ndarray, bins: int) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for idx in range(bins):
        low = edges[idx]
        high = edges[idx + 1]
        if idx == bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_low": low,
                "bin_high": high,
                "count": int(mask.sum()),
                "mean_predicted_probability": float(y_prob[mask].mean()),
                "observed_af_fraction": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def load_record_predictions(output_dir: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(output_dir / f"{split}_record_predictions.csv")
    return frame[["label", "prob"]].dropna()


def main() -> int:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
    threshold = float(metrics["best_val_threshold"])

    val = load_record_predictions(args.output_dir, "val")
    test = load_record_predictions(args.output_dir, "test")

    plt.figure(figsize=(10, 4.5))
    for split_name, frame in (("Validation", val), ("Test", test)):
        y = frame["label"].to_numpy(dtype=np.int64)
        p = frame["prob"].to_numpy(dtype=np.float64)
        fpr, tpr = roc_curve_points(y, p)
        recall, precision = pr_curve_points(y, p)
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=split_name)
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=split_name)
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Record-Level ROC")
    plt.legend(frameon=False)
    plt.subplot(1, 2, 2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Record-Level Precision-Recall")
    plt.legend(frameon=False)
    plt.tight_layout()
    roc_pr_path = args.figures_dir / "record_level_roc_pr_curves.png"
    plt.savefig(roc_pr_path, dpi=220)
    plt.close()

    calib = calibration_points(test["label"].to_numpy(dtype=np.int64), test["prob"].to_numpy(dtype=np.float64), args.bins)
    calib_path = args.figures_dir / "record_level_calibration_test.csv"
    calib.to_csv(calib_path, index=False)

    plt.figure(figsize=(5.2, 4.5))
    plt.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="Perfect calibration")
    plt.plot(
        calib["mean_predicted_probability"],
        calib["observed_af_fraction"],
        marker="o",
        label="Test bins",
    )
    plt.axvline(threshold, color="0.25", linestyle=":", linewidth=1, label=f"Threshold {threshold:.3f}")
    plt.xlabel("Mean predicted AF probability")
    plt.ylabel("Observed AF fraction")
    plt.title("Record-Level Calibration")
    plt.legend(frameon=False)
    plt.tight_layout()
    calibration_path = args.figures_dir / "record_level_calibration.png"
    plt.savefig(calibration_path, dpi=220)
    plt.close()

    print(f"Saved {roc_pr_path}")
    print(f"Saved {calibration_path}")
    print(f"Saved {calib_path}")

    saved_analysis_dir = Path("analysis/final_paper_saved_prediction_analysis")
    threshold_sweep_path = saved_analysis_dir / "record_test_threshold_sweep.csv"
    if threshold_sweep_path.exists():
        sweep = pd.read_csv(threshold_sweep_path)
        plt.figure(figsize=(7.0, 4.4))
        plt.plot(sweep["threshold"], sweep["sensitivity"], label="Sensitivity")
        plt.plot(sweep["threshold"], sweep["precision"], label="Precision")
        plt.plot(sweep["threshold"], sweep["f1"], label="F1")
        plt.axvline(threshold, color="0.25", linestyle=":", linewidth=1, label=f"Validation threshold {threshold:.3f}")
        plt.xlabel("Decision threshold")
        plt.ylabel("Metric value")
        plt.title("Record-Level Threshold Trade-Off")
        plt.ylim(0.0, 1.02)
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        threshold_path = args.figures_dir / "record_level_threshold_tradeoff.png"
        plt.savefig(threshold_path, dpi=220)
        plt.close()
        print(f"Saved {threshold_path}")

    evidence_dir = Path("analysis/final_paper_evidence")
    prefix_path = evidence_dir / "prefix_reliability_summary.csv"
    if prefix_path.exists():
        prefix = pd.read_csv(prefix_path)
        metrics_to_plot = ["sensitivity", "specificity", "precision", "f1"]
        x = np.arange(len(prefix))
        width = 0.19
        plt.figure(figsize=(7.2, 4.6))
        for idx, metric in enumerate(metrics_to_plot):
            plt.bar(x + (idx - 1.5) * width, prefix[metric], width=width, label=metric.capitalize())
        plt.xticks(x, prefix["prefix"])
        plt.ylim(0.0, 1.05)
        plt.xlabel("MIMIC-III-Ext-PPG prefix")
        plt.ylabel("Record-level metric")
        plt.title("Record-Level Performance by Prefix")
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        prefix_fig_path = args.figures_dir / "prefix_record_performance.png"
        plt.savefig(prefix_fig_path, dpi=220)
        plt.close()
        print(f"Saved {prefix_fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
