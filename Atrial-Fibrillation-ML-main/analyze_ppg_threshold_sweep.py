from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create threshold-sweep summaries from saved PPG prediction CSV files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Experiment directory containing prediction CSV files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional operating threshold to summarize. Defaults to metrics.json best_val_threshold when available.",
    )
    return parser.parse_args()


def build_threshold_sweep(labels: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.nan_to_num(np.asarray(probs, dtype=np.float32), nan=0.5, posinf=1.0, neginf=0.0)
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


def summarize_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.nan_to_num(np.asarray(probs, dtype=np.float32), nan=0.5, posinf=1.0, neginf=0.0)
    predictions = (probs >= threshold).astype(np.int64)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    f1 = (2.0 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) else 0.0
    return {
        "threshold": float(threshold),
        "n": int(labels.size),
        "actual_positive": int((labels == 1).sum()),
        "actual_negative": int((labels == 0).sum()),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "predicted_positive": int(predictions.sum()),
        "predicted_negative": int((predictions == 0).sum()),
    }


def best_row(sweep: pd.DataFrame, metric: str = "f1") -> dict[str, Any]:
    if sweep.empty:
        return {}
    row = sweep.loc[sweep[metric].idxmax()].to_dict()
    return {key: value.item() if hasattr(value, "item") else value for key, value in row.items()}


def load_threshold(output_dir: Path, override: float | None) -> float:
    if override is not None:
        return override
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "best_val_threshold" in metrics:
            return float(metrics["best_val_threshold"])
    return 0.5


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    threshold = load_threshold(output_dir, args.threshold)

    summary: dict[str, Any] = {"operating_threshold": threshold, "levels": {}}
    for level, prediction_name in (
        ("segment_test", "test_segment_predictions.csv"),
        ("record_test", "test_record_predictions.csv"),
        ("segment_val", "val_segment_predictions.csv"),
        ("record_val", "val_record_predictions.csv"),
    ):
        prediction_path = output_dir / prediction_name
        if not prediction_path.exists():
            continue
        predictions = pd.read_csv(prediction_path)
        labels = predictions["label"].astype(int).to_numpy()
        probs = predictions["prob"].to_numpy(dtype=np.float32)
        sweep = build_threshold_sweep(labels, probs)
        sweep_path = output_dir / f"{level}_threshold_sweep.csv"
        sweep.to_csv(sweep_path, index=False)
        summary["levels"][level] = {
            "at_operating_threshold": summarize_at_threshold(labels, probs, threshold),
            "best_f1_analysis_only": best_row(sweep, metric="f1"),
            "threshold_sweep_csv": str(sweep_path),
        }

    summary_path = output_dir / "threshold_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved threshold sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
