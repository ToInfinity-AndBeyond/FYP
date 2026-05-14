from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a saved PPG experiment with threshold sweeps, record aggregation "
            "comparisons, and false-positive summaries."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Experiment directory containing metrics and prediction CSV files.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Directory to write analysis artifacts. Defaults to <output-dir>/analysis.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional operating threshold override. Defaults to metrics.json best_val_threshold when available.",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default=None,
        help="Column used to aggregate segment predictions into records. Defaults to group_id if present, else record_id.",
    )
    parser.add_argument(
        "--top-k-list",
        type=str,
        default="3,5",
        help="Comma-separated top-k values used for top-k mean aggregation methods.",
    )
    parser.add_argument(
        "--fp-top-n",
        type=int,
        default=500,
        help="Maximum number of highest-probability false positives to store per level.",
    )
    return parser.parse_args()


def load_operating_threshold(output_dir: Path, override: float | None) -> float:
    if override is not None:
        return override
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = metrics.get("best_val_threshold")
        if threshold is not None:
            return float(threshold)
    return 0.5


def normalize_for_json(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {key: normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_for_json(item) for item in value]
    return value


def build_threshold_sweep(labels: np.ndarray, probs: np.ndarray, thresholds: np.ndarray | None = None) -> pd.DataFrame:
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


def best_row(sweep_df: pd.DataFrame, metric: str = "f1") -> dict[str, Any]:
    if sweep_df.empty:
        return {}
    row = sweep_df.loc[sweep_df[metric].idxmax()].to_dict()
    return {key: normalize_for_json(value) for key, value in row.items()}


def parse_top_k_list(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        normalized = part.strip()
        if not normalized:
            continue
        values.append(int(normalized))
    return sorted(set(value for value in values if value > 0))


def detect_group_column(predictions: pd.DataFrame, override: str | None) -> str:
    if override is not None:
        if override not in predictions.columns:
            raise ValueError(f"Requested group column '{override}' not found in predictions.")
        return override
    if "group_id" in predictions.columns:
        return "group_id"
    return "record_id"


def get_quality_column(predictions: pd.DataFrame) -> str | None:
    for column in ("quality_score_runtime", "quality_score"):
        if column in predictions.columns:
            return column
    return None


def aggregate_segment_predictions(
    predictions: pd.DataFrame,
    group_column: str,
    operating_threshold: float,
    top_k_list: list[int],
) -> dict[str, pd.DataFrame]:
    if predictions.empty:
        return {}

    quality_column = get_quality_column(predictions)
    metadata_columns = [
        column
        for column in ("record_id", "subject_id", "event_id", "signal_file_name", "patient", "folder_path")
        if column in predictions.columns and column != group_column
    ]
    grouped = predictions.groupby(group_column, sort=False)
    metadata_spec: dict[str, tuple[str, str]] = {"label": ("label", "first"), "segment_count": ("prob", "size")}
    for column in metadata_columns:
        metadata_spec[column] = (column, "first")
    if quality_column is not None:
        metadata_spec["quality_mean"] = (quality_column, "mean")
    metadata = grouped.agg(**metadata_spec).reset_index()

    def finalize(score_series: pd.Series, method_name: str) -> pd.DataFrame:
        frame = metadata.copy()
        frame["method"] = method_name
        frame["score"] = frame[group_column].map(score_series).to_numpy(dtype=np.float32)
        return frame

    aggregations: dict[str, pd.DataFrame] = {}
    aggregations["mean"] = finalize(grouped["prob"].mean(), "mean")
    aggregations["median"] = finalize(grouped["prob"].median(), "median")
    aggregations["max"] = finalize(grouped["prob"].max(), "max")
    aggregations["positive_fraction_base_threshold"] = finalize(
        grouped["prob"].apply(lambda values: float((values.to_numpy() >= operating_threshold).mean())),
        "positive_fraction_base_threshold",
    )

    for top_k in top_k_list:
        aggregations[f"top{top_k}_mean"] = finalize(
            grouped["prob"].apply(lambda values, top_k=top_k: float(values.nlargest(min(len(values), top_k)).mean())),
            f"top{top_k}_mean",
        )

    if quality_column is not None:
        aggregations["quality_weighted_mean"] = finalize(
            grouped.apply(
                lambda group: float(
                    np.average(
                        group["prob"].to_numpy(dtype=np.float32),
                        weights=np.clip(group[quality_column].to_numpy(dtype=np.float32), 1e-6, None),
                    )
                )
            ),
            "quality_weighted_mean",
        )

    return aggregations


def analyze_aggregation_methods(
    val_segment_predictions: pd.DataFrame,
    test_segment_predictions: pd.DataFrame,
    group_column: str,
    operating_threshold: float,
    top_k_list: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_segment_predictions.empty or test_segment_predictions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    val_aggregations = aggregate_segment_predictions(
        predictions=val_segment_predictions,
        group_column=group_column,
        operating_threshold=operating_threshold,
        top_k_list=top_k_list,
    )
    test_aggregations = aggregate_segment_predictions(
        predictions=test_segment_predictions,
        group_column=group_column,
        operating_threshold=operating_threshold,
        top_k_list=top_k_list,
    )

    prediction_rows = []
    sweep_rows = []
    summary_rows = []

    for method_name, val_frame in val_aggregations.items():
        test_frame = test_aggregations[method_name]
        val_sweep = build_threshold_sweep(val_frame["label"].to_numpy(), val_frame["score"].to_numpy())
        test_sweep = build_threshold_sweep(test_frame["label"].to_numpy(), test_frame["score"].to_numpy())
        val_best = best_row(val_sweep, metric="f1")
        test_at_val_threshold = summarize_at_threshold(
            test_frame["label"].to_numpy(),
            test_frame["score"].to_numpy(),
            threshold=float(val_best["threshold"]),
        )
        test_best = best_row(test_sweep, metric="f1")

        val_long = val_frame.copy()
        val_long["split"] = "val"
        test_long = test_frame.copy()
        test_long["split"] = "test"
        prediction_rows.extend([val_long, test_long])

        val_sweep_long = val_sweep.copy()
        val_sweep_long["method"] = method_name
        val_sweep_long["split"] = "val"
        test_sweep_long = test_sweep.copy()
        test_sweep_long["method"] = method_name
        test_sweep_long["split"] = "test"
        sweep_rows.extend([val_sweep_long, test_sweep_long])

        summary_rows.append(
            {
                "method": method_name,
                "group_column": group_column,
                "val_selected_threshold": float(val_best["threshold"]),
                "val_f1": float(val_best["f1"]),
                "val_precision": float(val_best["precision"]),
                "val_sensitivity": float(val_best["sensitivity"]),
                "val_specificity": float(val_best["specificity"]),
                "test_f1_at_val_threshold": float(test_at_val_threshold["f1"]),
                "test_precision_at_val_threshold": float(test_at_val_threshold["precision"]),
                "test_sensitivity_at_val_threshold": float(test_at_val_threshold["sensitivity"]),
                "test_specificity_at_val_threshold": float(test_at_val_threshold["specificity"]),
                "test_accuracy_at_val_threshold": float(test_at_val_threshold["accuracy"]),
                "test_fp_at_val_threshold": int(test_at_val_threshold["fp"]),
                "test_fn_at_val_threshold": int(test_at_val_threshold["fn"]),
                "test_best_f1_analysis_only": float(test_best["f1"]),
                "test_best_threshold_analysis_only": float(test_best["threshold"]),
            }
        )

    predictions_long = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    sweeps_long = pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()
    summary_df = pd.DataFrame.from_records(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("test_f1_at_val_threshold", ascending=False).reset_index(drop=True)
    return predictions_long, sweeps_long, summary_df


def false_positive_analysis(
    predictions: pd.DataFrame,
    threshold: float,
    fp_top_n: int,
    group_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
    if predictions.empty:
        empty = pd.DataFrame()
        return empty, None, None, None, {}

    quality_column = get_quality_column(predictions)
    false_positives = predictions.loc[
        (predictions["label"].astype(np.int64) == 0) & (predictions["prob"] >= threshold)
    ].copy()
    false_positives = false_positives.sort_values("prob", ascending=False).reset_index(drop=True)
    false_positives["margin_to_threshold"] = false_positives["prob"] - threshold
    false_positives["fp_rank"] = np.arange(1, false_positives.shape[0] + 1, dtype=np.int64)
    false_positives_head = false_positives.head(fp_top_n).copy()

    group_summary = None
    if group_column in false_positives.columns:
        agg_spec: dict[str, tuple[str, str]] = {
            "fp_count": ("prob", "size"),
            "max_prob": ("prob", "max"),
            "mean_prob": ("prob", "mean"),
            "label": ("label", "first"),
        }
        for column in ("record_id", "subject_id", "event_id", "signal_file_name", "patient"):
            if column in false_positives.columns and column != group_column:
                agg_spec[column] = (column, "first")
        if quality_column is not None:
            agg_spec["quality_mean"] = (quality_column, "mean")
        group_summary = (
            false_positives.groupby(group_column, as_index=False)
            .agg(**agg_spec)
            .sort_values(["fp_count", "max_prob"], ascending=[False, False])
            .reset_index(drop=True)
        )

    subject_summary = None
    if "subject_id" in false_positives.columns:
        subject_summary = (
            false_positives.groupby("subject_id", as_index=False)
            .agg(
                fp_count=("prob", "size"),
                max_prob=("prob", "max"),
                mean_prob=("prob", "mean"),
            )
            .sort_values(["fp_count", "max_prob"], ascending=[False, False])
            .reset_index(drop=True)
        )

    quality_bins = None
    if quality_column is not None:
        binned = false_positives.copy()
        binned["quality_bin"] = pd.cut(
            binned[quality_column].astype(float),
            bins=[0.0, 0.5, 0.7, 0.85, 1.0],
            include_lowest=True,
        )
        quality_bins = (
            binned.groupby("quality_bin", as_index=False)
            .agg(
                fp_count=("prob", "size"),
                mean_prob=("prob", "mean"),
                max_prob=("prob", "max"),
            )
            .sort_values("quality_bin")
        )

    actual_negative = int((predictions["label"].astype(np.int64) == 0).sum())
    summary = {
        "operating_threshold": float(threshold),
        "fp_count": int(false_positives.shape[0]),
        "negative_examples": actual_negative,
        "fp_rate_among_negatives": float(false_positives.shape[0] / max(actual_negative, 1)),
        "mean_fp_prob": float(false_positives["prob"].mean()) if not false_positives.empty else 0.0,
        "median_fp_prob": float(false_positives["prob"].median()) if not false_positives.empty else 0.0,
        "max_fp_prob": float(false_positives["prob"].max()) if not false_positives.empty else 0.0,
        "top_fp_groups": (
            group_summary.head(10).to_dict(orient="records") if group_summary is not None and not group_summary.empty else []
        ),
        "top_fp_subjects": (
            subject_summary.head(10).to_dict(orient="records")
            if subject_summary is not None and not subject_summary.empty
            else []
        ),
    }
    return false_positives_head, group_summary, subject_summary, quality_bins, normalize_for_json(summary)


def save_dataframe(df: pd.DataFrame | None, path: Path) -> None:
    if df is None:
        return
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    analysis_dir = (args.analysis_dir or (output_dir / "analysis")).resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    operating_threshold = load_operating_threshold(output_dir, args.threshold)
    top_k_list = parse_top_k_list(args.top_k_list)

    test_segment_predictions = pd.read_csv(output_dir / "test_segment_predictions.csv")
    group_column = detect_group_column(test_segment_predictions, args.group_column)

    summary: dict[str, Any] = {
        "output_dir": str(output_dir),
        "analysis_dir": str(analysis_dir),
        "operating_threshold": operating_threshold,
        "group_column": group_column,
        "top_k_list": top_k_list,
        "levels": {},
        "aggregation": {},
        "false_positive_analysis": {},
    }

    level_frames = {}
    for level_name, filename in (
        ("segment_test", "test_segment_predictions.csv"),
        ("record_test", "test_record_predictions.csv"),
        ("segment_val", "val_segment_predictions.csv"),
        ("record_val", "val_record_predictions.csv"),
    ):
        path = output_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        level_frames[level_name] = frame
        sweep = build_threshold_sweep(frame["label"].to_numpy(dtype=np.int64), frame["prob"].to_numpy(dtype=np.float32))
        sweep_path = analysis_dir / f"{level_name}_threshold_sweep.csv"
        save_dataframe(sweep, sweep_path)
        summary["levels"][level_name] = {
            "at_operating_threshold": summarize_at_threshold(
                frame["label"].to_numpy(dtype=np.int64),
                frame["prob"].to_numpy(dtype=np.float32),
                threshold=operating_threshold,
            ),
            "best_f1_analysis_only": best_row(sweep, metric="f1"),
            "threshold_sweep_csv": str(sweep_path),
        }

    if "segment_val" in level_frames and "segment_test" in level_frames:
        aggregation_predictions, aggregation_sweeps, aggregation_summary = analyze_aggregation_methods(
            val_segment_predictions=level_frames["segment_val"],
            test_segment_predictions=level_frames["segment_test"],
            group_column=group_column,
            operating_threshold=operating_threshold,
            top_k_list=top_k_list,
        )
        if not aggregation_predictions.empty:
            predictions_path = analysis_dir / "aggregation_predictions_long.csv"
            sweeps_path = analysis_dir / "aggregation_threshold_sweeps_long.csv"
            summary_path = analysis_dir / "aggregation_method_summary.csv"
            aggregation_predictions.to_csv(predictions_path, index=False)
            aggregation_sweeps.to_csv(sweeps_path, index=False)
            aggregation_summary.to_csv(summary_path, index=False)
            summary["aggregation"] = {
                "predictions_long_csv": str(predictions_path),
                "sweeps_long_csv": str(sweeps_path),
                "summary_csv": str(summary_path),
                "best_method_by_test_f1_at_val_threshold": (
                    normalize_for_json(aggregation_summary.iloc[0].to_dict()) if not aggregation_summary.empty else None
                ),
            }

    for level_name in ("segment_test", "record_test"):
        if level_name not in level_frames:
            continue
        fp_frame, group_summary, subject_summary, quality_bins, fp_summary = false_positive_analysis(
            predictions=level_frames[level_name],
            threshold=operating_threshold,
            fp_top_n=args.fp_top_n,
            group_column=group_column if level_name.startswith("segment") else ("group_id" if "group_id" in level_frames[level_name].columns else "record_id"),
        )
        fp_path = analysis_dir / f"{level_name}_false_positives.csv"
        fp_group_path = analysis_dir / f"{level_name}_false_positives_by_group.csv"
        fp_subject_path = analysis_dir / f"{level_name}_false_positives_by_subject.csv"
        fp_quality_path = analysis_dir / f"{level_name}_false_positives_by_quality_bin.csv"
        save_dataframe(fp_frame, fp_path)
        save_dataframe(group_summary, fp_group_path)
        save_dataframe(subject_summary, fp_subject_path)
        save_dataframe(quality_bins, fp_quality_path)
        fp_summary.update(
            {
                "false_positive_csv": str(fp_path),
                "by_group_csv": str(fp_group_path) if group_summary is not None else None,
                "by_subject_csv": str(fp_subject_path) if subject_summary is not None else None,
                "by_quality_bin_csv": str(fp_quality_path) if quality_bins is not None else None,
            }
        )
        summary["false_positive_analysis"][level_name] = fp_summary

    summary_path = analysis_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(normalize_for_json(summary), indent=2), encoding="utf-8")
    print(json.dumps(normalize_for_json(summary), indent=2))
    print(f"Saved analysis summary to {summary_path}")


if __name__ == "__main__":
    main()
