#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata


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

METRIC_COLUMNS = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auroc", "auprc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final-paper evidence tables from saved PPG artifacts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/vol/bitbucket/mc1920/mimic_ext_p00_p01_p02_sqi_v2_ppg_1to2_20260426_194112"),
        help="Final hybrid experiment directory containing prediction CSVs and metrics.json.",
    )
    parser.add_argument(
        "--bundle-roots",
        type=Path,
        nargs="+",
        default=[
            Path("/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p00_by_fold"),
            Path("/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p01_by_fold"),
            Path("/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p02_by_fold"),
        ],
        help="Per-prefix by-fold dataset roots containing fold_*/ppg summaries.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("Atrial-Fibrillation-ML-main/analysis/final_paper_evidence"),
        help="Directory for output CSV/JSON artifacts.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--models",
        default="logistic",
        help="Comma-separated feature-only baselines to train. Currently supported: logistic.",
    )
    return parser.parse_args()


def split_from_fold(fold: int) -> str:
    if fold <= 7:
        return "train"
    if fold == 8:
        return "validation"
    if fold == 9:
        return "test"
    raise ValueError(f"Unexpected fold {fold}.")


def prefix_from_root(root: Path) -> str:
    name = root.name
    for part in name.split("_"):
        if part.startswith("p") and len(part) == 3 and part[1:].isdigit():
            return part
    # Handles names such as mimic_ext_ppg_sqi_v2_p00_by_fold.
    for token in ("p00", "p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09"):
        if token in name:
            return token
    raise ValueError(f"Could not infer prefix from {root}")


def fold_summary_path(root: Path, fold: int, accepted: bool) -> Path:
    filename = "ppg_accepted_segment_summary.csv" if accepted else "ppg_segment_summary.csv"
    return root / f"fold_{fold}" / "ppg" / filename


def make_group_id(frame: pd.DataFrame) -> pd.Series:
    if "group_id" in frame.columns:
        return frame["group_id"].astype(str)
    event = frame["event_id"].fillna(-1).astype(int).astype(str) if "event_id" in frame.columns else "0"
    return frame["record_id"].astype(str) + "::event_" + event


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.nan_to_num(np.asarray(y_prob, dtype=np.float64), nan=0.5, posinf=1.0, neginf=0.0)
    y_pred = (y_prob >= threshold).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
    if np.unique(y_true).size == 2:
        auroc = float(binary_auroc(y_true, y_prob))
        auprc = float(binary_average_precision(y_true, y_prob))
    else:
        auroc = float("nan")
        auprc = float("nan")
    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "auroc": auroc,
        "auprc": auprc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def binary_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(y_prob, method="average")
    pos_rank_sum = float(ranks[y_true == 1].sum())
    return (pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)


def binary_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_prob, kind="mergesort")
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    thresholds = np.unique(
        np.clip(
            np.concatenate([np.linspace(0.001, 0.999, 999), np.quantile(y_prob, np.linspace(0, 1, 401))]),
            0.0,
            1.0,
        )
    )
    rows = []
    for threshold in thresholds:
        row = metrics_at_threshold(y_true, y_prob, float(threshold))
        row["threshold"] = float(threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    sweep = threshold_sweep(y_true, y_prob)
    return float(sweep.loc[sweep["f1"].idxmax(), "threshold"])


def aggregate_records(segment_frame: pd.DataFrame, method: str = "quality_weighted_mean") -> pd.DataFrame:
    frame = segment_frame.copy()
    frame["group_id"] = make_group_id(frame)
    quality_col = "quality_score_runtime" if "quality_score_runtime" in frame.columns else "quality_score"
    rows = []
    for group_id, group in frame.groupby("group_id", sort=False):
        probs = group["prob"].to_numpy(dtype=np.float64)
        if method == "quality_weighted_mean":
            weights = np.clip(group[quality_col].to_numpy(dtype=np.float64), 1e-6, None)
            prob = float(np.average(probs, weights=weights))
        elif method == "mean":
            prob = float(np.mean(probs))
        elif method == "median":
            prob = float(np.median(probs))
        elif method == "max":
            prob = float(np.max(probs))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        first = group.iloc[0]
        rows.append(
            {
                "group_id": group_id,
                "record_id": str(first["record_id"]),
                "event_id": int(first["event_id"]) if "event_id" in group.columns and pd.notna(first["event_id"]) else 0,
                "subject_id": int(first["subject_id"]) if "subject_id" in group.columns and pd.notna(first["subject_id"]) else -1,
                "folder_path": str(first["folder_path"]) if "folder_path" in group.columns else "",
                "prefix": str(first["folder_path"])[:3] if "folder_path" in group.columns else "",
                "label": int(first["label"]),
                "prob": prob,
                "segment_count": int(group.shape[0]),
                "quality_mean": float(group[quality_col].mean()) if quality_col in group.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_coverage_tables(bundle_roots: list[Path], analysis_dir: Path) -> dict[str, str]:
    usecols = ["label", "accepted", "record_id", "event_id", "subject_id", "quality_score"]
    rows = []
    for root in bundle_roots:
        prefix = prefix_from_root(root)
        for fold in range(10):
            path = fold_summary_path(root, fold, accepted=False)
            if not path.exists():
                continue
            print(f"[coverage] reading {path}", flush=True)
            frame = pd.read_csv(path, usecols=usecols)
            frame["prefix"] = prefix
            frame["fold"] = fold
            frame["split"] = split_from_fold(fold)
            frame["accepted"] = frame["accepted"].astype(bool)
            frame["label"] = frame["label"].astype(int)
            for keys in (["split"], ["prefix"], ["split", "prefix"]):
                for group_key, group in frame.groupby(keys, sort=False):
                    if not isinstance(group_key, tuple):
                        group_key = (group_key,)
                    key_values = dict(zip(keys, group_key))
                    raw = int(group.shape[0])
                    accepted = int(group["accepted"].sum())
                    af = group["label"] == 1
                    sr = group["label"] == 0
                    af_total = int(af.sum())
                    sr_total = int(sr.sum())
                    af_accepted = int((group["accepted"] & af).sum())
                    sr_accepted = int((group["accepted"] & sr).sum())
                    rows.append(
                        {
                            "level": "+".join(keys),
                            **key_values,
                            "raw_segments_before_sqi": raw,
                            "accepted_segments": accepted,
                            "rejected_segments": raw - accepted,
                            "raw_af_segments": af_total,
                            "accepted_af_segments": af_accepted,
                            "raw_sr_segments": sr_total,
                            "accepted_sr_segments": sr_accepted,
                            "quality_sum_all": float(group["quality_score"].sum()),
                            "quality_count_all": int(group["quality_score"].notna().sum()),
                            "quality_sum_accepted": float(group.loc[group["accepted"], "quality_score"].sum()),
                            "quality_count_accepted": int(group.loc[group["accepted"], "quality_score"].notna().sum()),
                        }
                    )
    if not rows:
        raise FileNotFoundError("No segment summary files found for coverage analysis.")
    partial = pd.DataFrame(rows)

    def finalize(level: str, keys: list[str]) -> pd.DataFrame:
        frame = partial.loc[partial["level"] == level].copy()
        summed = frame.groupby(keys, sort=False).sum(numeric_only=True).reset_index()
        summed["acceptance_rate"] = summed["accepted_segments"] / summed["raw_segments_before_sqi"]
        summed["af_acceptance_rate"] = summed["accepted_af_segments"] / summed["raw_af_segments"].replace(0, np.nan)
        summed["sr_acceptance_rate"] = summed["accepted_sr_segments"] / summed["raw_sr_segments"].replace(0, np.nan)
        summed["mean_quality_all"] = summed["quality_sum_all"] / summed["quality_count_all"].replace(0, np.nan)
        summed["mean_quality_accepted"] = summed["quality_sum_accepted"] / summed["quality_count_accepted"].replace(0, np.nan)
        drop_cols = ["quality_sum_all", "quality_count_all", "quality_sum_accepted", "quality_count_accepted"]
        return summed.drop(columns=drop_cols)

    by_split = finalize("split", ["split"])
    by_prefix = finalize("prefix", ["prefix"])
    by_split_prefix = finalize("split+prefix", ["split", "prefix"])

    paths = {
        "coverage_by_split": str(analysis_dir / "sqi_coverage_by_split.csv"),
        "coverage_by_prefix": str(analysis_dir / "sqi_coverage_by_prefix.csv"),
        "coverage_by_split_prefix": str(analysis_dir / "sqi_coverage_by_split_prefix.csv"),
    }
    by_split.to_csv(paths["coverage_by_split"], index=False)
    by_prefix.to_csv(paths["coverage_by_prefix"], index=False)
    by_split_prefix.to_csv(paths["coverage_by_split_prefix"], index=False)
    return paths


def load_accepted_features(bundle_roots: list[Path]) -> pd.DataFrame:
    usecols = FEATURE_COLUMNS + ["label", "record_id", "event_id", "subject_id", "quality_score", "folder_path"]
    frames = []
    for root in bundle_roots:
        prefix = prefix_from_root(root)
        for fold in range(10):
            path = fold_summary_path(root, fold, accepted=True)
            if not path.exists():
                continue
            print(f"[features] reading {path}", flush=True)
            frame = pd.read_csv(path, usecols=usecols)
            frame["prefix"] = prefix
            frame["fold"] = fold
            frame["split"] = split_from_fold(fold)
            frame["group_id"] = make_group_id(frame)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No accepted segment summary files found for feature baselines.")
    return pd.concat(frames, ignore_index=True)


class WeightedLogisticRegression:
    def __init__(self, l2: float = 1e-4, maxiter: int = 200):
        self.l2 = float(l2)
        self.maxiter = int(maxiter)
        self.coef_: np.ndarray | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray) -> None:
        x_train = np.asarray(x_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        design = np.column_stack([x_train, np.ones(x_train.shape[0], dtype=np.float64)])
        n_features = design.shape[1]
        denom = float(sample_weight.sum())

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            logits = design @ params
            # Stable weighted binary cross-entropy with logits.
            losses = np.logaddexp(0.0, logits) - y_train * logits
            penalty = 0.5 * self.l2 * float(np.dot(params[:-1], params[:-1]))
            value = float(np.dot(sample_weight, losses) / denom + penalty)
            probs = expit(logits)
            grad = design.T @ (sample_weight * (probs - y_train)) / denom
            grad[:-1] += self.l2 * params[:-1]
            return value, grad

        result = minimize(
            objective,
            np.zeros(n_features, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.maxiter, "ftol": 1e-8, "maxls": 50},
        )
        self.coef_ = result.x

    def predict_proba(self, x_values: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted.")
        x_values = np.asarray(x_values, dtype=np.float64)
        design = np.column_stack([x_values, np.ones(x_values.shape[0], dtype=np.float64)])
        probs = expit(design @ self.coef_)
        return np.column_stack([1.0 - probs, probs])


def median_impute_and_scale(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(train_x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)

    def impute(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        mask = ~np.isfinite(values)
        if mask.any():
            values = values.copy()
            values[mask] = np.take(medians, np.where(mask)[1])
        return values

    train_x = impute(train_x)
    val_x = impute(val_x)
    test_x = impute(test_x)
    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0)
    stds = np.where(stds > 0.0, stds, 1.0)
    return (train_x - means) / stds, (val_x - means) / stds, (test_x - means) / stds


def balanced_sample_weight(y_train: np.ndarray) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=np.int64)
    n_total = y_train.size
    weights = np.ones(n_total, dtype=np.float64)
    for label in (0, 1):
        count = int((y_train == label).sum())
        if count:
            weights[y_train == label] = n_total / (2.0 * count)
    return weights


def evaluate_feature_model(
    model_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    analysis_dir: Path,
) -> dict[str, Any]:
    x_train = train[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    x_val = val[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    x_test = test[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    x_train, x_val, x_test = median_impute_and_scale(x_train, x_val, x_test)

    y_train = train["label"].to_numpy(dtype=np.int64)
    sample_weight = balanced_sample_weight(y_train)
    model = WeightedLogisticRegression(l2=1e-4, maxiter=250)
    model.fit(x_train, y_train, sample_weight=sample_weight)

    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    val_seg = val[["record_id", "event_id", "subject_id", "folder_path", "group_id", "label", "quality_score"]].copy()
    val_seg["prob"] = val_prob
    test_seg = test[["record_id", "event_id", "subject_id", "folder_path", "group_id", "label", "quality_score"]].copy()
    test_seg["prob"] = test_prob

    val_rec = aggregate_records(val_seg.rename(columns={"quality_score": "quality_score_runtime"}))
    test_rec = aggregate_records(test_seg.rename(columns={"quality_score": "quality_score_runtime"}))

    threshold = best_f1_threshold(val_rec["label"].to_numpy(), val_rec["prob"].to_numpy())
    result = {
        "model": model_name,
        "val_selected_record_threshold": threshold,
        "segment_val": metrics_at_threshold(val_seg["label"].to_numpy(), val_seg["prob"].to_numpy(), threshold),
        "segment_test": metrics_at_threshold(test_seg["label"].to_numpy(), test_seg["prob"].to_numpy(), threshold),
        "record_val": metrics_at_threshold(val_rec["label"].to_numpy(), val_rec["prob"].to_numpy(), threshold),
        "record_test": metrics_at_threshold(test_rec["label"].to_numpy(), test_rec["prob"].to_numpy(), threshold),
        "counts": {
            "train_segments": int(train.shape[0]),
            "val_segments": int(val.shape[0]),
            "test_segments": int(test.shape[0]),
            "val_records": int(val_rec.shape[0]),
            "test_records": int(test_rec.shape[0]),
        },
    }

    val_rec.to_csv(analysis_dir / f"{model_name}_val_record_predictions.csv", index=False)
    test_rec.to_csv(analysis_dir / f"{model_name}_test_record_predictions.csv", index=False)
    val_seg.to_csv(analysis_dir / f"{model_name}_val_segment_predictions.csv", index=False)
    test_seg.to_csv(analysis_dir / f"{model_name}_test_segment_predictions.csv", index=False)
    return result


def run_feature_baselines(bundle_roots: list[Path], analysis_dir: Path, model_names: list[str]) -> dict[str, Any]:
    data = load_accepted_features(bundle_roots)
    train = data.loc[data["split"] == "train"].copy()
    val = data.loc[data["split"] == "validation"].copy()
    test = data.loc[data["split"] == "test"].copy()

    results: dict[str, Any] = {}
    if "logistic" in model_names:
        results["feature_logistic_regression"] = evaluate_feature_model(
            "feature_logistic_regression", train, val, test, analysis_dir
        )

    unsupported = sorted(set(model_names) - {"logistic"})
    if unsupported:
        raise ValueError(f"Unsupported model(s): {unsupported}. Supported models: logistic.")

    baseline_rows = []
    for name, result in results.items():
        row = {"model": name, "threshold": result["val_selected_record_threshold"]}
        row.update({f"record_test_{metric}": result["record_test"][metric] for metric in METRIC_COLUMNS})
        row.update({f"segment_test_{metric}": result["segment_test"][metric] for metric in METRIC_COLUMNS})
        baseline_rows.append(row)
    pd.DataFrame(baseline_rows).to_csv(analysis_dir / "feature_baseline_summary.csv", index=False)
    return results


def bootstrap_ci(
    frame: pd.DataFrame,
    threshold: float,
    iterations: int,
    seed: int,
    cluster_column: str | None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in METRIC_COLUMNS}
    if cluster_column and cluster_column in frame.columns:
        clusters = frame[cluster_column].dropna().unique()
        grouped_indices = {cluster: frame.index[frame[cluster_column] == cluster].to_numpy() for cluster in clusters}
        for _ in range(iterations):
            sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
            sampled_indices = np.concatenate([grouped_indices[cluster] for cluster in sampled_clusters])
            sample = frame.loc[sampled_indices]
            metrics = metrics_at_threshold(sample["label"].to_numpy(), sample["prob"].to_numpy(), threshold)
            for metric in METRIC_COLUMNS:
                values[metric].append(metrics[metric])
    else:
        indices = frame.index.to_numpy()
        for _ in range(iterations):
            sampled_indices = rng.choice(indices, size=len(indices), replace=True)
            sample = frame.loc[sampled_indices]
            metrics = metrics_at_threshold(sample["label"].to_numpy(), sample["prob"].to_numpy(), threshold)
            for metric in METRIC_COLUMNS:
                values[metric].append(metrics[metric])

    point = metrics_at_threshold(frame["label"].to_numpy(), frame["prob"].to_numpy(), threshold)
    rows = []
    for metric in METRIC_COLUMNS:
        samples = np.asarray(values[metric], dtype=np.float64)
        samples = samples[np.isfinite(samples)]
        rows.append(
            {
                "metric": metric,
                "value": point[metric],
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "bootstrap": "clustered" if cluster_column else "record_event",
                "cluster_column": cluster_column or "",
                "iterations": iterations,
            }
        )
    return pd.DataFrame(rows)


def build_prediction_based_tables(output_dir: Path, analysis_dir: Path, iterations: int, seed: int) -> dict[str, str]:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    threshold = float(metrics["best_val_threshold"])
    record_test = pd.read_csv(output_dir / "test_record_predictions.csv")
    record_test["prefix"] = record_test["folder_path"].astype(str).str.slice(0, 3)

    ci_record = bootstrap_ci(record_test, threshold, iterations, seed, cluster_column=None)
    ci_subject = bootstrap_ci(record_test, threshold, iterations, seed + 1, cluster_column="subject_id")
    ci = pd.concat([ci_record, ci_subject], ignore_index=True)
    ci_path = analysis_dir / "record_level_bootstrap_ci.csv"
    ci.to_csv(ci_path, index=False)

    prefix_rows = []
    for prefix, group in record_test.groupby("prefix", sort=True):
        row = {
            "prefix": prefix,
            "records": int(group.shape[0]),
            "af_records": int((group["label"] == 1).sum()),
            "mean_segment_count": float(group["segment_count"].mean()),
            "mean_quality": float(group["quality_mean"].mean()),
            "median_af_probability": float(group.loc[group["label"] == 1, "prob"].median()),
        }
        row.update(metrics_at_threshold(group["label"].to_numpy(), group["prob"].to_numpy(), threshold))
        prefix_rows.append(row)
    prefix_path = analysis_dir / "prefix_reliability_summary.csv"
    pd.DataFrame(prefix_rows).to_csv(prefix_path, index=False)
    return {
        "record_level_bootstrap_ci": str(ci_path),
        "prefix_reliability_summary": str(prefix_path),
    }


def main() -> int:
    args = parse_args()
    analysis_dir = args.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]

    print("[coverage] building SQI coverage tables", flush=True)
    coverage_paths = build_coverage_tables(args.bundle_roots, analysis_dir)

    print("[predictions] building bootstrap CIs and prefix reliability tables", flush=True)
    prediction_paths = build_prediction_based_tables(
        args.output_dir, analysis_dir, args.bootstrap_iterations, args.bootstrap_seed
    )

    print(f"[baselines] training feature-only models: {', '.join(model_names)}", flush=True)
    baseline_results = run_feature_baselines(args.bundle_roots, analysis_dir, model_names)

    summary = {
        "analysis_dir": str(analysis_dir.resolve()),
        "coverage_paths": coverage_paths,
        "prediction_paths": prediction_paths,
        "feature_baselines": baseline_results,
    }
    summary_path = analysis_dir / "final_paper_evidence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
