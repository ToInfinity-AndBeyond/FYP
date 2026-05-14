from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


MODEL_DIRS = {
    "waveform": "stage1_waveform",
    "gated": "stage1_gated",
    "qa_beatformer": "stage1_qa",
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        else:
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probs[mask].mean()))
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probs - labels) ** 2))


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    pred = (probs >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return {
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probs)) if np.unique(labels).size > 1 else float("nan"),
        "auprc": float(average_precision_score(labels, probs)) if np.unique(labels).size > 1 else float("nan"),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def best_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(
        np.clip(
            np.concatenate([np.linspace(0.001, 0.999, 999), np.quantile(probs, np.linspace(0.0, 1.0, 501))]),
            0.0,
            1.0,
        )
    )
    best_t = 0.5
    best_f1 = -1.0
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    for threshold in thresholds:
        score = f1_score(labels, probs >= threshold, zero_division=0)
        if score > best_f1:
            best_t = float(threshold)
            best_f1 = float(score)
    return best_t, best_f1


@dataclass
class Calibrator:
    name: str
    transform: Callable[[np.ndarray], np.ndarray]
    info: dict[str, float]


def fit_calibrators(probs: np.ndarray, labels: np.ndarray) -> list[Calibrator]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    logits = logit(probs)
    calibrators = [Calibrator("none", lambda p: np.asarray(p, dtype=np.float64), {})]

    temperatures = np.exp(np.linspace(np.log(0.05), np.log(20.0), 500))
    losses = []
    for temperature in temperatures:
        calibrated = sigmoid(logits / temperature)
        nll = -np.mean(labels * np.log(calibrated + 1e-8) + (1 - labels) * np.log(1 - calibrated + 1e-8))
        losses.append(nll)
    best_idx = int(np.argmin(losses))
    temperature = float(temperatures[best_idx])
    calibrators.append(
        Calibrator("temperature", lambda p, t=temperature: sigmoid(logit(p) / t), {"temperature": temperature})
    )

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, labels)
    calibrators.append(Calibrator("isotonic", lambda p, model=iso: model.transform(np.asarray(p)), {}))

    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(logits.reshape(-1, 1), labels)
    calibrators.append(
        Calibrator("platt", lambda p, model=platt: model.predict_proba(logit(p).reshape(-1, 1))[:, 1], {})
    )

    beta_features = np.column_stack([np.log(np.clip(probs, 1e-6, 1.0)), np.log(np.clip(1.0 - probs, 1e-6, 1.0))])
    beta = LogisticRegression(solver="lbfgs", max_iter=1000)
    beta.fit(beta_features, labels)
    calibrators.append(
        Calibrator(
            "beta_logistic",
            lambda p, model=beta: model.predict_proba(
                np.column_stack([np.log(np.clip(p, 1e-6, 1.0)), np.log(np.clip(1.0 - p, 1e-6, 1.0))])
            )[:, 1],
            {},
        )
    )
    return calibrators


def stratified_group_half_split(frame: pd.DataFrame, seed: int) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    groups = frame.groupby("group_id", as_index=False).agg(label=("label", "first"))
    cal_groups = []
    select_groups = []
    for label, label_groups in groups.groupby("label"):
        values = label_groups["group_id"].astype(str).to_numpy()
        rng.shuffle(values)
        midpoint = len(values) // 2
        cal_groups.extend(values[:midpoint])
        select_groups.extend(values[midpoint:])
    return set(cal_groups), set(select_groups)


def load_split(base_dir: Path, model_name: str, split: str) -> pd.DataFrame:
    path = base_dir / MODEL_DIRS[model_name] / f"{split}_segment_predictions.csv"
    frame = pd.read_csv(path)
    frame["group_id"] = frame["group_id"].astype(str)
    frame["prob"] = pd.to_numeric(frame["prob"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce").fillna(0).astype(int)
    for column in ("quality_score_runtime", "quality_score", "start_time_sec", "segment_index"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def aggregate_group(group: pd.DataFrame, method: str) -> float:
    probs = group["prob_cal"].to_numpy(dtype=np.float64)
    sqi_source = "quality_score_runtime" if "quality_score_runtime" in group.columns else "quality_score"
    sqi = group[sqi_source].fillna(0.5).clip(0.0, 1.0).to_numpy(dtype=np.float64) if sqi_source in group else None

    if method == "mean":
        return float(np.mean(probs))
    if method == "max":
        return float(np.max(probs))
    if method == "top5_quality_filter" and sqi is not None:
        mask = sqi >= np.nanmedian(sqi)
        filtered = probs[mask] if np.any(mask) else probs
        k = min(5, filtered.size)
        return float(np.mean(np.partition(filtered, -k)[-k:]))
    if method.startswith("top"):
        k = int(method[3:])
        k = min(k, probs.size)
        return float(np.mean(np.partition(probs, -k)[-k:]))
    if method == "trimmed_mean":
        if probs.size < 5:
            return float(np.mean(probs))
        lo, hi = np.quantile(probs, [0.1, 0.9])
        trimmed = probs[(probs >= lo) & (probs <= hi)]
        return float(np.mean(trimmed)) if trimmed.size else float(np.mean(probs))
    if method == "sqi_weighted_mean" and sqi is not None:
        weights = np.clip(sqi, 0.0, 1.0)
        return float(np.average(probs, weights=weights)) if weights.sum() > 0 else float(np.mean(probs))
    return float(np.mean(probs))


def record_predictions(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for group_id, group in frame.groupby("group_id", sort=False):
        probs = group["prob_cal"].to_numpy(dtype=np.float64)
        sqi_source = "quality_score_runtime" if "quality_score_runtime" in group.columns else "quality_score"
        sqi_mean = float(group[sqi_source].fillna(0.5).mean()) if sqi_source in group.columns else 0.5
        rows.append(
            {
                "group_id": group_id,
                "label": int(group["label"].iloc[0]),
                "prob": aggregate_group(group, method),
                "segment_count": int(group.shape[0]),
                "sqi_mean": sqi_mean,
                "segment_prob_std": float(np.std(probs)),
                "segment_prob_mean": float(np.mean(probs)),
            }
        )
    return pd.DataFrame.from_records(rows)


def record_predictions_all(frame: pd.DataFrame, methods: list[str]) -> dict[str, pd.DataFrame]:
    rows_by_method = {method: [] for method in methods}
    for group_id, group in frame.groupby("group_id", sort=False):
        probs = group["prob_cal"].to_numpy(dtype=np.float64)
        sqi_source = "quality_score_runtime" if "quality_score_runtime" in group.columns else "quality_score"
        sqi_values = (
            group[sqi_source].fillna(0.5).clip(0.0, 1.0).to_numpy(dtype=np.float64)
            if sqi_source in group.columns
            else np.full(probs.shape, 0.5, dtype=np.float64)
        )
        base = {
            "group_id": group_id,
            "label": int(group["label"].iloc[0]),
            "segment_count": int(group.shape[0]),
            "sqi_mean": float(np.mean(sqi_values)),
            "segment_prob_std": float(np.std(probs)),
            "segment_prob_mean": float(np.mean(probs)),
        }

        sorted_probs = np.sort(probs)
        top_means = {}
        for k in (3, 5, 10):
            kk = min(k, sorted_probs.size)
            top_means[f"top{k}"] = float(np.mean(sorted_probs[-kk:]))
        trimmed = probs
        if probs.size >= 5:
            lo, hi = np.quantile(probs, [0.1, 0.9])
            maybe_trimmed = probs[(probs >= lo) & (probs <= hi)]
            if maybe_trimmed.size:
                trimmed = maybe_trimmed
        high_quality = probs[sqi_values >= np.nanmedian(sqi_values)]
        if high_quality.size == 0:
            high_quality = probs
        k_hq = min(5, high_quality.size)
        values = {
            "mean": float(np.mean(probs)),
            "max": float(np.max(probs)),
            "top3": top_means["top3"],
            "top5": top_means["top5"],
            "top10": top_means["top10"],
            "trimmed_mean": float(np.mean(trimmed)),
            "sqi_weighted_mean": float(np.average(probs, weights=sqi_values)) if sqi_values.sum() > 0 else float(np.mean(probs)),
            "top5_quality_filter": float(np.mean(np.sort(high_quality)[-k_hq:])),
        }
        for method in methods:
            rows_by_method[method].append({**base, "prob": values[method]})
    return {method: pd.DataFrame.from_records(rows) for method, rows in rows_by_method.items()}


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        else:
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
        rows.append(
            {
                "bin_left": float(bins[i]),
                "bin_right": float(bins[i + 1]),
                "count": int(mask.sum()),
                "confidence": float(probs[mask].mean()) if mask.any() else np.nan,
                "accuracy": float(labels[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def coverage_curve(records: pd.DataFrame, confidence_name: str, confidence: np.ndarray, threshold: float) -> pd.DataFrame:
    labels = records["label"].to_numpy(dtype=np.int64)
    probs = records["prob"].to_numpy(dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    rows = []
    for coverage in np.arange(0.5, 1.001, 0.05):
        n_keep = max(int(round(len(records) * coverage)), 10)
        idx = np.argsort(-confidence)[:n_keep]
        metrics = binary_metrics(labels[idx], probs[idx], threshold)
        rows.append(
            {
                "confidence": confidence_name,
                "coverage": float(coverage),
                "n_kept": int(n_keep),
                **metrics,
            }
        )
    return pd.DataFrame.from_records(rows)


def plot_reliability(before: pd.DataFrame, after: pd.DataFrame, model: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.plot(before["confidence"], before["accuracy"], marker="o", label="Before")
    ax.plot(after["confidence"], after["accuracy"], marker="s", label="After")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(model)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_coverage(curves: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, group in curves.groupby("confidence"):
        ax.plot(group["coverage"], group["f1"], marker="o", label=name)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("F1")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strategy 1 calibration and Strategy 2 selective classification.")
    parser.add_argument("--prediction-base", type=Path, default=Path("/vol/bitbucket/mc1920/precisionguard_mil_20260514_063410"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/results/s1_s2_posthoc"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)

    aggregation_methods = ["mean", "max", "top3", "top5", "top10", "trimmed_mean", "sqi_weighted_mean", "top5_quality_filter"]
    calibration_rows = []
    aggregation_rows = []
    all_record_predictions: dict[tuple[str, str, str], dict[str, pd.DataFrame]] = {}
    best_by_model: dict[str, tuple[str, str, float]] = {}

    for model_name in MODEL_DIRS:
        val = load_split(args.prediction_base, model_name, "val")
        test = load_split(args.prediction_base, model_name, "test")
        cal_groups, select_groups = stratified_group_half_split(val, args.seed)
        val_cal = val[val["group_id"].isin(cal_groups)].copy()
        val_select = val[val["group_id"].isin(select_groups)].copy()

        calibrators = fit_calibrators(val_cal["prob"].to_numpy(), val_cal["label"].to_numpy())
        model_best = ("none", "mean", -1.0)
        for calibrator in calibrators:
            val_calibrated = val_select.copy()
            test_calibrated = test.copy()
            val_calibrated["prob_cal"] = calibrator.transform(val_calibrated["prob"].to_numpy())
            test_calibrated["prob_cal"] = calibrator.transform(test_calibrated["prob"].to_numpy())

            segment_ece_before = expected_calibration_error(test["prob"].to_numpy(), test["label"].to_numpy())
            segment_ece_after = expected_calibration_error(test_calibrated["prob_cal"].to_numpy(), test_calibrated["label"].to_numpy())
            segment_brier_before = brier_score(test["prob"].to_numpy(), test["label"].to_numpy())
            segment_brier_after = brier_score(test_calibrated["prob_cal"].to_numpy(), test_calibrated["label"].to_numpy())

            if calibrator.name != "none":
                before_bins = reliability_bins(test["prob"].to_numpy(), test["label"].to_numpy())
                after_bins = reliability_bins(test_calibrated["prob_cal"].to_numpy(), test_calibrated["label"].to_numpy())
                before_bins.to_csv(args.output_dir / f"reliability_{model_name}_{calibrator.name}_before.csv", index=False)
                after_bins.to_csv(args.output_dir / f"reliability_{model_name}_{calibrator.name}_after.csv", index=False)

            val_records_by_method = record_predictions_all(val_calibrated, aggregation_methods)
            test_records_by_method = record_predictions_all(test_calibrated, aggregation_methods)
            for method in aggregation_methods:
                val_records = val_records_by_method[method]
                test_records = test_records_by_method[method]
                threshold, val_f1 = best_threshold(val_records["label"].to_numpy(), val_records["prob"].to_numpy())
                val_metrics = binary_metrics(val_records["label"].to_numpy(), val_records["prob"].to_numpy(), threshold)
                test_metrics = binary_metrics(test_records["label"].to_numpy(), test_records["prob"].to_numpy(), threshold)
                row = {
                    "model": model_name,
                    "calibration": calibrator.name,
                    "aggregation": method,
                    "threshold": threshold,
                    "segment_ece_before": segment_ece_before,
                    "segment_ece_after": segment_ece_after,
                    "segment_brier_before": segment_brier_before,
                    "segment_brier_after": segment_brier_after,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    **calibrator.info,
                }
                calibration_rows.append(row)
                aggregation_rows.append(row)
                all_record_predictions[(model_name, calibrator.name, method)] = {"val": val_records, "test": test_records}
                if val_f1 > model_best[2]:
                    model_best = (calibrator.name, method, val_f1)
        best_by_model[model_name] = model_best

        best_cal, best_agg, _ = model_best
        before_bins = reliability_bins(test["prob"].to_numpy(), test["label"].to_numpy())
        cal = [c for c in calibrators if c.name == best_cal][0]
        after_probs = cal.transform(test["prob"].to_numpy())
        after_bins = reliability_bins(after_probs, test["label"].to_numpy())
        plot_reliability(before_bins, after_bins, model_name, args.output_dir / "figures" / f"reliability_{model_name}.png")

    calibration_df = pd.DataFrame(calibration_rows).sort_values(["val_f1", "test_f1"], ascending=False)
    aggregation_df = pd.DataFrame(aggregation_rows).sort_values(["model", "val_f1"], ascending=[True, False])
    calibration_df.to_csv(args.output_dir / "calibration_results.csv", index=False)
    aggregation_df.to_csv(args.output_dir / "aggregation_comparison.csv", index=False)

    best_global = calibration_df.iloc[0]
    best_key = (best_global["model"], best_global["calibration"], best_global["aggregation"])
    best_test_records = all_record_predictions[best_key]["test"].copy()
    best_threshold_value = float(best_global["threshold"])
    probs = best_test_records["prob"].to_numpy(dtype=np.float64)
    labels = best_test_records["label"].to_numpy(dtype=np.int64)
    p = np.clip(probs, 1e-7, 1.0 - 1e-7)
    confidence_curves = []
    confidence_curves.append(coverage_curve(best_test_records, "margin", np.abs(probs - 0.5), best_threshold_value))
    confidence_curves.append(coverage_curve(best_test_records, "neg_entropy", p * np.log(p) + (1 - p) * np.log(1 - p), best_threshold_value))
    confidence_curves.append(coverage_curve(best_test_records, "sqi_margin", np.abs(probs - 0.5) * best_test_records["sqi_mean"].to_numpy(), best_threshold_value))
    confidence_curves.append(coverage_curve(best_test_records, "segment_agreement", -best_test_records["segment_prob_std"].to_numpy(), best_threshold_value))
    coverage_df = pd.concat(confidence_curves, ignore_index=True)
    coverage_df.insert(0, "selected_model", best_key[0])
    coverage_df.insert(1, "selected_calibration", best_key[1])
    coverage_df.insert(2, "selected_aggregation", best_key[2])
    coverage_df.to_csv(args.output_dir / "coverage_curves.csv", index=False)
    plot_coverage(coverage_df, args.output_dir / "figures" / "selective_classification_main.png")

    summary = {
        "best_by_model": {
            model: {"calibration": cal, "aggregation": agg, "val_f1": f1} for model, (cal, agg, f1) in best_by_model.items()
        },
        "best_global": {
            "model": str(best_key[0]),
            "calibration": str(best_key[1]),
            "aggregation": str(best_key[2]),
            "threshold": best_threshold_value,
            "val_f1": float(best_global["val_f1"]),
            "test_f1": float(best_global["test_f1"]),
            "test_precision": float(best_global["test_precision"]),
            "test_sensitivity": float(best_global["test_sensitivity"]),
            "test_specificity": float(best_global["test_specificity"]),
            "test_auroc": float(best_global["test_auroc"]),
            "test_auprc": float(best_global["test_auprc"]),
        },
        "best_selective_by_f1": coverage_df.sort_values("f1", ascending=False).head(10).to_dict(orient="records"),
    }
    (args.output_dir / "numbers.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
