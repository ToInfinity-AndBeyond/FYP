from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from signal_pipeline import default_ppg_config


@dataclass(frozen=True)
class ThresholdCandidate:
    min_peak_count: int
    min_heart_band_energy_ratio: float
    max_abs_skewness: float
    min_template_correlation: float


def stratified_record_split(
    summary_df: pd.DataFrame,
    val_record_count: int,
    test_record_count: int,
    seed: int,
) -> dict[str, list[str]]:
    records = (
        summary_df.groupby("record_id", as_index=False)
        .agg(label=("label", "max"))
        .reset_index(drop=True)
    )
    pos_records = records.loc[records["label"] == 1, "record_id"].tolist()
    neg_records = records.loc[records["label"] == 0, "record_id"].tolist()

    rng = random.Random(seed)
    rng.shuffle(pos_records)
    rng.shuffle(neg_records)

    if len(records) < 3:
        return {
            "train": sorted(records["record_id"].tolist()),
            "val": [],
            "test": [],
        }

    total_records = len(records)
    test_pos = round(test_record_count * len(pos_records) / total_records)
    test_neg = max(0, test_record_count - test_pos)
    val_pos = round(val_record_count * len(pos_records) / total_records)
    val_neg = max(0, val_record_count - val_pos)

    test_records = pos_records[:test_pos] + neg_records[:test_neg]
    val_records = pos_records[test_pos : test_pos + val_pos] + neg_records[test_neg : test_neg + val_neg]
    train_records = pos_records[test_pos + val_pos :] + neg_records[test_neg + val_neg :]

    return {
        "train": sorted(train_records),
        "val": sorted(val_records),
        "test": sorted(test_records),
    }


def create_reference_quality_labels(
    summary_df: pd.DataFrame,
    min_matched_peaks: float,
    min_match_ratio: float,
    max_ibi_mae_ms: float,
    min_ibi_corr: float,
) -> pd.DataFrame:
    frame = summary_df.copy()
    ecg_ok = frame.get("ecg_accepted", pd.Series(True, index=frame.index)).fillna(False).astype(bool)
    matched = frame["timing_matched_peak_count"].fillna(0.0).to_numpy(dtype=np.float32)
    peak_count = frame["ppg_peak_count"].fillna(0.0).to_numpy(dtype=np.float32)
    match_ratio = matched / np.maximum(peak_count, 1.0)
    ibi_mae = frame["timing_ibi_mae_ms"].fillna(np.inf).to_numpy(dtype=np.float32)
    ibi_corr = frame["timing_ibi_corr"].fillna(-np.inf).to_numpy(dtype=np.float32)

    good = (
        ecg_ok.to_numpy()
        & (matched >= min_matched_peaks)
        & (match_ratio >= min_match_ratio)
        & (ibi_mae <= max_ibi_mae_ms)
        & (ibi_corr >= min_ibi_corr)
    )
    bad = (
        (~ecg_ok.to_numpy())
        | (matched < max(4.0, min_matched_peaks * 0.6))
        | (match_ratio < max(0.10, min_match_ratio * 0.6))
        | (ibi_mae >= max_ibi_mae_ms * 1.75)
        | (ibi_corr <= max(0.10, min_ibi_corr - 0.25))
    )

    reference_label = np.full(frame.shape[0], np.nan, dtype=np.float32)
    reference_label[good] = 1.0
    reference_label[bad & ~good] = 0.0
    frame["reference_quality_label"] = reference_label
    frame["reference_match_ratio"] = match_ratio
    return frame


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": f1,
        "score": float(balanced_accuracy + f1),
    }


def evaluate_candidate(frame: pd.DataFrame, candidate: ThresholdCandidate) -> dict[str, float]:
    y_true = frame["reference_quality_label"].to_numpy(dtype=np.int64)
    y_pred = (
        (frame["ppg_peak_count"].to_numpy(dtype=np.float32) >= candidate.min_peak_count)
        & (frame["ppg_heart_band_energy_ratio"].to_numpy(dtype=np.float32) >= candidate.min_heart_band_energy_ratio)
        & (np.abs(frame["ppg_signal_skewness"].to_numpy(dtype=np.float32)) <= candidate.max_abs_skewness)
        & (frame["ppg_template_correlation"].fillna(-np.inf).to_numpy(dtype=np.float32) >= candidate.min_template_correlation)
    ).astype(np.int64)
    return compute_metrics(y_true, y_pred)


def evaluate_on_subset(
    frame: pd.DataFrame,
    candidate: ThresholdCandidate,
) -> dict[str, float] | None:
    if frame.empty:
        return None
    y_true = frame["reference_quality_label"].to_numpy(dtype=np.int64)
    if np.unique(y_true).size < 2:
        return None
    return evaluate_candidate(frame, candidate)


def search_thresholds(train_df: pd.DataFrame) -> tuple[ThresholdCandidate, pd.DataFrame]:
    default_quality = default_ppg_config().quality

    peak_values = sorted(
        {
            int(round(value))
            for value in np.quantile(train_df["ppg_peak_count"].to_numpy(dtype=np.float32), [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
        }
        | {int(default_quality.min_peak_count)}
    )
    energy_values = sorted(
        {
            round(float(value), 3)
            for value in np.quantile(
                train_df["ppg_heart_band_energy_ratio"].to_numpy(dtype=np.float32),
                [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
            )
        }
        | {round(float(default_quality.min_heart_band_energy_ratio), 3)}
    )
    skew_values = sorted(
        {
            round(float(value), 3)
            for value in np.quantile(
                np.abs(train_df["ppg_signal_skewness"].to_numpy(dtype=np.float32)),
                [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
            )
        }
        | {round(float(default_quality.max_abs_skewness), 3)}
    )
    template_values = sorted(
        {
            round(float(value), 3)
            for value in np.quantile(
                train_df["ppg_template_correlation"].fillna(-np.inf).to_numpy(dtype=np.float32),
                [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
            )
            if np.isfinite(value)
        }
        | {round(float(default_quality.min_template_correlation), 3)}
    )

    candidates = []
    for min_peak_count, min_energy_ratio, max_abs_skewness, min_template_corr in product(
        peak_values,
        energy_values,
        skew_values,
        template_values,
    ):
        candidate = ThresholdCandidate(
            min_peak_count=int(min_peak_count),
            min_heart_band_energy_ratio=float(min_energy_ratio),
            max_abs_skewness=float(max_abs_skewness),
            min_template_correlation=float(min_template_corr),
        )
        metrics = evaluate_candidate(train_df, candidate)
        row = asdict(candidate)
        row.update({f"train_{key}": value for key, value in metrics.items()})
        candidates.append(row)

    ranking_df = pd.DataFrame(candidates).sort_values(
        by=["train_score", "train_balanced_accuracy", "train_f1", "min_peak_count"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best = ranking_df.iloc[0]
    return (
        ThresholdCandidate(
            min_peak_count=int(best["min_peak_count"]),
            min_heart_band_energy_ratio=float(best["min_heart_band_energy_ratio"]),
            max_abs_skewness=float(best["max_abs_skewness"]),
            min_template_correlation=float(best["min_template_correlation"]),
        ),
        ranking_df,
    )


def load_and_concat_summary_csvs(paths: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune PPG quality gate thresholds against ECG-derived timing consistency on the training set."
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        nargs="+",
        default=[Path("artifacts/physio_distill/physio_multimodal_segment_summary.csv")],
        help="One or more multimodal segment summary CSV paths",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/quality_tuning/ppg_quality_thresholds.json"),
        help="Where to save the tuned quality gate JSON",
    )
    parser.add_argument(
        "--output-ranking-csv",
        type=Path,
        default=Path("artifacts/quality_tuning/ppg_quality_threshold_ranking.csv"),
        help="Where to save the ranked threshold candidates",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-records", type=int, default=5)
    parser.add_argument("--test-records", type=int, default=5)
    parser.add_argument(
        "--include-record-ids",
        nargs="+",
        default=None,
        help="Optional subset of record_id values to use for tuning",
    )
    parser.add_argument("--reference-min-matched-peaks", type=float, default=10.0)
    parser.add_argument("--reference-min-match-ratio", type=float, default=0.30)
    parser.add_argument("--reference-max-ibi-mae-ms", type=float, default=120.0)
    parser.add_argument("--reference-min-ibi-corr", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df = load_and_concat_summary_csvs(args.summary_path)
    if args.include_record_ids is not None:
        summary_df = summary_df.loc[summary_df["record_id"].isin(args.include_record_ids)].copy()
    if summary_df.empty:
        raise ValueError("No rows available after loading/filtering the summary CSVs.")

    summary_df = create_reference_quality_labels(
        summary_df=summary_df,
        min_matched_peaks=args.reference_min_matched_peaks,
        min_match_ratio=args.reference_min_match_ratio,
        max_ibi_mae_ms=args.reference_max_ibi_mae_ms,
        min_ibi_corr=args.reference_min_ibi_corr,
    )
    summary_df = summary_df.loc[summary_df["reference_quality_label"].notna()].copy()
    if summary_df.empty:
        raise ValueError("Reference quality labeling produced no usable windows.")

    split_records = stratified_record_split(
        summary_df=summary_df,
        val_record_count=args.val_records,
        test_record_count=args.test_records,
        seed=args.seed,
    )
    if split_records["val"] or split_records["test"]:
        train_df = summary_df.loc[summary_df["record_id"].isin(split_records["train"])].copy()
        val_df = summary_df.loc[summary_df["record_id"].isin(split_records["val"])].copy()
        test_df = summary_df.loc[summary_df["record_id"].isin(split_records["test"])].copy()
    else:
        train_df = summary_df.copy()
        val_df = pd.DataFrame(columns=summary_df.columns)
        test_df = pd.DataFrame(columns=summary_df.columns)

    if np.unique(train_df["reference_quality_label"].to_numpy(dtype=np.int64)).size < 2:
        raise ValueError("Training subset has only one reference-quality class; add more records.")

    default_quality = default_ppg_config().quality
    default_candidate = ThresholdCandidate(
        min_peak_count=int(default_quality.min_peak_count),
        min_heart_band_energy_ratio=float(default_quality.min_heart_band_energy_ratio),
        max_abs_skewness=float(default_quality.max_abs_skewness),
        min_template_correlation=float(default_quality.min_template_correlation),
    )

    best_candidate, ranking_df = search_thresholds(train_df)

    report = {
        "reference_definition": {
            "min_matched_peaks": args.reference_min_matched_peaks,
            "min_match_ratio": args.reference_min_match_ratio,
            "max_ibi_mae_ms": args.reference_max_ibi_mae_ms,
            "min_ibi_corr": args.reference_min_ibi_corr,
        },
        "record_split": split_records,
        "dataset_counts": {
            "all_rows": int(summary_df.shape[0]),
            "train_rows": int(train_df.shape[0]),
            "val_rows": int(val_df.shape[0]),
            "test_rows": int(test_df.shape[0]),
            "record_count": int(summary_df["record_id"].nunique()),
        },
        "default_quality_overrides": asdict(default_candidate),
        "best_quality_overrides": asdict(best_candidate),
        "default_metrics": {
            "train": evaluate_on_subset(train_df, default_candidate),
            "val": evaluate_on_subset(val_df, default_candidate),
            "test": evaluate_on_subset(test_df, default_candidate),
        },
        "best_metrics": {
            "train": evaluate_on_subset(train_df, best_candidate),
            "val": evaluate_on_subset(val_df, best_candidate),
            "test": evaluate_on_subset(test_df, best_candidate),
        },
        "quality_overrides": asdict(best_candidate),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_ranking_csv.parent.mkdir(parents=True, exist_ok=True)
    ranking_df.to_csv(args.output_ranking_csv, index=False)

    print("reference windows:", int(summary_df.shape[0]))
    print("records:", int(summary_df["record_id"].nunique()))
    print("best quality overrides:", json.dumps(asdict(best_candidate), indent=2))
    print("default train metrics:", json.dumps(report["default_metrics"]["train"], indent=2))
    print("best train metrics:", json.dumps(report["best_metrics"]["train"], indent=2))
    if report["best_metrics"]["val"] is not None:
        print("best val metrics:", json.dumps(report["best_metrics"]["val"], indent=2))
    if report["best_metrics"]["test"] is not None:
        print("best test metrics:", json.dumps(report["best_metrics"]["test"], indent=2))
    print("saved quality json:", args.output_json)
    print("saved ranking csv:", args.output_ranking_csv)


if __name__ == "__main__":
    main()
