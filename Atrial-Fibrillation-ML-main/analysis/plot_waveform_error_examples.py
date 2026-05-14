#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTCOMES = [
    ("tp", "True positive", 1, 1, "tab:green"),
    ("fp", "False positive", 0, 1, "tab:orange"),
    ("fn", "False negative", 1, 0, "tab:red"),
    ("tn", "True negative", 0, 0, "tab:blue"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot representative TP/FP/FN/TN PPG waveform examples.")
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=Path("/vol/bitbucket/mc1920/mimic_ext_p00_p01_p02_sqi_v2_ppg_1to2_20260426_194112"),
    )
    parser.add_argument(
        "--data-root-template",
        type=str,
        default="/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_{prefix}_by_fold/fold_9/ppg",
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--threshold", type=float, default=0.713)
    return parser.parse_args()


def choose_record_examples(record_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome, name, label, pred, _ in OUTCOMES:
        subset = record_df[
            (record_df["label"].astype(int) == label)
            & (record_df["predicted_label_at_best_threshold"].astype(int) == pred)
        ].copy()
        if subset.empty:
            raise RuntimeError(f"No records found for {name}")
        if outcome == "tp":
            subset["rank_score"] = -subset["prob"]
        elif outcome == "fp":
            subset["rank_score"] = -subset["prob"]
        elif outcome == "fn":
            subset["rank_score"] = subset["prob"]
        else:
            subset["rank_score"] = subset["prob"]
        chosen = subset.sort_values(["rank_score", "segment_count"], ascending=[True, False]).iloc[0].copy()
        chosen["outcome"] = outcome
        chosen["outcome_name"] = name
        rows.append(chosen)
    return pd.DataFrame(rows)


def choose_segment_for_record(segment_df: pd.DataFrame, record_row: pd.Series) -> pd.Series:
    group_id = record_row["group_id"]
    segments = segment_df[segment_df["group_id"] == group_id].copy()
    if segments.empty:
        raise RuntimeError(f"No segment rows found for {group_id}")

    outcome = record_row["outcome"]
    if outcome in {"tp", "fp"}:
        return segments.sort_values(["prob", "quality_score"], ascending=[False, False]).iloc[0]
    return segments.sort_values(["prob", "quality_score"], ascending=[True, False]).iloc[0]


def load_waveform(row: pd.Series, data_root_template: str) -> tuple[np.ndarray, pd.Series]:
    prefix = str(row["patient"])[:3]
    root = Path(data_root_template.format(prefix=prefix))
    summary_path = root / "ppg_accepted_segment_summary.csv"
    segments_path = root / "ppg_accepted_segments.npz"
    summary = pd.read_csv(summary_path, usecols=["signal_file_name", "quality_score", "event_rhythm"])
    matches = summary.index[summary["signal_file_name"] == row["signal_file_name"]].to_numpy()
    if matches.size != 1:
        raise RuntimeError(f"Expected one waveform match for {row['signal_file_name']}, found {matches.size}")
    idx = int(matches[0])
    segments = np.load(segments_path)["segments"]
    return segments[idx].astype(np.float32), summary.iloc[idx]


def main() -> int:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    record_df = pd.read_csv(args.prediction_dir / "test_record_predictions.csv")
    segment_df = pd.read_csv(args.prediction_dir / "test_segment_predictions.csv")

    chosen_records = choose_record_examples(record_df)
    chosen_rows = []
    waveforms = []
    for _, record_row in chosen_records.iterrows():
        segment_row = choose_segment_for_record(segment_df, record_row)
        waveform, source_row = load_waveform(segment_row, args.data_root_template)
        row = {
            "outcome": record_row["outcome"],
            "outcome_name": record_row["outcome_name"],
            "record_id": record_row["record_id"],
            "group_id": record_row["group_id"],
            "patient_prefix": str(segment_row["patient"])[:3],
            "signal_file_name": segment_row["signal_file_name"],
            "label": int(record_row["label"]),
            "predicted_label": int(record_row["predicted_label_at_best_threshold"]),
            "record_probability": float(record_row["prob"]),
            "segment_probability": float(segment_row["prob"]),
            "record_segment_count": int(record_row["segment_count"]),
            "record_quality_mean": float(record_row["quality_mean"]),
            "segment_quality": float(segment_row["quality_score"]),
            "template_correlation": float(segment_row["template_correlation"]),
            "estimated_hr_bpm": float(segment_row["estimated_hr_bpm"]),
            "event_rhythm": source_row.get("event_rhythm", ""),
        }
        chosen_rows.append(row)
        waveforms.append(waveform)

    chosen = pd.DataFrame(chosen_rows)
    chosen_path = args.figures_dir / "waveform_error_examples_selected.csv"
    chosen.to_csv(chosen_path, index=False)

    t = np.arange(waveforms[0].shape[0], dtype=np.float32) / 125.0
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.2), sharex=True)
    axes = axes.ravel()
    for ax, (_, name, _, _, color), row, waveform in zip(axes, OUTCOMES, chosen_rows, waveforms):
        ax.plot(t, waveform, color=color, linewidth=0.8)
        ax.axhline(0.0, color="0.75", linewidth=0.7)
        ref = "AF" if row["label"] == 1 else "SR"
        pred = "AF" if row["predicted_label"] == 1 else "SR"
        title = f"{name}: reference {ref}, predicted {pred}"
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.text(
            0.015,
            0.04,
            (
                f"record p={row['record_probability']:.3f}, segment p={row['segment_probability']:.3f}\n"
                f"SQI={row['segment_quality']:.3f}, template r={row['template_correlation']:.3f}, "
                f"HR={row['estimated_hr_bpm']:.1f} bpm"
            ),
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
        )
        ax.set_ylabel("PPG (z-score)")
        ax.set_xlim(0.0, 30.0)
    for ax in axes[2:]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Representative Test-Set PPG Waveforms by Prediction Outcome", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = args.figures_dir / "waveform_error_examples.png"
    pdf_path = args.figures_dir / "waveform_error_examples.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {chosen_path}")
    print(chosen[["outcome", "signal_file_name", "label", "predicted_label", "record_probability", "segment_probability", "segment_quality"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
