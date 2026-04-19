from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Zenodo preprocessing diagnostics from summary CSVs.")
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("/vol/bitbucket/mc1920/zenodo_build_by_subject"),
        help="Root containing subject-wise physio_distill bundles.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/homes/mc1920/FYP/Atrial-Fibrillation-ML-main/artifacts/visuals/zenodo_preprocessing_diagnostics.png"),
        help="Where to save the visualization PNG.",
    )
    return parser.parse_args()


def load_summary_dataframe(build_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(build_root.glob("*/physio_distill/physio_multimodal_segment_summary.csv")):
        df = pd.read_csv(csv_path, low_memory=False)
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        raise RuntimeError("No Zenodo segment summary CSVs found.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_summary_dataframe(args.build_root)
    accepted_mask = df["joint_accepted"] == True
    accepted_df = df.loc[accepted_mask].copy()
    rejected_df = df.loc[~accepted_mask].copy()

    accepted_count = int(accepted_df.shape[0])
    rejected_count = int(rejected_df.shape[0])
    total_count = accepted_count + rejected_count

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

    # Panel 1: accepted vs rejected
    axes[0, 0].bar(
        ["Accepted", "Rejected"],
        [accepted_count, rejected_count],
        color=["#54A24B", "#E45756"],
    )
    axes[0, 0].set_title("Zenodo Window Acceptance")
    axes[0, 0].set_ylabel("Window count")
    axes[0, 0].text(
        0.02,
        0.96,
        (
            f"Total windows: {total_count:,}\n"
            f"Accepted: {accepted_count:,} ({accepted_count / total_count:.1%})\n"
            f"Rejected: {rejected_count:,} ({rejected_count / total_count:.1%})"
        ),
        transform=axes[0, 0].transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "#CCCCCC"},
    )

    # Panel 2: quality score histogram
    bins = np.linspace(0.0, 1.0, 41)
    axes[0, 1].hist(
        rejected_df["ppg_quality_score"].dropna().to_numpy(),
        bins=bins,
        alpha=0.55,
        color="#E45756",
        label="Rejected",
        density=True,
    )
    axes[0, 1].hist(
        accepted_df["ppg_quality_score"].dropna().to_numpy(),
        bins=bins,
        alpha=0.55,
        color="#54A24B",
        label="Accepted",
        density=True,
    )
    axes[0, 1].set_title("PPG Quality Score Distribution")
    axes[0, 1].set_xlabel("ppg_quality_score")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend(frameon=False)

    # Panel 3: top rejection reasons
    top_reasons = (
        rejected_df["joint_rejection_reason"]
        .fillna("unknown")
        .value_counts()
        .head(8)
        .sort_values(ascending=True)
    )
    axes[1, 0].barh(top_reasons.index, top_reasons.values, color="#F58518")
    axes[1, 0].set_title("Top Joint Rejection Reasons")
    axes[1, 0].set_xlabel("Count")

    # Panel 4: accepted AF vs SR timing/quality sanity
    af_df = accepted_df.loc[accepted_df["label"] == 1]
    sr_df = accepted_df.loc[accepted_df["label"] == 0]
    axes[1, 1].scatter(
        sr_df["ppg_quality_score"],
        sr_df["timing_ibi_corr"],
        s=6,
        alpha=0.15,
        color="#4C78A8",
        label="Accepted SR",
    )
    axes[1, 1].scatter(
        af_df["ppg_quality_score"],
        af_df["timing_ibi_corr"],
        s=6,
        alpha=0.15,
        color="#E45756",
        label="Accepted AF",
    )
    axes[1, 1].set_title("Accepted Windows: PPG Quality vs ECG-PPG Timing Consistency")
    axes[1, 1].set_xlabel("ppg_quality_score")
    axes[1, 1].set_ylabel("timing_ibi_corr")
    axes[1, 1].set_xlim(0.0, 1.02)
    axes[1, 1].set_ylim(-1.02, 1.02)
    axes[1, 1].legend(frameon=False, markerscale=2)

    fig.suptitle(
        "Zenodo Preprocessing Diagnostics: Acceptance, Quality, Rejection Reasons, and ECG-PPG Alignment",
        fontsize=16,
    )
    fig.savefig(args.output_path, dpi=180, bbox_inches="tight")
    print(f"saved diagnostics to {args.output_path}")


if __name__ == "__main__":
    main()
