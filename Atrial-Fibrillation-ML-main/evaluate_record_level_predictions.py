from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_ppg_hybrid import compute_metrics, summarize_by_record


def _parse_fold_list(value: str) -> list[int]:
    folds = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            folds.append(int(part))
    if not folds:
        raise argparse.ArgumentTypeError("Expected at least one fold id.")
    return folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute record-level metrics from saved segment predictions.")
    parser.add_argument(
        "--summary-path",
        type=Path,
        nargs="+",
        required=True,
        help="Accepted segment summary CSV paths used for the experiment.",
    )
    parser.add_argument(
        "--segment-predictions",
        type=Path,
        required=True,
        help="Path to test_segment_predictions.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Where to save record-level predictions CSV.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        required=True,
        help="Where to save record-level metrics JSON.",
    )
    parser.add_argument(
        "--test-folds",
        type=_parse_fold_list,
        default=_parse_fold_list("9"),
        help="Comma-separated metadata fold ids used for test split.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Decision threshold to use for record-level metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_df = pd.concat([pd.read_csv(path) for path in args.summary_path], ignore_index=True)
    test_mask = summary_df["strat_fold"].isin(args.test_folds).to_numpy()
    test_summary = summary_df.loc[test_mask].reset_index(drop=True)

    segment_predictions = pd.read_csv(args.segment_predictions).reset_index(drop=True)
    if len(segment_predictions) != len(test_summary):
        raise ValueError(
            "Segment prediction count does not match reconstructed test summary rows: "
            f"{len(segment_predictions)} vs {len(test_summary)}"
        )

    if not np.array_equal(segment_predictions["record_id"].astype(str).to_numpy(), test_summary["record_id"].astype(str).to_numpy()):
        raise ValueError("record_id order mismatch between test_segment_predictions.csv and reconstructed test summary.")

    if not np.array_equal(segment_predictions["label"].astype(int).to_numpy(), test_summary["label"].astype(int).to_numpy()):
        raise ValueError("label order mismatch between test_segment_predictions.csv and reconstructed test summary.")

    record_predictions = summarize_by_record(
        record_ids=segment_predictions["record_id"].astype(str).to_numpy(),
        labels=segment_predictions["label"].astype(np.int64).to_numpy(),
        probs=segment_predictions["prob"].astype(np.float32).to_numpy(),
        quality_scores=test_summary["quality_score"].to_numpy(dtype=np.float32),
    )

    record_metrics = compute_metrics(
        y_true=record_predictions["label"].to_numpy(dtype=np.int64),
        y_prob=record_predictions["prob"].to_numpy(dtype=np.float32),
        threshold=args.threshold,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    record_predictions.to_csv(args.output_path, index=False)
    args.metrics_output.write_text(json.dumps(record_metrics, indent=2), encoding="utf-8")

    print("Record-level metrics:", json.dumps(record_metrics, indent=2))
    print(f"Saved record predictions to: {args.output_path}")
    print(f"Saved record metrics to: {args.metrics_output}")


if __name__ == "__main__":
    main()
