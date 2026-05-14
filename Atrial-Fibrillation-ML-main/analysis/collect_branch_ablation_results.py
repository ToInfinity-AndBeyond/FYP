#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


VARIANTS = ("full_fusion", "waveform_only", "spectral_only", "feature_only")
METRICS = ("accuracy", "sensitivity", "specificity", "precision", "f1", "auroc", "auprc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect branch-ablation metrics from completed training runs.")
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("/vol/bitbucket/mc1920/mimic_ext_p00_p01_p02_branch_ablation_20260506_155101"),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("Atrial-Fibrillation-ML-main/analysis/final_paper_branch_ablation_summary.csv"),
    )
    return parser.parse_args()


def get_nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def main() -> int:
    args = parse_args()
    rows = []
    for variant in VARIANTS:
        metrics_path = args.output_base / variant / "metrics.json"
        if not metrics_path.exists():
            rows.append({"model_variant": variant, "status": "missing", "metrics_path": str(metrics_path)})
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "model_variant": variant,
            "status": "complete",
            "metrics_path": str(metrics_path),
            "epochs_ran": data.get("epochs_ran"),
            "best_epoch": data.get("best_epoch"),
            "best_val_threshold": data.get("best_val_threshold"),
            "runtime_seconds": data.get("runtime_seconds"),
        }
        for metric in METRICS:
            row[f"record_test_{metric}"] = get_nested(data, "record_level", "test", metric)
            row[f"segment_test_{metric}"] = get_nested(data, "segment_level", "test", metric)
        rows.append(row)

    summary = pd.DataFrame(rows)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved {args.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
