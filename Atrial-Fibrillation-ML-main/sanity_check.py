from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except Exception as exc:  # pragma: no cover - diagnostic script
    torch = None
    torch_import_error = exc
else:
    torch_import_error = None


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Check dataset artifacts and PyTorch CUDA visibility.")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=base_dir / "artifacts/zenodo10_full/signal_pipeline/ppg/ppg_accepted_segment_summary.csv",
        help="CSV summary to validate when data checks are enabled.",
    )
    parser.add_argument(
        "--segments-path",
        type=Path,
        default=base_dir / "artifacts/zenodo10_full/signal_pipeline/ppg/ppg_accepted_segments.npz",
        help="NPZ bundle to validate when data checks are enabled.",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip dataset checks and only report Python / torch / CUDA diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.summary_path.resolve()
    segments_path = args.segments_path.resolve()

    print(f"cwd={Path.cwd()}")
    print(f"summary_path={summary_path}")
    print(f"segments_path={segments_path}")

    if args.skip_data:
        print("data_checks=skipped")
    else:
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
        if not segments_path.exists():
            raise FileNotFoundError(f"Missing segments NPZ: {segments_path}")

        with segments_path.open("rb") as f:
            header = f.read(64)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                "Segments file is a Git LFS pointer, not real NPZ data. "
                "Run `git lfs pull` with working GitHub credentials first."
            )

        summary_df = pd.read_csv(summary_path)
        segments_npz = np.load(segments_path, allow_pickle=True)

        print(f"summary_shape={summary_df.shape}")
        print(f"summary_columns={list(summary_df.columns)}")
        print(f"npz_keys={segments_npz.files}")

        if "segments" in segments_npz.files:
            segments = segments_npz["segments"]
            print(f"segments_shape={segments.shape}")
            print(f"segments_dtype={segments.dtype}")
            if summary_df.shape[0] != segments.shape[0]:
                raise ValueError(
                    "Row count mismatch: "
                    f"summary rows={summary_df.shape[0]} vs segments rows={segments.shape[0]}"
                )
        else:
            print("WARNING: 'segments' key not found in NPZ.")

    if torch is None:
        print(f"torch_import_failed={torch_import_error}")
        return

    print(f"python_executable={Path(sys.executable)}")
    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"cuda_device_name={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
