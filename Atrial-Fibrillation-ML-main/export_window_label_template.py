from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from signal_pipeline import infer_label_from_path


ORIGINAL_METADATA_PATTERN = re.compile(
    r"<Original Subject ID>:\s*(?P<subject_id>\S+)\s+"
    r"<Original Recording ID>:\s*(?P<recording_id>\S+)\s+"
    r"<Original File>:\s*(?P<original_file>.+?)\s+<subject group>:"
)


def discover_csvs(dataset_root: Path) -> list[Path]:
    patterns = [
        "mimic_perform_af_csv/*_data.csv",
        "mimic_perform_non_af_csv/*_data.csv",
    ]
    csv_paths: list[Path] = []
    for pattern in patterns:
        csv_paths.extend(sorted(dataset_root.glob(pattern)))
    return sorted(csv_paths)


def corresponding_header_path(dataset_root: Path, csv_path: Path) -> Path:
    wfdb_parent = csv_path.parent.name.replace("_csv", "_wfdb")
    record_id = csv_path.stem.replace("_data", "")
    return dataset_root / wfdb_parent / f"{record_id}.hea"


def parse_original_metadata(hea_path: Path) -> dict[str, str]:
    if not hea_path.exists():
        return {
            "original_subject_id": "",
            "original_recording_id": "",
            "original_file": "",
        }

    header_text = hea_path.read_text(encoding="utf-8", errors="ignore")
    match = ORIGINAL_METADATA_PATTERN.search(header_text)
    if not match:
        return {
            "original_subject_id": "",
            "original_recording_id": "",
            "original_file": "",
        }

    original_file = match.group("original_file").split(";")[0].strip()
    return {
        "original_subject_id": match.group("subject_id").strip(),
        "original_recording_id": match.group("recording_id").strip(),
        "original_file": original_file,
    }


def build_template(
    dataset_root: Path,
    segment_length_sec: float,
    stride_sec: float,
    sample_rate_hz: float,
    seed_labels_from_path: bool,
    limit_files: int | None,
) -> pd.DataFrame:
    csv_paths = discover_csvs(dataset_root)
    if limit_files is not None:
        csv_paths = csv_paths[:limit_files]

    segment_samples = int(round(segment_length_sec * sample_rate_hz))
    stride_samples = int(round(stride_sec * sample_rate_hz))
    rows: list[dict[str, object]] = []

    for csv_path in csv_paths:
        dataframe = pd.read_csv(csv_path, usecols=["Time"])
        record_id = csv_path.stem.replace("_data", "")
        time_values = dataframe["Time"].to_numpy(dtype=float)
        header_path = corresponding_header_path(dataset_root, csv_path)
        metadata = parse_original_metadata(header_path)
        path_label = infer_label_from_path(csv_path)

        for start in range(0, time_values.size - segment_samples + 1, stride_samples):
            end = start + segment_samples
            segment_index = int(start // stride_samples)
            row = {
                "record_id": record_id,
                "csv_parent": csv_path.parent.name,
                "csv_file": csv_path.name,
                "hea_file": header_path.name,
                "original_subject_id": metadata["original_subject_id"],
                "original_recording_id": metadata["original_recording_id"],
                "original_file": metadata["original_file"],
                "segment_index": segment_index,
                "start_sample": int(start),
                "end_sample": int(end),
                "start_time_sec": float(time_values[start]),
                "end_time_sec": float(time_values[end - 1]),
                "path_label": int(path_label),
                "label": float(path_label) if seed_labels_from_path else np.nan,
                "use_for_training": int(seed_labels_from_path),
                "label_source": "path_seed" if seed_labels_from_path else "unlabeled_template",
                "notes": "",
            }
            rows.append(row)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a per-window label template for the MIMIC PERform AF dataset. "
            "Use this to merge official ECG annotation intervals or manual ECG review into segment-level labels."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root that contains mimic_perform_af_csv and mimic_perform_non_af_csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "labeling" / "window_label_template.csv",
        help="Where to write the exported template CSV",
    )
    parser.add_argument(
        "--segment-length-sec",
        type=float,
        default=30.0,
        help="Window length in seconds",
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=30.0,
        help="Window stride in seconds",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=125.0,
        help="Sampling rate used to convert window seconds to samples",
    )
    parser.add_argument(
        "--seed-labels-from-path",
        action="store_true",
        help=(
            "Fill the label column with current AF/non-AF folder labels. "
            "Useful as a starting point, but overwrite these with ECG-based window labels for rigorous experiments."
        ),
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional limit for smoke tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_df = build_template(
        dataset_root=args.dataset_root,
        segment_length_sec=args.segment_length_sec,
        stride_sec=args.stride_sec,
        sample_rate_hz=args.sample_rate_hz,
        seed_labels_from_path=args.seed_labels_from_path,
        limit_files=args.limit_files,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    template_df.to_csv(args.output_csv, index=False)

    labeled_count = int(template_df["label"].notna().sum())
    print(f"windows exported: {int(template_df.shape[0])}")
    print(f"windows with seeded labels: {labeled_count}")
    print(f"output_csv: {args.output_csv}")


if __name__ == "__main__":
    main()
