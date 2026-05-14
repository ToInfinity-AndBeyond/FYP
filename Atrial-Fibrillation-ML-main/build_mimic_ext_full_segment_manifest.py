#!/usr/bin/env python3
"""Build a full eligible p03-p09 segment manifest for MIMIC-III-Ext-PPG."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_METADATA = Path("/vol/bitbucket/mc1920/FYP/metadata.csv")
DEFAULT_OUTPUT_CSV = Path("artifacts/mimic_ext_subject_selection/full_p03_p09_eligible_segments.csv")
DEFAULT_PREFIXES = ("p03", "p04", "p05", "p06", "p07", "p08", "p09")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a full eligible p03-p09 segment manifest.")
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--prefixes", nargs="+", default=list(DEFAULT_PREFIXES))
    parser.add_argument("--progress-every", type=int, default=500_000)
    return parser.parse_args()


def count_good_pleth_segments(raw_value: str) -> tuple[int, int]:
    if raw_value is None:
        return 0, 0
    text = str(raw_value).strip()
    if not text or text.lower() == "nan":
        return 0, 0
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return 0, 0
    total = 0
    good = 0
    for part in text.split(","):
        token = part.strip()
        if not token or token.lower() == "nan":
            continue
        total += 1
        if token == "1":
            good += 1
    return good, total


def resolve_wfdb_record_path(folder_path: str, signal_file_name: str) -> str:
    folder = Path(str(folder_path))
    signal_name = str(signal_file_name)
    if folder.name == signal_name:
        return folder.as_posix()
    return (folder / signal_name).as_posix()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    prefixes = set(args.prefixes)
    rows_scanned = 0
    eligible_rows = 0
    subject_counts: dict[int, Counter] = defaultdict(Counter)
    subject_prefix: dict[int, str] = {}
    prefix_counts = Counter()
    label_counts = Counter()

    fieldnames = [
        "subject_id",
        "prefix",
        "subject_folder",
        "subject_folder_rel",
        "label_name",
        "label",
        "folder_path",
        "signal_file_name",
        "record_id",
        "event_id",
        "segment_id",
        "start_segment",
        "event_time",
        "pleth_good_fraction",
        "wfdb_record_path",
        "strat_fold",
    ]

    with args.metadata_csv.open("r", encoding="utf-8", newline="") as handle, args.output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as out_handle:
        reader = csv.DictReader(handle)
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            rows_scanned += 1
            if args.progress_every > 0 and rows_scanned % args.progress_every == 0:
                print(
                    f"[progress] scanned {rows_scanned:,} rows eligible={eligible_rows:,} subjects={len(subject_counts):,}",
                    flush=True,
                )

            folder_path = str(row["folder_path"])
            prefix = folder_path.split("/", 1)[0]
            if prefix not in prefixes:
                continue

            rhythm = str(row["event_rhythm"])
            if rhythm not in {"AF", "SR"}:
                continue

            good_count, total_count = count_good_pleth_segments(row.get("vector_10s_pleth_sqi"))
            if total_count == 0:
                continue
            good_fraction = good_count / total_count
            if good_count < 2 or good_fraction < (2.0 / 3.0):
                continue

            subject_id = int(row["subject_id"])
            label = 1 if rhythm == "AF" else 0
            writer.writerow(
                {
                    "subject_id": subject_id,
                    "prefix": prefix,
                    "subject_folder": f"p{subject_id:06d}",
                    "subject_folder_rel": f"{prefix}/p{subject_id:06d}",
                    "label_name": rhythm,
                    "label": label,
                    "folder_path": folder_path,
                    "signal_file_name": str(row["signal_file_name"]),
                    "record_id": str(row.get("record_id", "")),
                    "event_id": str(row.get("event_id", "")),
                    "segment_id": str(row.get("segment_id", "")),
                    "start_segment": str(row.get("start_segment", "")),
                    "event_time": str(row.get("event_time", "")),
                    "pleth_good_fraction": f"{good_fraction:.6f}",
                    "wfdb_record_path": resolve_wfdb_record_path(folder_path, str(row["signal_file_name"])),
                    "strat_fold": str(row.get("strat_fold", "")),
                }
            )

            eligible_rows += 1
            subject_prefix[subject_id] = prefix
            subject_counts[subject_id]["total"] += 1
            subject_counts[subject_id][rhythm] += 1
            prefix_counts[prefix] += 1
            label_counts[rhythm] += 1

    subject_summary_csv = args.output_csv.with_name(args.output_csv.stem + "_subject_summary.csv")
    with subject_summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject_id", "subject_folder_rel", "selected_total", "selected_af", "selected_sr"],
        )
        writer.writeheader()
        for subject_id, counts in sorted(subject_counts.items()):
            prefix = subject_prefix[subject_id]
            writer.writerow(
                {
                    "subject_id": subject_id,
                    "subject_folder_rel": f"{prefix}/p{subject_id:06d}",
                    "selected_total": int(counts["total"]),
                    "selected_af": int(counts["AF"]),
                    "selected_sr": int(counts["SR"]),
                }
            )

    summary = {
        "rows_scanned": rows_scanned,
        "eligible_segments": eligible_rows,
        "eligible_subjects": len(subject_counts),
        "prefix_counts": dict(sorted(prefix_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "output_csv": str(args.output_csv),
        "subject_summary_csv": str(subject_summary_csv),
    }
    summary_json = args.output_csv.with_name(args.output_csv.stem + "_summary.json")
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
