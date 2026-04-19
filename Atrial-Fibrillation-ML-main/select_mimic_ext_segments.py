#!/usr/bin/env python3
"""Create a segment-level download manifest from a selected MIMIC-III-Ext-PPG subject cohort."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_METADATA = Path("/vol/bitbucket/mc1920/FYP/metadata.csv")
DEFAULT_SUBJECTS_CSV = Path("artifacts/mimic_ext_subject_selection/selected_subjects.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/mimic_ext_subject_selection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a partial-download segment manifest for selected subjects.")
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--subjects-csv", type=Path, default=DEFAULT_SUBJECTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-total-per-subject", type=int, default=256)
    parser.add_argument("--max-af-per-subject", type=int, default=128)
    parser.add_argument("--max-sr-per-subject", type=int, default=128)
    parser.add_argument("--max-per-record-per-class", type=int, default=24)
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


def even_sample(items: list[dict[str, str]], target_count: int) -> list[dict[str, str]]:
    if target_count <= 0 or not items:
        return []
    if len(items) <= target_count:
        return list(items)
    if target_count == 1:
        return [items[len(items) // 2]]
    indices = [round(i * (len(items) - 1) / (target_count - 1)) for i in range(target_count)]
    sampled = []
    last_index = None
    for index in indices:
        if index == last_index:
            continue
        sampled.append(items[index])
        last_index = index
    if len(sampled) < target_count:
        seen = {id(item) for item in sampled}
        for item in items:
            if id(item) in seen:
                continue
            sampled.append(item)
            if len(sampled) == target_count:
                break
    return sampled[:target_count]


def sort_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("record_id", "")),
        str(row.get("start_segment", "")),
        str(row.get("signal_file_name", "")),
    )


def resolve_wfdb_record_path(folder_path: str, signal_file_name: str) -> str:
    folder = Path(str(folder_path))
    signal_name = str(signal_file_name)
    if folder.name == signal_name:
        return folder.as_posix()
    return (folder / signal_name).as_posix()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_subjects = []
    selected_subject_ids = set()
    with args.subjects_csv.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            subject_id = int(row["subject_id"])
            selected_subject_ids.add(subject_id)
            selected_subjects.append(
                {
                    "subject_id": subject_id,
                    "subject_rank": index + 1,
                    "prefix": row["prefix"],
                    "subject_folder": row["subject_folder"],
                    "subject_folder_rel": row["subject_folder_rel"],
                }
            )

    subject_lookup = {row["subject_id"]: row for row in selected_subjects}
    candidates: dict[int, dict[str, list[dict[str, str]]]] = defaultdict(lambda: {"AF": [], "SR": []})
    scanned_rows = 0

    with args.metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scanned_rows += 1
            if args.progress_every > 0 and scanned_rows % args.progress_every == 0:
                print(f"[progress] scanned {scanned_rows:,} metadata rows", flush=True)

            rhythm = str(row["event_rhythm"])
            if rhythm not in {"AF", "SR"}:
                continue

            subject_id = int(row["subject_id"])
            if subject_id not in selected_subject_ids:
                continue

            good_count, total_count = count_good_pleth_segments(row.get("vector_10s_pleth_sqi"))
            if total_count == 0:
                continue
            good_fraction = good_count / total_count
            if good_count < 2 or good_fraction < (2.0 / 3.0):
                continue

            candidates[subject_id][rhythm].append(
                {
                    "subject_id": str(subject_id),
                    "prefix": subject_lookup[subject_id]["prefix"],
                    "subject_folder": subject_lookup[subject_id]["subject_folder"],
                    "subject_folder_rel": subject_lookup[subject_id]["subject_folder_rel"],
                    "subject_rank": str(subject_lookup[subject_id]["subject_rank"]),
                    "label_name": rhythm,
                    "label": "1" if rhythm == "AF" else "0",
                    "folder_path": str(row["folder_path"]),
                    "signal_file_name": str(row["signal_file_name"]),
                    "record_id": str(row.get("record_id", "")),
                    "event_id": str(row.get("event_id", "")),
                    "segment_id": str(row.get("segment_id", "")),
                    "start_segment": str(row.get("start_segment", "")),
                    "event_time": str(row.get("event_time", "")),
                    "pleth_good_fraction": f"{good_fraction:.6f}",
                    "wfdb_record_path": resolve_wfdb_record_path(row["folder_path"], row["signal_file_name"]),
                }
            )

    manifest_rows = []
    subject_summary = []
    for subject in selected_subjects:
        subject_id = subject["subject_id"]
        label_limits = {"AF": args.max_af_per_subject, "SR": args.max_sr_per_subject}
        chosen_rows = []
        label_counts = {}
        for label_name in ("AF", "SR"):
            rows = sorted(candidates[subject_id][label_name], key=sort_key)
            by_record: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                by_record[row["record_id"]].append(row)

            trimmed_rows = []
            for record_rows in by_record.values():
                trimmed_rows.extend(even_sample(record_rows, args.max_per_record_per_class))
            trimmed_rows.sort(key=sort_key)
            trimmed_rows = even_sample(trimmed_rows, label_limits[label_name])
            chosen_rows.extend(trimmed_rows)
            label_counts[label_name] = len(trimmed_rows)

        chosen_rows.sort(key=sort_key)
        chosen_rows = even_sample(chosen_rows, args.max_total_per_subject)
        for order, row in enumerate(chosen_rows, start=1):
            row = dict(row)
            row["subject_segment_rank"] = str(order)
            manifest_rows.append(row)

        subject_summary.append(
            {
                "subject_id": subject_id,
                "subject_folder_rel": subject["subject_folder_rel"],
                "selected_total": len(chosen_rows),
                "selected_af": sum(1 for row in chosen_rows if row["label_name"] == "AF"),
                "selected_sr": sum(1 for row in chosen_rows if row["label_name"] == "SR"),
                "available_af": len(candidates[subject_id]["AF"]),
                "available_sr": len(candidates[subject_id]["SR"]),
            }
        )

    if not manifest_rows:
        raise RuntimeError("No segment manifest rows were selected.")

    manifest_csv = args.output_dir / "selected_segments.csv"
    fieldnames = list(manifest_rows[0].keys())
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    subject_summary_csv = args.output_dir / "selected_segments_subject_summary.csv"
    with subject_summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subject_summary[0].keys()))
        writer.writeheader()
        writer.writerows(subject_summary)

    label_counter = Counter(row["label_name"] for row in manifest_rows)
    summary = {
        "scanned_rows": scanned_rows,
        "selected_subjects": len(selected_subjects),
        "selected_segments": len(manifest_rows),
        "selected_af_segments": label_counter.get("AF", 0),
        "selected_sr_segments": label_counter.get("SR", 0),
        "segment_selection_limits": {
            "max_total_per_subject": args.max_total_per_subject,
            "max_af_per_subject": args.max_af_per_subject,
            "max_sr_per_subject": args.max_sr_per_subject,
            "max_per_record_per_class": args.max_per_record_per_class,
        },
    }
    (args.output_dir / "selected_segments_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "Selected MIMIC-III-Ext-PPG segment manifest",
        f"Metadata: {args.metadata_csv}",
        f"Subjects CSV: {args.subjects_csv}",
        f"Rows scanned: {scanned_rows:,}",
        f"Selected segments: {len(manifest_rows):,}",
        f"Selected AF segments: {label_counter.get('AF', 0):,}",
        f"Selected SR segments: {label_counter.get('SR', 0):,}",
        "",
        "Per-subject selection summary",
    ]
    for row in subject_summary[:20]:
        report_lines.append(
            f"- {row['subject_folder_rel']}: total={row['selected_total']} "
            f"AF={row['selected_af']} SR={row['selected_sr']} "
            f"(available AF={row['available_af']}, SR={row['available_sr']})"
        )
    (args.output_dir / "selected_segments_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved segment manifest to: {manifest_csv}")


if __name__ == "__main__":
    main()
