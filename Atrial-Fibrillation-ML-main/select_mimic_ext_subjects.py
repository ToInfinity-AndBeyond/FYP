#!/usr/bin/env python3
"""Select a high-yield subject subset from MIMIC-III-Ext-PPG metadata."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_METADATA = Path("/vol/bitbucket/mc1920/FYP/metadata.csv")
DEFAULT_DATASET_ROOT = Path("/vol/bitbucket/mc1920/FYP/physionet.org/files/mimic-iii-ext-ppg/1.1.0")
DEFAULT_OUTPUT_DIR = Path("artifacts/mimic_ext_subject_selection")
DEFAULT_PREFIXES = ("p03", "p04", "p05", "p06", "p07", "p08", "p09")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank MIMIC-III-Ext-PPG subjects for partial download.")
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefixes", nargs="+", default=list(DEFAULT_PREFIXES))
    parser.add_argument("--top-n", type=int, default=56, help="Total number of selected subjects.")
    parser.add_argument(
        "--per-prefix-limit",
        type=int,
        default=8,
        help="Maximum selected subjects per pXX prefix.",
    )
    parser.add_argument("--min-total-segments", type=int, default=200)
    parser.add_argument("--min-af-segments", type=int, default=40)
    parser.add_argument("--min-sr-segments", type=int, default=40)
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


def subject_folder_name(subject_id: int) -> str:
    return f"p{subject_id:06d}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prefixes = set(args.prefixes)
    subject_stats: dict[int, dict[str, object]] = {}
    rhythm_counter = Counter()
    rows_scanned = 0

    with args.metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_scanned += 1
            if args.progress_every > 0 and rows_scanned % args.progress_every == 0:
                print(f"[progress] scanned {rows_scanned:,} metadata rows", flush=True)

            folder_path = str(row["folder_path"])
            prefix = folder_path.split("/", 1)[0]
            if prefix not in prefixes:
                continue

            rhythm = str(row["event_rhythm"])
            if rhythm not in {"AF", "SR"}:
                continue
            rhythm_counter[rhythm] += 1

            good_count, total_count = count_good_pleth_segments(row.get("vector_10s_pleth_sqi"))
            if total_count == 0:
                continue
            good_fraction = good_count / total_count
            if good_count < 2 or good_fraction < (2.0 / 3.0):
                continue

            subject_id = int(row["subject_id"])
            stats = subject_stats.setdefault(
                subject_id,
                {
                    "subject_id": subject_id,
                    "prefix": prefix,
                    "af_segments": 0,
                    "sr_segments": 0,
                    "accepted_segments": 0,
                    "quality_sum": 0.0,
                    "folds": set(),
                },
            )
            stats["accepted_segments"] += 1
            stats["quality_sum"] += good_fraction
            stats["folds"].add(str(row["strat_fold"]))
            if rhythm == "AF":
                stats["af_segments"] += 1
            else:
                stats["sr_segments"] += 1

    ranked_rows = []
    for stats in subject_stats.values():
        accepted_segments = int(stats["accepted_segments"])
        af_segments = int(stats["af_segments"])
        sr_segments = int(stats["sr_segments"])
        if accepted_segments < args.min_total_segments:
            continue
        if af_segments < args.min_af_segments:
            continue
        if sr_segments < args.min_sr_segments:
            continue

        af_fraction = af_segments / accepted_segments
        balance_score = 1.0 - abs(af_fraction - 0.5) * 2.0
        avg_quality = float(stats["quality_sum"]) / accepted_segments
        selection_score = accepted_segments * (0.65 + 0.35 * balance_score) * (0.75 + 0.25 * avg_quality)
        prefix = str(stats["prefix"])
        subject_folder = subject_folder_name(int(stats["subject_id"]))
        local_dir = args.dataset_root / prefix / subject_folder
        local_present = local_dir.exists() and any(local_dir.iterdir())
        ranked_rows.append(
            {
                "subject_id": int(stats["subject_id"]),
                "prefix": prefix,
                "subject_folder": subject_folder,
                "subject_folder_rel": f"{prefix}/{subject_folder}",
                "accepted_segments": accepted_segments,
                "af_segments": af_segments,
                "sr_segments": sr_segments,
                "af_fraction": af_fraction,
                "balance_score": balance_score,
                "avg_pleth_good_fraction": avg_quality,
                "selection_score": selection_score,
                "folds": ",".join(sorted(stats["folds"])),
                "local_present": int(local_present),
            }
        )

    ranked_rows.sort(
        key=lambda row: (row["selection_score"], row["accepted_segments"], row["balance_score"]),
        reverse=True,
    )

    selected_rows = []
    prefix_counter = Counter()
    for row in ranked_rows:
        if len(selected_rows) >= args.top_n:
            break
        if prefix_counter[row["prefix"]] >= args.per_prefix_limit:
            continue
        selected_rows.append(row)
        prefix_counter[row["prefix"]] += 1

    if not selected_rows:
        raise RuntimeError("No subjects satisfied the selection criteria.")

    ranked_csv = args.output_dir / "ranked_subjects.csv"
    selected_csv = args.output_dir / "selected_subjects.csv"
    fieldnames = list(ranked_rows[0].keys())
    with ranked_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked_rows)
    with selected_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    summary = {
        "rows_scanned": rows_scanned,
        "candidate_subjects": len(ranked_rows),
        "selected_subjects": len(selected_rows),
        "prefix_quota": dict(prefix_counter),
        "raw_rhythm_counts": dict(rhythm_counter),
        "selection_criteria": {
            "prefixes": list(args.prefixes),
            "top_n": args.top_n,
            "per_prefix_limit": args.per_prefix_limit,
            "min_total_segments": args.min_total_segments,
            "min_af_segments": args.min_af_segments,
            "min_sr_segments": args.min_sr_segments,
        },
    }
    (args.output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "Selected MIMIC-III-Ext-PPG subject cohort",
        f"Metadata: {args.metadata_csv}",
        f"Dataset root: {args.dataset_root}",
        f"Rows scanned: {rows_scanned:,}",
        f"Candidate subjects: {len(ranked_rows):,}",
        f"Selected subjects: {len(selected_rows):,}",
        "",
        "Per-prefix selection counts",
    ]
    for prefix, count in sorted(prefix_counter.items()):
        report_lines.append(f"- {prefix}: {count}")
    report_lines.extend(
        [
            "",
            "Top selected subjects",
        ]
    )
    for row in selected_rows[:20]:
        report_lines.append(
            f"- {row['subject_folder_rel']}: total={row['accepted_segments']} "
            f"AF={row['af_segments']} SR={row['sr_segments']} "
            f"avg_good_fraction={row['avg_pleth_good_fraction']:.3f} "
            f"score={row['selection_score']:.1f}"
        )
    (args.output_dir / "selection_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved ranked subject list to: {ranked_csv}")
    print(f"Saved selected subject list to: {selected_csv}")


if __name__ == "__main__":
    main()
