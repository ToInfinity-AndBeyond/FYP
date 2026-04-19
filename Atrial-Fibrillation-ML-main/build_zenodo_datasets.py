from __future__ import annotations

import argparse
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from build_physio_multimodal_dataset import (
    MAX_IBI_LENGTH,
    RespConfig,
    pad_sequence,
    prefix_metrics,
    save_multimodal_bundle,
    summarize_timing_consistency,
)
from signal_pipeline import (
    SignalPipelineConfig,
    apply_quality_overrides,
    default_ecg_config,
    default_ppg_config,
    load_quality_overrides,
    minimum_segment_samples,
    process_segment,
    save_dataset_bundle,
)
from zenodo_longterm_loader import (
    ZenodoECGRecord,
    ZenodoPPGSegment,
    build_subject_window_labels,
    load_zenodo_ecg_mat,
    load_zenodo_ppg_mat,
)


def discover_subject_pairs(dataset_root: Path) -> list[tuple[str, Path, Path]]:
    ecg_paths = {path.stem.replace("_ECG", ""): path for path in sorted(dataset_root.glob("*_ECG.mat"))}
    ppg_paths = {path.stem.replace("_PPG", ""): path for path in sorted(dataset_root.glob("*_PPG.mat"))}
    subject_ids = sorted(set(ecg_paths) & set(ppg_paths))
    return [(subject_id, ecg_paths[subject_id], ppg_paths[subject_id]) for subject_id in subject_ids]


def format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def should_report_progress(current_index: int, total_count: int, progress_every: int) -> bool:
    if current_index <= 1 or current_index >= total_count:
        return True
    if progress_every > 0 and current_index % progress_every == 0:
        return True
    if total_count > 0:
        completed_percent = int((current_index * 100) / total_count)
        previous_percent = int(((current_index - 1) * 100) / total_count)
        return completed_percent != previous_percent and completed_percent % 10 == 0
    return False


def resample_1d(signal_values: np.ndarray, source_rate_hz: float, target_rate_hz: float) -> np.ndarray:
    values = np.asarray(signal_values, dtype=float).reshape(-1)
    if values.size == 0 or source_rate_hz == target_rate_hz:
        return values.astype(np.float32, copy=False)

    ratio = Fraction(target_rate_hz / source_rate_hz).limit_denominator(1000)
    resampled = sp_signal.resample_poly(values, up=ratio.numerator, down=ratio.denominator)
    expected_size = int(round(values.size * target_rate_hz / source_rate_hz))
    if resampled.size > expected_size:
        resampled = resampled[:expected_size]
    elif resampled.size < expected_size:
        resampled = np.pad(resampled, (0, expected_size - resampled.size), mode="edge")
    return resampled.astype(np.float32)


def resample_acc(acc_xyz: np.ndarray, source_rate_hz: float, target_rate_hz: float) -> np.ndarray:
    if acc_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    channels = [
        resample_1d(acc_xyz[:, channel], source_rate_hz=source_rate_hz, target_rate_hz=target_rate_hz)
        for channel in range(acc_xyz.shape[1])
    ]
    return np.column_stack(channels).astype(np.float32)


def slice_ppg_window(
    segment: ZenodoPPGSegment,
    window_index: int,
    window_length_sec: float,
    stride_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    window_samples = int(round(window_length_sec * segment.ppg_sample_rate_hz))
    stride_samples = int(round(stride_sec * segment.ppg_sample_rate_hz))
    start_sample = window_index * stride_samples
    end_sample = start_sample + window_samples

    ppg_window = segment.ppg_green[start_sample:end_sample]
    acc_start = int(round(start_sample * segment.acc_sample_rate_hz / segment.ppg_sample_rate_hz))
    acc_end = int(round(end_sample * segment.acc_sample_rate_hz / segment.ppg_sample_rate_hz))
    acc_window = np.column_stack(
        [
            segment.acc_x[acc_start:acc_end],
            segment.acc_y[acc_start:acc_end],
            segment.acc_z[acc_start:acc_end],
        ]
    ).astype(np.float32)
    return ppg_window.astype(np.float32), acc_window


def slice_ecg_window(
    ecg_record: ZenodoECGRecord,
    start_time_sec: float,
    window_length_sec: float,
) -> np.ndarray:
    start_sample = int(round(start_time_sec * ecg_record.sample_rate_hz))
    end_sample = start_sample + int(round(window_length_sec * ecg_record.sample_rate_hz))
    return ecg_record.ecg[start_sample:end_sample].astype(np.float32)


def build_subject_ppg_rows(
    subject_id: str,
    ecg_record: ZenodoECGRecord,
    ppg_segments: list[ZenodoPPGSegment],
    ppg_config: SignalPipelineConfig,
    window_length_sec: float,
    stride_sec: float,
    positive_fraction_threshold: float,
    max_windows_per_subject: int | None = None,
    progress_every: int = 250,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    segment_lookup = {segment.segment_index: segment for segment in ppg_segments}
    print(f"[zenodo-ppg] subject {subject_id}: building ECG-aligned window labels", flush=True)
    window_label_df = build_subject_window_labels(
        ecg_record=ecg_record,
        ppg_segments=ppg_segments,
        window_length_sec=window_length_sec,
        stride_sec=stride_sec,
        positive_fraction_threshold=positive_fraction_threshold,
    )
    window_label_df = window_label_df.loc[window_label_df["use_for_training"] == 1].copy()
    if max_windows_per_subject is not None:
        window_label_df = window_label_df.head(max_windows_per_subject)

    rows: list[dict[str, Any]] = []
    signals: list[np.ndarray] = []
    segment_length_samples = int(round(window_length_sec * ppg_config.sample_rate_hz))
    skipped_short_windows = 0
    skipped_incomplete_windows = 0
    min_ppg_samples = minimum_segment_samples(ppg_config)
    total_windows = int(window_label_df.shape[0])
    start_time = time.time()
    print(
        f"[zenodo-ppg] subject {subject_id}: {total_windows} labeled windows to process",
        flush=True,
    )

    for window_counter, (_, label_row) in enumerate(window_label_df.iterrows(), start=1):
        segment = segment_lookup[int(label_row["ppg_segment_index"])]
        ppg_window, acc_window = slice_ppg_window(
            segment,
            window_index=int(label_row["segment_index"]),
            window_length_sec=window_length_sec,
            stride_sec=stride_sec,
        )
        ppg_resampled = resample_1d(ppg_window, segment.ppg_sample_rate_hz, ppg_config.sample_rate_hz)
        acc_resampled = resample_acc(acc_window, segment.acc_sample_rate_hz, ppg_config.sample_rate_hz)
        if ppg_resampled.size < min_ppg_samples:
            skipped_short_windows += 1
            continue
        if ppg_resampled.size != segment_length_samples:
            skipped_incomplete_windows += 1
            continue
        processed = process_segment(ppg_resampled, ppg_config, acc_segment=acc_resampled)

        row = {
            "record_id": f"zenodo_{subject_id}",
            "label": int(label_row["label"]),
            "label_source": str(label_row["label_source"]),
            "signal_name": ppg_config.signal_name,
            "segment_index": int(label_row["segment_index"]),
            "start_sample": int(round(float(label_row["start_time_sec"]) * ppg_config.sample_rate_hz)),
            "end_sample": int(round(float(label_row["end_time_sec"]) * ppg_config.sample_rate_hz)),
            "start_time_sec": float(label_row["start_time_sec"]),
            "end_time_sec": float(label_row["end_time_sec"]),
            "dataset_name": "zenodo_longterm_af",
            "subject_id": subject_id,
            "ppg_segment_index": int(label_row["ppg_segment_index"]),
            "af_fraction": float(label_row["af_fraction"]),
        }
        row.update(processed.quality_metrics)
        row.update(processed.feature_metrics)
        rows.append(row)
        signals.append(processed.normalized_signal.astype(np.float32))

        if should_report_progress(window_counter, total_windows, progress_every):
            elapsed = time.time() - start_time
            avg_seconds = elapsed / max(window_counter, 1)
            eta_seconds = avg_seconds * max(total_windows - window_counter, 0)
            accepted_count = sum(int(item.get("accepted", False)) for item in rows)
            percent = (window_counter / max(total_windows, 1)) * 100.0
            print(
                f"[zenodo-ppg] subject {subject_id}: "
                f"{window_counter}/{total_windows} ({percent:5.1f}%) "
                f"accepted={accepted_count} "
                f"skipped_short={skipped_short_windows} "
                f"skipped_incomplete={skipped_incomplete_windows} "
                f"elapsed={format_duration(elapsed)} "
                f"eta={format_duration(eta_seconds)}",
                flush=True,
            )

    return rows, signals


def build_subject_multimodal_rows(
    subject_id: str,
    ecg_record: ZenodoECGRecord,
    ppg_segments: list[ZenodoPPGSegment],
    ppg_config: SignalPipelineConfig,
    ecg_config: SignalPipelineConfig,
    window_length_sec: float,
    stride_sec: float,
    positive_fraction_threshold: float,
    max_windows_per_subject: int | None = None,
    progress_every: int = 250,
) -> tuple[list[dict[str, Any]], dict[str, list[np.ndarray] | list[int]]]:
    segment_lookup = {segment.segment_index: segment for segment in ppg_segments}
    print(f"[zenodo-physio] subject {subject_id}: building ECG-aligned window labels", flush=True)
    window_label_df = build_subject_window_labels(
        ecg_record=ecg_record,
        ppg_segments=ppg_segments,
        window_length_sec=window_length_sec,
        stride_sec=stride_sec,
        positive_fraction_threshold=positive_fraction_threshold,
    )
    window_label_df = window_label_df.loc[window_label_df["use_for_training"] == 1].copy()
    if max_windows_per_subject is not None:
        window_label_df = window_label_df.head(max_windows_per_subject)

    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray] | list[int]] = {
        "ppg_segments": [],
        "ecg_segments": [],
        "resp_segments": [],
        "ppg_ibi_sequences": [],
        "ecg_ibi_sequences": [],
        "ppg_ibi_lengths": [],
        "ecg_ibi_lengths": [],
        "labels": [],
        "ppg_quality_score": [],
        "joint_accepted": [],
        "start_time_sec": [],
    }

    segment_length_samples = int(round(window_length_sec * ppg_config.sample_rate_hz))
    min_ppg_samples = minimum_segment_samples(ppg_config)
    min_ecg_samples = minimum_segment_samples(ecg_config)
    skipped_short_windows = 0
    skipped_incomplete_windows = 0
    total_windows = int(window_label_df.shape[0])
    start_time = time.time()
    print(
        f"[zenodo-physio] subject {subject_id}: {total_windows} labeled windows to process",
        flush=True,
    )

    for window_counter, (_, label_row) in enumerate(window_label_df.iterrows(), start=1):
        segment = segment_lookup[int(label_row["ppg_segment_index"])]
        ppg_window, acc_window = slice_ppg_window(
            segment,
            window_index=int(label_row["segment_index"]),
            window_length_sec=window_length_sec,
            stride_sec=stride_sec,
        )
        ecg_window = slice_ecg_window(
            ecg_record=ecg_record,
            start_time_sec=float(label_row["start_time_sec"]),
            window_length_sec=window_length_sec,
        )

        ppg_resampled = resample_1d(ppg_window, segment.ppg_sample_rate_hz, ppg_config.sample_rate_hz)
        ecg_resampled = resample_1d(ecg_window, ecg_record.sample_rate_hz, ecg_config.sample_rate_hz)
        acc_resampled = resample_acc(acc_window, segment.acc_sample_rate_hz, ppg_config.sample_rate_hz)
        if ppg_resampled.size < min_ppg_samples or ecg_resampled.size < min_ecg_samples:
            skipped_short_windows += 1
            continue
        if ppg_resampled.size != segment_length_samples or ecg_resampled.size != segment_length_samples:
            skipped_incomplete_windows += 1
            continue

        ppg_processed = process_segment(ppg_resampled, ppg_config, acc_segment=acc_resampled)
        ecg_processed = process_segment(ecg_resampled, ecg_config)
        timing_metrics = summarize_timing_consistency(
            ecg_processed.peaks,
            ppg_processed.peaks,
            ppg_config.sample_rate_hz,
        )
        ppg_ibi_padded, ppg_ibi_length = pad_sequence(ppg_processed.ibi_seconds)
        ecg_ibi_padded, ecg_ibi_length = pad_sequence(ecg_processed.ibi_seconds)

        ppg_accepted = bool(ppg_processed.quality_metrics["accepted"])
        ecg_accepted = bool(ecg_processed.quality_metrics["accepted"])
        joint_accepted = ppg_accepted and ecg_accepted

        row = {
            "record_id": f"zenodo_{subject_id}",
            "label": int(label_row["label"]),
            "label_source": str(label_row["label_source"]),
            "segment_index": int(label_row["segment_index"]),
            "start_sample": int(round(float(label_row["start_time_sec"]) * ppg_config.sample_rate_hz)),
            "end_sample": int(round(float(label_row["end_time_sec"]) * ppg_config.sample_rate_hz)),
            "start_time_sec": float(label_row["start_time_sec"]),
            "end_time_sec": float(label_row["end_time_sec"]),
            "dataset_name": "zenodo_longterm_af",
            "subject_id": subject_id,
            "ppg_segment_index": int(label_row["ppg_segment_index"]),
            "af_fraction": float(label_row["af_fraction"]),
            "ppg_quality_score": float(ppg_processed.quality_metrics["quality_score"]),
            "resp_available": 0,
            "ppg_accepted": ppg_accepted,
            "ecg_accepted": ecg_accepted,
            "joint_accepted": joint_accepted,
            "joint_rejection_reason": ";".join(
                reason
                for reason in [
                    ppg_processed.quality_metrics["rejection_reason"],
                    ecg_processed.quality_metrics["rejection_reason"],
                ]
                if isinstance(reason, str) and reason
            ),
            "ppg_ibi_length": int(ppg_ibi_length),
            "ecg_ibi_length": int(ecg_ibi_length),
            "resp_rate_bpm": float("nan"),
            "resp_spectral_entropy": float("nan"),
            "resp_ppg_amplitude_corr": float("nan"),
            "resp_ibi_corr": float("nan"),
        }
        row.update(prefix_metrics(ppg_processed.quality_metrics, "ppg_"))
        row.update(prefix_metrics(ppg_processed.feature_metrics, "ppg_"))
        row.update(prefix_metrics(ecg_processed.quality_metrics, "ecg_"))
        row.update(prefix_metrics(ecg_processed.feature_metrics, "ecg_"))
        row.update(timing_metrics)
        rows.append(row)

        arrays["ppg_segments"].append(ppg_processed.normalized_signal.astype(np.float32))
        arrays["ecg_segments"].append(ecg_processed.normalized_signal.astype(np.float32))
        arrays["resp_segments"].append(np.zeros(segment_length_samples, dtype=np.float32))
        arrays["ppg_ibi_sequences"].append(ppg_ibi_padded)
        arrays["ecg_ibi_sequences"].append(ecg_ibi_padded)
        arrays["ppg_ibi_lengths"].append(int(ppg_ibi_length))
        arrays["ecg_ibi_lengths"].append(int(ecg_ibi_length))
        arrays["labels"].append(int(label_row["label"]))
        arrays["ppg_quality_score"].append(float(ppg_processed.quality_metrics["quality_score"]))
        arrays["joint_accepted"].append(bool(joint_accepted))
        arrays["start_time_sec"].append(float(label_row["start_time_sec"]))

        if should_report_progress(window_counter, total_windows, progress_every):
            elapsed = time.time() - start_time
            avg_seconds = elapsed / max(window_counter, 1)
            eta_seconds = avg_seconds * max(total_windows - window_counter, 0)
            joint_accepted_count = sum(int(item.get("joint_accepted", False)) for item in rows)
            percent = (window_counter / max(total_windows, 1)) * 100.0
            print(
                f"[zenodo-physio] subject {subject_id}: "
                f"{window_counter}/{total_windows} ({percent:5.1f}%) "
                f"joint_accepted={joint_accepted_count} "
                f"skipped_short={skipped_short_windows} "
                f"skipped_incomplete={skipped_incomplete_windows} "
                f"elapsed={format_duration(elapsed)} "
                f"eta={format_duration(eta_seconds)}",
                flush=True,
            )

    return rows, arrays


def finalize_multimodal_arrays(arrays: dict[str, list[np.ndarray] | list[int]]) -> dict[str, np.ndarray]:
    if not arrays["ppg_segments"]:
        return {
            "ppg_segments": np.empty((0, 0), dtype=np.float32),
            "ecg_segments": np.empty((0, 0), dtype=np.float32),
            "resp_segments": np.empty((0, 0), dtype=np.float32),
            "ppg_ibi_sequences": np.empty((0, MAX_IBI_LENGTH), dtype=np.float32),
            "ecg_ibi_sequences": np.empty((0, MAX_IBI_LENGTH), dtype=np.float32),
            "ppg_ibi_lengths": np.empty((0,), dtype=np.int16),
            "ecg_ibi_lengths": np.empty((0,), dtype=np.int16),
            "labels": np.empty((0,), dtype=np.int8),
            "ppg_quality_score": np.empty((0,), dtype=np.float32),
            "joint_accepted": np.empty((0,), dtype=bool),
            "start_time_sec": np.empty((0,), dtype=np.float32),
        }
    return {
        "ppg_segments": np.stack(arrays["ppg_segments"]).astype(np.float32),
        "ecg_segments": np.stack(arrays["ecg_segments"]).astype(np.float32),
        "resp_segments": np.stack(arrays["resp_segments"]).astype(np.float32),
        "ppg_ibi_sequences": np.stack(arrays["ppg_ibi_sequences"]).astype(np.float32),
        "ecg_ibi_sequences": np.stack(arrays["ecg_ibi_sequences"]).astype(np.float32),
        "ppg_ibi_lengths": np.asarray(arrays["ppg_ibi_lengths"], dtype=np.int16),
        "ecg_ibi_lengths": np.asarray(arrays["ecg_ibi_lengths"], dtype=np.int16),
        "labels": np.asarray(arrays["labels"], dtype=np.int8),
        "ppg_quality_score": np.asarray(arrays["ppg_quality_score"], dtype=np.float32),
        "joint_accepted": np.asarray(arrays["joint_accepted"], dtype=bool),
        "start_time_sec": np.asarray(arrays["start_time_sec"], dtype=np.float32),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Zenodo long-term AF datasets in the same format used by the current MIMIC pipelines."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent / "zenodo_longterm_af",
        help="Directory that contains subject MAT files such as 001_ECG.mat and 001_PPG.mat",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "zenodo",
        help="Root output directory for Zenodo artifacts",
    )
    parser.add_argument(
        "--mode",
        choices=["ppg", "physio", "both"],
        default="both",
        help="Which dataset flavor to build",
    )
    parser.add_argument(
        "--subject-ids",
        nargs="+",
        default=None,
        help="Optional subject IDs such as 001 002 003",
    )
    parser.add_argument(
        "--limit-subjects",
        type=int,
        default=None,
        help="Optional subject limit for smoke tests",
    )
    parser.add_argument("--window-length-sec", type=float, default=30.0)
    parser.add_argument("--stride-sec", type=float, default=30.0)
    parser.add_argument("--positive-fraction-threshold", type=float, default=0.5)
    parser.add_argument(
        "--target-sample-rate-hz",
        type=float,
        default=125.0,
        help="Resample PPG/ECG windows to this rate so they remain compatible with the existing MIMIC models.",
    )
    parser.add_argument(
        "--max-windows-per-subject",
        type=int,
        default=None,
        help="Optional cap for smoke tests or quick iterations.",
    )
    parser.add_argument(
        "--ppg-quality-json",
        type=Path,
        default=None,
        help="Optional JSON file containing tuned PPG quality gate overrides.",
    )
    parser.add_argument(
        "--ecg-quality-json",
        type=Path,
        default=None,
        help="Optional JSON file containing tuned ECG quality gate overrides.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print per-subject progress every N processed windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subject_pairs = discover_subject_pairs(args.dataset_root)
    if args.subject_ids is not None:
        wanted = set(args.subject_ids)
        subject_pairs = [pair for pair in subject_pairs if pair[0] in wanted]
    if args.limit_subjects is not None:
        subject_pairs = subject_pairs[: args.limit_subjects]
    if not subject_pairs:
        raise FileNotFoundError(f"No matching ECG/PPG MAT pairs found under {args.dataset_root}")

    ppg_config = apply_quality_overrides(
        default_ppg_config(sample_rate_hz=args.target_sample_rate_hz),
        load_quality_overrides(args.ppg_quality_json),
    )
    ecg_config = apply_quality_overrides(
        default_ecg_config(sample_rate_hz=args.target_sample_rate_hz),
        load_quality_overrides(args.ecg_quality_json),
    )
    resp_config = RespConfig(sample_rate_hz=args.target_sample_rate_hz)

    if args.mode in {"ppg", "both"}:
        ppg_rows: list[dict[str, Any]] = []
        ppg_signals: list[np.ndarray] = []
        for index, (subject_id, ecg_path, ppg_path) in enumerate(subject_pairs, start=1):
            print(
                f"[zenodo-ppg] processing subject {index}/{len(subject_pairs)}: {subject_id}",
                flush=True,
            )
            print(f"[zenodo-ppg] subject {subject_id}: loading ECG MAT", flush=True)
            ecg_record = load_zenodo_ecg_mat(ecg_path)
            print(f"[zenodo-ppg] subject {subject_id}: loading PPG MAT", flush=True)
            ppg_segments, _ = load_zenodo_ppg_mat(ppg_path, ecg_record=ecg_record)
            subject_rows, subject_signals = build_subject_ppg_rows(
                subject_id=subject_id,
                ecg_record=ecg_record,
                ppg_segments=ppg_segments,
                ppg_config=ppg_config,
                window_length_sec=args.window_length_sec,
                stride_sec=args.stride_sec,
                positive_fraction_threshold=args.positive_fraction_threshold,
                max_windows_per_subject=args.max_windows_per_subject,
                progress_every=args.progress_every,
            )
            ppg_rows.extend(subject_rows)
            ppg_signals.extend(subject_signals)
            print(
                f"[zenodo-ppg] subject {subject_id}: windows={len(subject_rows)} "
                f"accepted={sum(int(row.get('accepted', False)) for row in subject_rows)}",
                flush=True,
            )

        ppg_summary = pd.DataFrame(ppg_rows)
        ppg_segments_array = np.stack(ppg_signals).astype(np.float32) if ppg_signals else np.empty((0, 0), dtype=np.float32)
        saved = save_dataset_bundle(ppg_summary, ppg_segments_array, args.output_root / "signal_pipeline" / "ppg", ppg_config)
        accepted_count = int(ppg_summary["accepted"].sum()) if not ppg_summary.empty else 0
        print(f"[zenodo-ppg] subjects processed: {len(subject_pairs)}")
        print(f"[zenodo-ppg] segments: {int(ppg_summary.shape[0])}")
        print(f"[zenodo-ppg] accepted: {accepted_count}")
        print(f"[zenodo-ppg] AF segments: {int(ppg_summary['label'].sum()) if not ppg_summary.empty else 0}")
        for name, path in saved.items():
            print(f"  - {name}: {path}")

    if args.mode in {"physio", "both"}:
        physio_rows: list[dict[str, Any]] = []
        physio_arrays_accumulator: dict[str, list[np.ndarray] | list[int]] = {
            "ppg_segments": [],
            "ecg_segments": [],
            "resp_segments": [],
            "ppg_ibi_sequences": [],
            "ecg_ibi_sequences": [],
            "ppg_ibi_lengths": [],
            "ecg_ibi_lengths": [],
            "labels": [],
            "ppg_quality_score": [],
            "joint_accepted": [],
            "start_time_sec": [],
        }
        for index, (subject_id, ecg_path, ppg_path) in enumerate(subject_pairs, start=1):
            print(
                f"[zenodo-physio] processing subject {index}/{len(subject_pairs)}: {subject_id}",
                flush=True,
            )
            print(f"[zenodo-physio] subject {subject_id}: loading ECG MAT", flush=True)
            ecg_record = load_zenodo_ecg_mat(ecg_path)
            print(f"[zenodo-physio] subject {subject_id}: loading PPG MAT", flush=True)
            ppg_segments, _ = load_zenodo_ppg_mat(ppg_path, ecg_record=ecg_record)
            subject_rows, subject_arrays = build_subject_multimodal_rows(
                subject_id=subject_id,
                ecg_record=ecg_record,
                ppg_segments=ppg_segments,
                ppg_config=ppg_config,
                ecg_config=ecg_config,
                window_length_sec=args.window_length_sec,
                stride_sec=args.stride_sec,
                positive_fraction_threshold=args.positive_fraction_threshold,
                max_windows_per_subject=args.max_windows_per_subject,
                progress_every=args.progress_every,
            )
            physio_rows.extend(subject_rows)
            for key, values in subject_arrays.items():
                physio_arrays_accumulator[key].extend(values)
            print(
                f"[zenodo-physio] subject {subject_id}: windows={len(subject_rows)} "
                f"joint_accepted={sum(int(row.get('joint_accepted', False)) for row in subject_rows)}",
                flush=True,
            )

        physio_summary = pd.DataFrame(physio_rows)
        physio_arrays = finalize_multimodal_arrays(physio_arrays_accumulator)
        saved = save_multimodal_bundle(
            summary_df=physio_summary,
            arrays=physio_arrays,
            output_dir=args.output_root / "physio_distill",
            ppg_config=ppg_config,
            ecg_config=ecg_config,
            resp_config=resp_config,
        )
        joint_accepted = int(physio_summary["joint_accepted"].sum()) if not physio_summary.empty else 0
        print(f"[zenodo-physio] subjects processed: {len(subject_pairs)}")
        print(f"[zenodo-physio] segments: {int(physio_summary.shape[0])}")
        print(f"[zenodo-physio] joint accepted: {joint_accepted}")
        print(f"[zenodo-physio] AF segments: {int(physio_summary['label'].sum()) if not physio_summary.empty else 0}")
        for name, path in saved.items():
            print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
