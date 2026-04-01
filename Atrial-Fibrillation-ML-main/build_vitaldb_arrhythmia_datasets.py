from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    process_segment,
    save_dataset_bundle,
)
from vitaldb_arrhythmia_loader import (
    VitalDBWaveformRecord,
    build_case_window_labels,
    load_annotation_metadata,
    load_case_annotations,
    load_case_waveforms,
)


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


def slice_waveform_window(
    signal_values: np.ndarray,
    start_time_sec: float,
    window_length_sec: float,
    sample_rate_hz: float,
) -> np.ndarray | None:
    start_sample = int(round(start_time_sec * sample_rate_hz))
    window_samples = int(round(window_length_sec * sample_rate_hz))
    end_sample = start_sample + window_samples
    if start_sample < 0 or end_sample > signal_values.size:
        return None
    return signal_values[start_sample:end_sample].astype(np.float32)


def _safe_process_segment(
    signal_values: np.ndarray,
    config: SignalPipelineConfig,
) -> tuple[Any | None, float]:
    nan_fraction = float(np.mean(np.isnan(signal_values))) if signal_values.size else 1.0
    if signal_values.size == 0 or np.all(np.isnan(signal_values)) or nan_fraction >= 0.50:
        return None, nan_fraction
    try:
        return process_segment(signal_values, config), nan_fraction
    except ValueError:
        return None, nan_fraction


def build_case_ppg_rows(
    waveform_record: VitalDBWaveformRecord,
    window_label_df: pd.DataFrame,
    ppg_config: SignalPipelineConfig,
    progress_every: int = 100,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    rows: list[dict[str, Any]] = []
    signals: list[np.ndarray] = []
    total_windows = int(window_label_df.shape[0])
    start_clock = time.time()

    for window_counter, (_, label_row) in enumerate(window_label_df.iterrows(), start=1):
        ppg_window = slice_waveform_window(
            waveform_record.ppg,
            start_time_sec=float(label_row["start_time_sec"]),
            window_length_sec=ppg_config.segment.length_seconds,
            sample_rate_hz=ppg_config.sample_rate_hz,
        )
        if ppg_window is None:
            continue

        ppg_processed, ppg_nan_fraction = _safe_process_segment(ppg_window, ppg_config)
        if ppg_processed is None:
            continue

        start_sample = int(round(float(label_row["start_time_sec"]) * ppg_config.sample_rate_hz))
        end_sample = start_sample + ppg_window.size
        row = {
            "record_id": str(label_row["record_id"]),
            "label": int(label_row["label"]),
            "label_source": str(label_row["label_source"]),
            "signal_name": ppg_config.signal_name,
            "segment_index": int(label_row["segment_index"]),
            "start_sample": int(start_sample),
            "end_sample": int(end_sample),
            "start_time_sec": float(label_row["start_time_sec"]),
            "end_time_sec": float(label_row["end_time_sec"]),
            "dataset_name": "vitaldb_arrhythmia",
            "case_id": int(waveform_record.case_id),
            "analysis_start_time_sec": float(waveform_record.analysis_start_time_sec),
            "analysis_end_time_sec": float(waveform_record.analysis_end_time_sec),
            "annotation_coverage_fraction": float(label_row["annotation_coverage_fraction"]),
            "af_fraction": float(label_row["af_fraction"]),
            "normal_fraction": float(label_row["normal_fraction"]),
            "other_rhythm_fraction": float(label_row["other_rhythm_fraction"]),
            "bad_signal_quality_fraction": float(label_row["bad_signal_quality_fraction"]),
            "ppg_track_name": waveform_record.ppg_track_name,
            "raw_nan_fraction": float(ppg_nan_fraction),
        }
        row.update(ppg_processed.quality_metrics)
        row.update(ppg_processed.feature_metrics)
        rows.append(row)
        signals.append(ppg_processed.normalized_signal.astype(np.float32))

        if should_report_progress(window_counter, total_windows, progress_every):
            elapsed = time.time() - start_clock
            avg_seconds = elapsed / max(window_counter, 1)
            eta_seconds = avg_seconds * max(total_windows - window_counter, 0)
            accepted_count = sum(int(item.get("accepted", False)) for item in rows)
            percent = (window_counter / max(total_windows, 1)) * 100.0
            print(
                f"[vitaldb-ppg] case {waveform_record.case_id}: "
                f"{window_counter}/{total_windows} ({percent:5.1f}%) "
                f"accepted={accepted_count} "
                f"elapsed={format_duration(elapsed)} "
                f"eta={format_duration(eta_seconds)}",
                flush=True,
            )

    return rows, signals


def build_case_multimodal_rows(
    waveform_record: VitalDBWaveformRecord,
    window_label_df: pd.DataFrame,
    ppg_config: SignalPipelineConfig,
    ecg_config: SignalPipelineConfig,
    progress_every: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, list[np.ndarray] | list[int]]]:
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
    total_windows = int(window_label_df.shape[0])
    start_clock = time.time()
    resp_template = np.zeros(int(round(ppg_config.segment.length_seconds * ppg_config.sample_rate_hz)), dtype=np.float32)

    for window_counter, (_, label_row) in enumerate(window_label_df.iterrows(), start=1):
        ppg_window = slice_waveform_window(
            waveform_record.ppg,
            start_time_sec=float(label_row["start_time_sec"]),
            window_length_sec=ppg_config.segment.length_seconds,
            sample_rate_hz=ppg_config.sample_rate_hz,
        )
        ecg_window = slice_waveform_window(
            waveform_record.ecg,
            start_time_sec=float(label_row["start_time_sec"]),
            window_length_sec=ecg_config.segment.length_seconds,
            sample_rate_hz=ecg_config.sample_rate_hz,
        )
        if ppg_window is None or ecg_window is None:
            continue

        ppg_processed, ppg_nan_fraction = _safe_process_segment(ppg_window, ppg_config)
        ecg_processed, ecg_nan_fraction = _safe_process_segment(ecg_window, ecg_config)
        if ppg_processed is None or ecg_processed is None:
            continue

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

        start_sample = int(round(float(label_row["start_time_sec"]) * ppg_config.sample_rate_hz))
        end_sample = start_sample + ppg_window.size
        row: dict[str, float | int | str | bool] = {
            "record_id": str(label_row["record_id"]),
            "label": int(label_row["label"]),
            "label_source": str(label_row["label_source"]),
            "segment_index": int(label_row["segment_index"]),
            "start_sample": int(start_sample),
            "end_sample": int(end_sample),
            "start_time_sec": float(label_row["start_time_sec"]),
            "end_time_sec": float(label_row["end_time_sec"]),
            "dataset_name": "vitaldb_arrhythmia",
            "case_id": int(waveform_record.case_id),
            "analysis_start_time_sec": float(waveform_record.analysis_start_time_sec),
            "analysis_end_time_sec": float(waveform_record.analysis_end_time_sec),
            "annotation_coverage_fraction": float(label_row["annotation_coverage_fraction"]),
            "af_fraction": float(label_row["af_fraction"]),
            "normal_fraction": float(label_row["normal_fraction"]),
            "other_rhythm_fraction": float(label_row["other_rhythm_fraction"]),
            "bad_signal_quality_fraction": float(label_row["bad_signal_quality_fraction"]),
            "ppg_track_name": waveform_record.ppg_track_name,
            "ecg_track_name": waveform_record.ecg_track_name,
            "resp_available": 0,
            "ppg_quality_score": float(ppg_processed.quality_metrics["quality_score"]),
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
            "ppg_raw_nan_fraction": float(ppg_nan_fraction),
            "ecg_raw_nan_fraction": float(ecg_nan_fraction),
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
        arrays["resp_segments"].append(resp_template.copy())
        arrays["ppg_ibi_sequences"].append(ppg_ibi_padded.astype(np.float32))
        arrays["ecg_ibi_sequences"].append(ecg_ibi_padded.astype(np.float32))
        arrays["ppg_ibi_lengths"].append(int(ppg_ibi_length))
        arrays["ecg_ibi_lengths"].append(int(ecg_ibi_length))
        arrays["labels"].append(int(label_row["label"]))
        arrays["ppg_quality_score"].append(float(ppg_processed.quality_metrics["quality_score"]))
        arrays["joint_accepted"].append(bool(joint_accepted))
        arrays["start_time_sec"].append(float(label_row["start_time_sec"]))

        if should_report_progress(window_counter, total_windows, progress_every):
            elapsed = time.time() - start_clock
            avg_seconds = elapsed / max(window_counter, 1)
            eta_seconds = avg_seconds * max(total_windows - window_counter, 0)
            joint_count = sum(int(item.get("joint_accepted", False)) for item in rows)
            percent = (window_counter / max(total_windows, 1)) * 100.0
            print(
                f"[vitaldb-physio] case {waveform_record.case_id}: "
                f"{window_counter}/{total_windows} ({percent:5.1f}%) "
                f"joint_accepted={joint_count} "
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
        description="Build VitalDB arrhythmia datasets that match the project's existing PPG and multimodal formats."
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path(__file__).resolve().parent / "vitaldb-arrhythmia",
        help="Directory that contains metadata.csv and Annotation_Files from the VitalDB arrhythmia dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "vitaldb_arrhythmia",
        help="Root output directory for VitalDB-derived artifacts.",
    )
    parser.add_argument(
        "--mode",
        choices=["ppg", "physio", "both"],
        default="both",
        help="Which dataset flavor to build.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of specific VitalDB case IDs to process.",
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=None,
        help="Optional case limit for smoke tests.",
    )
    parser.add_argument("--window-length-sec", type=float, default=30.0)
    parser.add_argument("--stride-sec", type=float, default=30.0)
    parser.add_argument("--positive-fraction-threshold", type=float, default=0.5)
    parser.add_argument(
        "--min-annotation-coverage-fraction",
        type=float,
        default=0.8,
        help="Minimum fraction of a 30-second window that must be covered by beat annotations.",
    )
    parser.add_argument(
        "--negative-label-policy",
        choices=["normal_only", "all_non_af"],
        default="normal_only",
        help="How to define negative windows.",
    )
    parser.add_argument(
        "--max-bad-quality-fraction",
        type=float,
        default=1.0,
        help="Exclude windows whose bad ECG signal-quality fraction exceeds this threshold.",
    )
    parser.add_argument(
        "--target-sample-rate-hz",
        type=float,
        default=125.0,
        help="Resample VitalDB waveforms to this rate so they remain compatible with existing models.",
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
        default=100,
        help="Print per-case progress every N processed windows.",
    )
    parser.add_argument(
        "--no-download-annotations",
        action="store_true",
        help="Do not auto-download metadata.csv or Annotation_Files when they are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_df = load_annotation_metadata(
        args.annotation_root,
        download_if_missing=not args.no_download_annotations,
    )
    if args.case_ids is not None:
        wanted = set(int(case_id) for case_id in args.case_ids)
        metadata_df = metadata_df.loc[metadata_df["case_id"].isin(wanted)].copy()
    if args.limit_cases is not None:
        metadata_df = metadata_df.head(args.limit_cases).copy()
    if metadata_df.empty:
        raise FileNotFoundError("No VitalDB arrhythmia cases selected for processing.")

    ppg_config = apply_quality_overrides(
        default_ppg_config(sample_rate_hz=args.target_sample_rate_hz),
        load_quality_overrides(args.ppg_quality_json),
    )
    ecg_config = apply_quality_overrides(
        default_ecg_config(sample_rate_hz=args.target_sample_rate_hz),
        load_quality_overrides(args.ecg_quality_json),
    )
    resp_config = RespConfig(sample_rate_hz=args.target_sample_rate_hz)

    ppg_rows: list[dict[str, Any]] = []
    ppg_signals: list[np.ndarray] = []
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

    processed_case_count = 0
    for case_index, metadata_row in enumerate(metadata_df.to_dict(orient="records"), start=1):
        case_id = int(metadata_row["case_id"])
        print(f"[vitaldb] processing case {case_index}/{len(metadata_df)}: {case_id}", flush=True)
        try:
            annotation_df = load_case_annotations(
                case_id,
                args.annotation_root,
                download_if_missing=not args.no_download_annotations,
            )
            window_label_df = build_case_window_labels(
                case_id=case_id,
                metadata_row=pd.Series(metadata_row),
                annotation_df=annotation_df,
                window_length_sec=args.window_length_sec,
                stride_sec=args.stride_sec,
                positive_fraction_threshold=args.positive_fraction_threshold,
                min_annotation_coverage_fraction=args.min_annotation_coverage_fraction,
                negative_label_policy=args.negative_label_policy,
                max_bad_quality_fraction=args.max_bad_quality_fraction,
            )
            window_label_df = window_label_df.loc[window_label_df["use_for_training"] == 1].copy()
            if window_label_df.empty:
                print(f"[vitaldb] case {case_id}: no eligible labeled windows after filtering", flush=True)
                continue

            waveform_record = load_case_waveforms(
                case_id=case_id,
                analysis_start_time_sec=float(metadata_row["analysis_start_time_sec"]),
                analysis_end_time_sec=float(metadata_row["analysis_end_time_sec"]),
                target_sample_rate_hz=args.target_sample_rate_hz,
            )
            if waveform_record is None:
                print(f"[vitaldb] case {case_id}: missing usable PLETH/ECG waveforms", flush=True)
                continue

            processed_case_count += 1
            if args.mode in {"ppg", "both"}:
                case_ppg_rows, case_ppg_signals = build_case_ppg_rows(
                    waveform_record=waveform_record,
                    window_label_df=window_label_df,
                    ppg_config=ppg_config,
                    progress_every=args.progress_every,
                )
                ppg_rows.extend(case_ppg_rows)
                ppg_signals.extend(case_ppg_signals)
                print(
                    f"[vitaldb-ppg] case {case_id}: windows={len(case_ppg_rows)} "
                    f"accepted={sum(int(row.get('accepted', False)) for row in case_ppg_rows)}",
                    flush=True,
                )

            if args.mode in {"physio", "both"}:
                case_physio_rows, case_physio_arrays = build_case_multimodal_rows(
                    waveform_record=waveform_record,
                    window_label_df=window_label_df,
                    ppg_config=ppg_config,
                    ecg_config=ecg_config,
                    progress_every=args.progress_every,
                )
                physio_rows.extend(case_physio_rows)
                for key, values in case_physio_arrays.items():
                    physio_arrays_accumulator[key].extend(values)
                print(
                    f"[vitaldb-physio] case {case_id}: windows={len(case_physio_rows)} "
                    f"joint_accepted={sum(int(row.get('joint_accepted', False)) for row in case_physio_rows)}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[vitaldb] case {case_id}: skipped due to {type(exc).__name__}: {exc}", flush=True)

    print(f"[vitaldb] cases with usable outputs: {processed_case_count}")

    if args.mode in {"ppg", "both"}:
        ppg_summary = pd.DataFrame(ppg_rows)
        ppg_segments_array = np.stack(ppg_signals).astype(np.float32) if ppg_signals else np.empty((0, 0), dtype=np.float32)
        saved = save_dataset_bundle(
            ppg_summary,
            ppg_segments_array,
            args.output_root / "signal_pipeline" / "ppg",
            ppg_config,
        )
        accepted_count = int(ppg_summary["accepted"].sum()) if not ppg_summary.empty else 0
        print(f"[vitaldb-ppg] segments: {int(ppg_summary.shape[0])}")
        print(f"[vitaldb-ppg] accepted: {accepted_count}")
        print(f"[vitaldb-ppg] AF segments: {int(ppg_summary['label'].sum()) if not ppg_summary.empty else 0}")
        for name, path in saved.items():
            print(f"  - {name}: {path}")

    if args.mode in {"physio", "both"}:
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
        print(f"[vitaldb-physio] segments: {int(physio_summary.shape[0])}")
        print(f"[vitaldb-physio] joint accepted: {joint_accepted}")
        print(f"[vitaldb-physio] AF segments: {int(physio_summary['label'].sum()) if not physio_summary.empty else 0}")
        for name, path in saved.items():
            print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
