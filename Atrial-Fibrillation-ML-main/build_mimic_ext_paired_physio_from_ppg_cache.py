from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import wfdb
except ImportError:  # pragma: no cover
    wfdb = None

from build_physio_multimodal_dataset import (
    MAX_IBI_LENGTH,
    RespConfig,
    pad_sequence,
    prefix_metrics,
    preprocess_respiration,
    summarize_respiration,
    summarize_timing_consistency,
)
from signal_pipeline import (
    SignalPipelineConfig,
    apply_quality_overrides,
    default_ecg_config,
    default_ppg_config,
    load_quality_overrides,
    process_segment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired PPG/ECG multimodal bundles from an existing MIMIC-III-Ext-PPG "
            "PPG accepted cache. Rows and labels follow the PPG accepted summary."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/vol/bitbucket/mc1920/FYP/1.1.0"))
    parser.add_argument("--ppg-summary", type=Path, required=True)
    parser.add_argument("--ppg-segments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ppg-quality-json", type=Path, default=None)
    parser.add_argument("--ecg-quality-json", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--require-ecg-accepted",
        action="store_true",
        help="If set, accepted bundle keeps only rows passing both original PPG and ECG gates.",
    )
    return parser.parse_args()


def resolve_record_base(dataset_root: Path, row: pd.Series) -> Path:
    if "wfdb_record_path" in row and pd.notna(row["wfdb_record_path"]):
        candidate = dataset_root / str(row["wfdb_record_path"])
        if candidate.with_suffix(".hea").exists() or candidate.with_suffix(".dat").exists():
            return candidate
        # Some older PPG caches stored pXX/patient/record/record even though
        # the WFDB files live directly under pXX/patient/record.{hea,dat}.
        if candidate.parent.name == candidate.name:
            return candidate.parent
        return candidate
    folder = Path(str(row["folder_path"]))
    signal_name = str(row["signal_file_name"])
    if folder.name == signal_name:
        return dataset_root / folder
    return dataset_root / folder / signal_name


def read_wfdb_channels(record_base: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    if wfdb is None:
        raise ImportError("wfdb is required. Use the project venv where wfdb is installed.")
    record = wfdb.rdrecord(str(record_base))
    if record.p_signal is None:
        raise ValueError(f"No physical signal array in {record_base}")
    names = [str(name).upper() for name in record.sig_name]
    ppg_idx = next((idx for idx, name in enumerate(names) if name == "PLETH"), None)
    ecg_idx = next((idx for idx, name in enumerate(names) if name == "II" or "ECG" in name), None)
    resp_idx = next((idx for idx, name in enumerate(names) if name in {"RESP", "IMP"}), None)
    if ppg_idx is None:
        raise KeyError(f"No PLETH channel in {record_base}")
    if ecg_idx is None:
        raise KeyError(f"No ECG/II channel in {record_base}")
    ppg = record.p_signal[:, ppg_idx].astype(float)
    ecg = record.p_signal[:, ecg_idx].astype(float)
    resp = record.p_signal[:, resp_idx].astype(float) if resp_idx is not None else None
    return ppg, ecg, resp, float(record.fs)


def empty_bundle(segment_samples: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    arrays = {
        "ppg_segments": np.empty((0, segment_samples), dtype=np.float32),
        "ecg_segments": np.empty((0, segment_samples), dtype=np.float32),
        "resp_segments": np.empty((0, segment_samples), dtype=np.float32),
        "ppg_ibi_sequences": np.empty((0, MAX_IBI_LENGTH), dtype=np.float32),
        "ecg_ibi_sequences": np.empty((0, MAX_IBI_LENGTH), dtype=np.float32),
        "ppg_ibi_lengths": np.empty((0,), dtype=np.int16),
        "ecg_ibi_lengths": np.empty((0,), dtype=np.int16),
        "labels": np.empty((0,), dtype=np.int8),
        "ppg_quality_score": np.empty((0,), dtype=np.float32),
        "joint_accepted": np.empty((0,), dtype=bool),
        "start_time_sec": np.empty((0,), dtype=np.float32),
    }
    return pd.DataFrame(), arrays


def save_bundle(
    summary_df: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    output_dir: Path,
    ppg_config: SignalPipelineConfig,
    ecg_config: SignalPipelineConfig,
    resp_config: RespConfig,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "physio_multimodal_segment_summary.csv"
    accepted_summary_path = output_dir / "physio_multimodal_accepted_segment_summary.csv"
    segments_path = output_dir / "physio_multimodal_segments.npz"
    accepted_segments_path = output_dir / "physio_multimodal_accepted_segments.npz"
    config_path = output_dir / "physio_multimodal_config.json"

    summary_df.to_csv(summary_path, index=False)
    accepted_mask = summary_df["joint_accepted"].to_numpy(dtype=bool) if not summary_df.empty else np.empty((0,), dtype=bool)
    summary_df.loc[accepted_mask].to_csv(accepted_summary_path, index=False)
    np.savez_compressed(segments_path, **arrays)

    accepted_arrays = {
        key: values[accepted_mask] if values.shape[:1] == accepted_mask.shape else values
        for key, values in arrays.items()
    }
    np.savez_compressed(accepted_segments_path, **accepted_arrays)
    config_path.write_text(
        json.dumps(
            {
                "ppg": asdict(ppg_config),
                "ecg": asdict(ecg_config),
                "resp": asdict(resp_config),
                "max_ibi_length": MAX_IBI_LENGTH,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "summary_csv": summary_path,
        "accepted_summary_csv": accepted_summary_path,
        "segments_npz": segments_path,
        "accepted_segments_npz": accepted_segments_path,
        "config_json": config_path,
    }


def main() -> None:
    args = parse_args()
    ppg_config = apply_quality_overrides(default_ppg_config(), load_quality_overrides(args.ppg_quality_json))
    ecg_config = apply_quality_overrides(default_ecg_config(), load_quality_overrides(args.ecg_quality_json))
    resp_config = RespConfig(sample_rate_hz=ppg_config.sample_rate_hz)

    ppg_summary = pd.read_csv(args.ppg_summary, low_memory=False)
    ppg_npz = np.load(args.ppg_segments)
    ppg_segments_cached = ppg_npz["segments"].astype(np.float32)
    if ppg_summary.shape[0] != ppg_segments_cached.shape[0]:
        raise ValueError("PPG accepted summary and accepted segments have different row counts.")
    if args.max_rows is not None:
        ppg_summary = ppg_summary.head(args.max_rows).copy()
        ppg_segments_cached = ppg_segments_cached[: args.max_rows]

    segment_samples = int(round(ppg_config.segment.length_seconds * ppg_config.sample_rate_hz))
    rows: list[dict[str, Any]] = []
    ppg_segments: list[np.ndarray] = []
    ecg_segments: list[np.ndarray] = []
    resp_segments: list[np.ndarray] = []
    ppg_ibi_sequences: list[np.ndarray] = []
    ecg_ibi_sequences: list[np.ndarray] = []
    ppg_ibi_lengths: list[int] = []
    ecg_ibi_lengths: list[int] = []
    skipped_missing_file = 0
    skipped_missing_channel = 0
    skipped_bad_length = 0
    skipped_processing = 0

    for row_idx, (_, source_row) in enumerate(ppg_summary.iterrows(), start=1):
        record_base = resolve_record_base(args.dataset_root, source_row)
        try:
            if not record_base.with_suffix(".hea").exists() or not record_base.with_suffix(".dat").exists():
                skipped_missing_file += 1
                continue
            raw_ppg, raw_ecg, raw_resp, fs = read_wfdb_channels(record_base)
        except KeyError:
            skipped_missing_channel += 1
            continue
        except Exception:
            skipped_processing += 1
            continue

        if int(round(fs)) != int(round(ppg_config.sample_rate_hz)):
            skipped_processing += 1
            continue
        if raw_ppg.shape[0] < segment_samples or raw_ecg.shape[0] < segment_samples:
            skipped_bad_length += 1
            continue

        raw_ppg = raw_ppg[:segment_samples]
        raw_ecg = raw_ecg[:segment_samples]
        try:
            ppg_processed = process_segment(raw_ppg, ppg_config)
            ecg_processed = process_segment(raw_ecg, ecg_config)
        except Exception:
            skipped_processing += 1
            continue

        if raw_resp is None or raw_resp.shape[0] < segment_samples:
            resp_available = 0
            resp_processed = np.zeros(segment_samples, dtype=np.float32)
            resp_metrics = {
                "resp_rate_bpm": float("nan"),
                "resp_spectral_entropy": float("nan"),
                "resp_ppg_amplitude_corr": float("nan"),
                "resp_ibi_corr": float("nan"),
            }
        else:
            resp_available = 1
            resp_processed = preprocess_respiration(raw_resp[:segment_samples], resp_config).astype(np.float32)
            resp_metrics = summarize_respiration(
                resp_processed,
                ppg_processed.normalized_signal,
                ppg_processed.peaks,
                ppg_config.sample_rate_hz,
            )

        ppg_ibi_padded, ppg_ibi_length = pad_sequence(ppg_processed.ibi_seconds)
        ecg_ibi_padded, ecg_ibi_length = pad_sequence(ecg_processed.ibi_seconds)
        timing_metrics = summarize_timing_consistency(ecg_processed.peaks, ppg_processed.peaks, ppg_config.sample_rate_hz)

        ppg_accepted = bool(source_row.get("accepted", True)) and bool(ppg_processed.quality_metrics["accepted"])
        ecg_accepted = bool(ecg_processed.quality_metrics["accepted"])
        joint_accepted = ppg_accepted and (ecg_accepted if args.require_ecg_accepted else True)
        label = int(source_row["label"])
        out_row: dict[str, Any] = {
            "record_id": str(source_row["record_id"]),
            "label": label,
            "label_source": str(source_row.get("label_source", "ppg_cache")),
            "segment_index": int(source_row.get("segment_index", 0)),
            "start_sample": 0,
            "end_sample": segment_samples,
            "start_time_sec": float(source_row.get("start_time_sec", 0.0)),
            "end_time_sec": float(source_row.get("end_time_sec", (segment_samples - 1) / ppg_config.sample_rate_hz)),
            "ppg_quality_score": float(source_row.get("quality_score", ppg_processed.quality_metrics["quality_score"])),
            "resp_available": resp_available,
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
        }
        for column in (
            "event_id",
            "segment_id",
            "signal_file_name",
            "patient",
            "folder_path",
            "subject_id",
            "hadm_id",
            "icustay_id",
            "event_rhythm",
            "strat_fold",
            "wfdb_record_path",
            "metadata_label",
        ):
            if column in source_row.index:
                out_row[column] = source_row[column]
        out_row.update(prefix_metrics(ppg_processed.quality_metrics, "ppg_"))
        out_row.update(prefix_metrics(ppg_processed.feature_metrics, "ppg_"))
        out_row.update(prefix_metrics(ecg_processed.quality_metrics, "ecg_"))
        out_row.update(prefix_metrics(ecg_processed.feature_metrics, "ecg_"))
        out_row.update(timing_metrics)
        out_row.update(resp_metrics)

        rows.append(out_row)
        ppg_segments.append(ppg_processed.normalized_signal.astype(np.float32))
        ecg_segments.append(ecg_processed.normalized_signal.astype(np.float32))
        resp_segments.append(resp_processed.astype(np.float32))
        ppg_ibi_sequences.append(ppg_ibi_padded)
        ecg_ibi_sequences.append(ecg_ibi_padded)
        ppg_ibi_lengths.append(ppg_ibi_length)
        ecg_ibi_lengths.append(ecg_ibi_length)

        if row_idx == 1 or row_idx % 1000 == 0 or row_idx == ppg_summary.shape[0]:
            print(
                f"processed {row_idx}/{ppg_summary.shape[0]} kept={len(rows)} "
                f"missing_file={skipped_missing_file} missing_channel={skipped_missing_channel} "
                f"bad_length={skipped_bad_length} processing={skipped_processing}",
                flush=True,
            )

    if not rows:
        summary_df, arrays = empty_bundle(segment_samples)
    else:
        summary_df = pd.DataFrame(rows)
        arrays = {
            "ppg_segments": np.stack(ppg_segments).astype(np.float32),
            "ecg_segments": np.stack(ecg_segments).astype(np.float32),
            "resp_segments": np.stack(resp_segments).astype(np.float32),
            "ppg_ibi_sequences": np.stack(ppg_ibi_sequences).astype(np.float32),
            "ecg_ibi_sequences": np.stack(ecg_ibi_sequences).astype(np.float32),
            "ppg_ibi_lengths": np.asarray(ppg_ibi_lengths, dtype=np.int16),
            "ecg_ibi_lengths": np.asarray(ecg_ibi_lengths, dtype=np.int16),
            "labels": summary_df["label"].to_numpy(dtype=np.int8),
            "ppg_quality_score": summary_df["ppg_quality_score"].to_numpy(dtype=np.float32),
            "joint_accepted": summary_df["joint_accepted"].to_numpy(dtype=bool),
            "start_time_sec": summary_df["start_time_sec"].to_numpy(dtype=np.float32),
        }

    saved = save_bundle(summary_df, arrays, args.output_dir, ppg_config, ecg_config, resp_config)
    print(
        json.dumps(
            {
                "source_rows": int(ppg_summary.shape[0]),
                "built_rows": int(summary_df.shape[0]),
                "accepted_rows": int(summary_df["joint_accepted"].sum()) if not summary_df.empty else 0,
                "label_counts": summary_df["label"].value_counts().sort_index().to_dict() if not summary_df.empty else {},
                "skipped_missing_file": skipped_missing_file,
                "skipped_missing_channel": skipped_missing_channel,
                "skipped_bad_length": skipped_bad_length,
                "skipped_processing": skipped_processing,
                "saved": {key: str(value) for key, value in saved.items()},
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
