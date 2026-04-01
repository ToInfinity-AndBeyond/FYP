from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request

import numpy as np
import pandas as pd
import vitaldb


ANNOTATION_BASE_URL = "https://physionet.org/files/vitaldb-arrhythmia/1.0.0"
METADATA_FILENAME = "metadata.csv"
ANNOTATION_DIRNAME = "Annotation_Files"

PPG_TRACK_CANDIDATES = ("SNUADC/PLETH", "SNUADCM/PLETH", "PLETH")
ECG_TRACK_CANDIDATES = ("SNUADC/ECG_II", "SNUADCM/ECG_II", "ECG_II")

RHYTHM_LABEL_ALIASES = {
    "N": "N",
    "NORMAL SINUS RHYTHM": "N",
    "AFIB/AFL": "AFIB/AFL",
    "ATRIAL FIBRILLATION": "AFIB/AFL",
    "ATRIAL FIBRILLATION / ATRIAL FLUTTER": "AFIB/AFL",
    "NOISE": "Noise",
    "PATTERNED ATRIAL ECTOPY": "Patterned Atrial Ectopy",
    "PATTERNED VENTRICULAR ECTOPY": "Patterned Ventricular Ectopy",
    "SVTA": "SVTA",
    "SUPRAVENTRICULAR TACHYARRHYTHMIA": "SVTA",
    "VT": "VT",
    "VENTRICULAR TACHYARRHYTHMIA": "VT",
    "SND": "SND",
    "SINUS NODE DYSFUNCTION": "SND",
    "WAP/MAT": "WAP/MAT",
    "WANDERING ATRIAL PACEMAKER / MULTIFOCAL ATRIAL RHYTHM": "WAP/MAT",
    "AVB": "AVB",
    "ATRIOVENTRICULAR BLOCK": "AVB",
    "UNCLASSIFIABLE": "Unclassifiable",
}

AF_RHYTHMS = {"AFIB/AFL"}
NORMAL_RHYTHMS = {"N"}


@dataclass(frozen=True)
class VitalDBWaveformRecord:
    case_id: int
    ppg: np.ndarray
    ecg: np.ndarray
    sample_rate_hz: float
    ppg_track_name: str
    ecg_track_name: str
    analysis_start_time_sec: float
    analysis_end_time_sec: float


def _coerce_bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).astype(float) != 0.0
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def canonicalize_rhythm_label(label: Any) -> str:
    if pd.isna(label):
        return ""
    text = str(label).strip()
    if not text:
        return ""
    return RHYTHM_LABEL_ALIASES.get(text.upper(), text)


def is_af_label(label: str) -> bool:
    return canonicalize_rhythm_label(label) in AF_RHYTHMS


def is_normal_label(label: str) -> bool:
    return canonicalize_rhythm_label(label) in NORMAL_RHYTHMS


def ensure_metadata_csv(annotation_root: Path, download_if_missing: bool = True) -> Path:
    annotation_root.mkdir(parents=True, exist_ok=True)
    metadata_path = annotation_root / METADATA_FILENAME
    if metadata_path.exists():
        return metadata_path
    if not download_if_missing:
        raise FileNotFoundError(f"Missing VitalDB arrhythmia metadata: {metadata_path}")
    request.urlretrieve(f"{ANNOTATION_BASE_URL}/{METADATA_FILENAME}", metadata_path)
    return metadata_path


def load_annotation_metadata(annotation_root: Path, download_if_missing: bool = True) -> pd.DataFrame:
    metadata_path = ensure_metadata_csv(annotation_root, download_if_missing=download_if_missing)
    metadata_df = pd.read_csv(metadata_path)
    if "case_id" not in metadata_df.columns:
        raise ValueError(f"VitalDB arrhythmia metadata is missing case_id: {metadata_path}")
    metadata_df = metadata_df.copy()
    metadata_df["case_id"] = metadata_df["case_id"].astype(int)
    return metadata_df.sort_values("case_id").reset_index(drop=True)


def ensure_annotation_file(case_id: int, annotation_root: Path, download_if_missing: bool = True) -> Path:
    annotation_dir = annotation_root / ANNOTATION_DIRNAME
    annotation_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = annotation_dir / f"Annotation_file_{int(case_id)}.csv"
    if annotation_path.exists():
        return annotation_path
    if not download_if_missing:
        raise FileNotFoundError(f"Missing VitalDB annotation file: {annotation_path}")
    request.urlretrieve(
        f"{ANNOTATION_BASE_URL}/{ANNOTATION_DIRNAME}/Annotation_file_{int(case_id)}.csv",
        annotation_path,
    )
    return annotation_path


def load_case_annotations(
    case_id: int,
    annotation_root: Path,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    annotation_path = ensure_annotation_file(case_id, annotation_root, download_if_missing=download_if_missing)
    annotation_df = pd.read_csv(annotation_path)
    required_columns = {
        "time_second",
        "beat_type",
        "rhythm_label",
        "bad_signal_quality",
        "bad_signal_quality_label",
    }
    missing = required_columns - set(annotation_df.columns)
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise ValueError(f"VitalDB annotation file is missing columns: {missing_display}")

    annotation_df = annotation_df.copy()
    annotation_df["time_second"] = annotation_df["time_second"].astype(float)
    annotation_df["rhythm_label_raw"] = annotation_df["rhythm_label"].astype(str)
    annotation_df["rhythm_label"] = annotation_df["rhythm_label"].map(canonicalize_rhythm_label)
    annotation_df["bad_signal_quality"] = _coerce_bool_series(annotation_df["bad_signal_quality"])
    return annotation_df.sort_values("time_second").reset_index(drop=True)


def _pick_track_name(available_tracks: list[str], candidates: tuple[str, ...]) -> str | None:
    available = [str(track) for track in available_tracks]
    available_set = set(available)

    for candidate in candidates:
        if candidate in available_set:
            return candidate

    for candidate in candidates:
        for track_name in available:
            if track_name.endswith(candidate):
                return track_name
    return None


def resolve_case_track_names(case_id: int) -> dict[str, str | None]:
    track_df = vitaldb.get_track_names([int(case_id)])
    if track_df.empty:
        return {"ppg": None, "ecg": None}
    available_tracks = track_df.iloc[0]["tnames"]
    return {
        "ppg": _pick_track_name(available_tracks, PPG_TRACK_CANDIDATES),
        "ecg": _pick_track_name(available_tracks, ECG_TRACK_CANDIDATES),
    }


def load_case_waveforms(
    case_id: int,
    analysis_start_time_sec: float,
    analysis_end_time_sec: float,
    target_sample_rate_hz: float = 125.0,
) -> VitalDBWaveformRecord | None:
    tracks = resolve_case_track_names(case_id)
    ppg_track = tracks["ppg"]
    ecg_track = tracks["ecg"]
    if ppg_track is None or ecg_track is None:
        return None

    values = vitaldb.load_case(int(case_id), [ppg_track, ecg_track], 1 / target_sample_rate_hz)
    values = np.asarray(values, dtype=float)
    if values.size == 0 or values.ndim != 2 or values.shape[1] < 2:
        return None

    ppg = values[:, 0].astype(np.float32)
    ecg = values[:, 1].astype(np.float32)
    if ppg.size == 0 or ecg.size == 0:
        return None
    if np.all(np.isnan(ppg)) or np.all(np.isnan(ecg)):
        return None

    return VitalDBWaveformRecord(
        case_id=int(case_id),
        ppg=ppg,
        ecg=ecg,
        sample_rate_hz=float(target_sample_rate_hz),
        ppg_track_name=str(ppg_track),
        ecg_track_name=str(ecg_track),
        analysis_start_time_sec=float(analysis_start_time_sec),
        analysis_end_time_sec=float(analysis_end_time_sec),
    )


def build_case_window_labels(
    case_id: int,
    metadata_row: pd.Series,
    annotation_df: pd.DataFrame,
    window_length_sec: float = 30.0,
    stride_sec: float = 30.0,
    positive_fraction_threshold: float = 0.5,
    min_annotation_coverage_fraction: float = 0.8,
    negative_label_policy: str = "normal_only",
    max_bad_quality_fraction: float = 1.0,
) -> pd.DataFrame:
    if annotation_df.shape[0] < 2:
        return pd.DataFrame()

    beat_times = annotation_df["time_second"].to_numpy(dtype=float)
    interval_start_sec = beat_times[:-1]
    interval_end_sec = beat_times[1:]
    valid_mask = interval_end_sec > interval_start_sec
    if not np.any(valid_mask):
        return pd.DataFrame()

    interval_start_sec = interval_start_sec[valid_mask]
    interval_end_sec = interval_end_sec[valid_mask]
    interval_rhythm = annotation_df["rhythm_label"].to_numpy(dtype=object)[:-1][valid_mask]
    interval_bad_quality = (
        annotation_df["bad_signal_quality"].to_numpy(dtype=bool)[:-1][valid_mask]
        | annotation_df["bad_signal_quality"].to_numpy(dtype=bool)[1:][valid_mask]
    )

    analysis_start_time_sec = float(metadata_row.get("analysis_start_time_sec", interval_start_sec.min()))
    analysis_end_time_sec = float(metadata_row.get("analysis_end_time_sec", interval_end_sec.max()))
    analysis_start_time_sec = max(analysis_start_time_sec, float(interval_start_sec.min()))
    analysis_end_time_sec = min(analysis_end_time_sec, float(interval_end_sec.max()))
    if analysis_end_time_sec - analysis_start_time_sec < window_length_sec:
        return pd.DataFrame()

    last_window_start = analysis_end_time_sec - window_length_sec
    window_starts = np.arange(analysis_start_time_sec, last_window_start + 1e-9, stride_sec, dtype=float)
    rows: list[dict[str, Any]] = []

    for segment_index, window_start_sec in enumerate(window_starts):
        window_end_sec = float(window_start_sec + window_length_sec)
        left_index = int(np.searchsorted(interval_end_sec, window_start_sec, side="right"))
        right_index = int(np.searchsorted(interval_start_sec, window_end_sec, side="left"))

        coverage_sec = 0.0
        af_sec = 0.0
        normal_sec = 0.0
        bad_quality_sec = 0.0

        if right_index > left_index:
            current_start = interval_start_sec[left_index:right_index]
            current_end = interval_end_sec[left_index:right_index]
            overlap = np.clip(
                np.minimum(current_end, window_end_sec) - np.maximum(current_start, window_start_sec),
                0.0,
                None,
            )
            coverage_sec = float(overlap.sum())
            af_mask = np.isin(interval_rhythm[left_index:right_index], list(AF_RHYTHMS))
            normal_mask = np.isin(interval_rhythm[left_index:right_index], list(NORMAL_RHYTHMS))
            bad_quality_mask = interval_bad_quality[left_index:right_index]
            af_sec = float(overlap[af_mask].sum())
            normal_sec = float(overlap[normal_mask].sum())
            bad_quality_sec = float(overlap[bad_quality_mask].sum())

        coverage_fraction = coverage_sec / window_length_sec
        af_fraction = af_sec / window_length_sec
        normal_fraction = normal_sec / window_length_sec
        bad_quality_fraction = bad_quality_sec / window_length_sec
        other_fraction = max(coverage_fraction - af_fraction - normal_fraction, 0.0)

        label = np.nan
        use_for_training = 0
        label_source = "vitaldb_arrhythmia_annotation"
        exclusion_reason = ""

        if coverage_fraction < min_annotation_coverage_fraction:
            exclusion_reason = "insufficient_annotation_coverage"
        elif bad_quality_fraction > max_bad_quality_fraction:
            exclusion_reason = "excessive_bad_signal_quality"
        elif af_fraction >= positive_fraction_threshold:
            label = 1
            use_for_training = 1
            label_source = "vitaldb_arrhythmia_af"
        elif negative_label_policy == "all_non_af":
            if af_fraction == 0.0:
                label = 0
                use_for_training = 1
                label_source = "vitaldb_arrhythmia_non_af"
            else:
                exclusion_reason = "mixed_af_window"
        else:
            if af_fraction == 0.0 and other_fraction <= 1e-6 and normal_fraction >= min_annotation_coverage_fraction:
                label = 0
                use_for_training = 1
                label_source = "vitaldb_arrhythmia_normal"
            else:
                exclusion_reason = "non_normal_non_af_window"

        rows.append(
            {
                "record_id": f"vitaldb_{int(case_id)}",
                "case_id": int(case_id),
                "segment_index": int(segment_index),
                "global_window_id": f"vitaldb_{int(case_id)}_win{int(segment_index):04d}",
                "start_time_sec": float(window_start_sec),
                "end_time_sec": float(window_end_sec),
                "analysis_start_time_sec": float(analysis_start_time_sec),
                "analysis_end_time_sec": float(analysis_end_time_sec),
                "label": label,
                "use_for_training": int(use_for_training),
                "label_source": label_source,
                "annotation_coverage_fraction": float(coverage_fraction),
                "af_fraction": float(af_fraction),
                "normal_fraction": float(normal_fraction),
                "other_rhythm_fraction": float(other_fraction),
                "bad_signal_quality_fraction": float(bad_quality_fraction),
                "rhythm_classes": str(metadata_row.get("rhythm_classes", "")),
                "exclusion_reason": exclusion_reason,
            }
        )

    return pd.DataFrame(rows)
