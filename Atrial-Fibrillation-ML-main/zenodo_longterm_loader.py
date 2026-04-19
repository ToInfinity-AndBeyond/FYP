from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat


SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class ZenodoECGRecord:
    ecg: np.ndarray
    qrs_index: np.ndarray
    rr_seconds: np.ndarray
    af_annotation: np.ndarray
    sample_rate_hz: float
    start_day: int
    start_time_text: str


@dataclass(frozen=True)
class ZenodoPPGSegment:
    subject_id: str
    segment_index: int
    ppg_green: np.ndarray
    ppg_ambient: np.ndarray
    acc_x: np.ndarray
    acc_y: np.ndarray
    acc_z: np.ndarray
    ppg_sample_rate_hz: float
    acc_sample_rate_hz: float
    start_day: int
    start_time_text: str
    start_offset_sec: float


def _read_char_array(values: np.ndarray) -> str:
    flat = np.asarray(values).astype(np.uint16).flatten()
    return "".join(chr(int(value)) for value in flat if int(value))


def _read_hdf_value(file_handle: h5py.File, dataset: h5py.Dataset, index: int) -> np.ndarray:
    reference = dataset[index, 0]
    return np.asarray(file_handle[reference])


def _decode_header_value(file_handle: h5py.File, dataset: h5py.Dataset, index: int) -> Any:
    values = _read_hdf_value(file_handle, dataset, index)
    if values.dtype.kind in {"u", "i"}:
        text = _read_char_array(values)
        if text:
            return text
    flat = np.asarray(values).astype(float).flatten()
    if flat.size == 1:
        return float(flat[0])
    return flat


def _signal_header_to_dict(file_handle: h5py.File, signal_header: h5py.Group) -> dict[str, list[Any]]:
    header: dict[str, list[Any]] = {}
    for key in signal_header.keys():
        dataset = signal_header[key]
        header[key] = [
            _decode_header_value(file_handle, dataset, index)
            for index in range(dataset.shape[0])
        ]
    return header


def _parse_day_time(day_value: int | str, time_text: str) -> float:
    parts = time_text.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Unexpected time format: {time_text}")
    hours, minutes, seconds = (int(part) for part in parts)
    return (int(day_value) - 1) * SECONDS_PER_DAY + hours * 3600 + minutes * 60 + seconds


def _load_hdf5_ecg_mat(ecg_mat_path: Path) -> ZenodoECGRecord:
    with h5py.File(ecg_mat_path, "r") as file_handle:
        header = _signal_header_to_dict(file_handle, file_handle["signalHeader"])
        signal_labels = header["signal_labels"]
        sample_rates = header["samples_in_record"]
        label_to_rate = {str(label): float(rate) for label, rate in zip(signal_labels, sample_rates)}

        start_day = int(_read_char_array(np.asarray(file_handle["recording_startday"])))
        start_time_text = _read_char_array(np.asarray(file_handle["recording_starttime"]))
        ecg = np.asarray(file_handle["ECG"]).astype(np.float32).reshape(-1)
        qrs_index = np.asarray(file_handle["QRSindex"]).astype(np.float32).reshape(-1)
        rr_seconds = np.asarray(file_handle["rr"]).astype(np.float32).reshape(-1)
        af_annotation = np.asarray(file_handle["AF_annotation"]).astype(np.float32).reshape(-1)

    return ZenodoECGRecord(
        ecg=ecg,
        qrs_index=qrs_index,
        rr_seconds=rr_seconds,
        af_annotation=af_annotation,
        sample_rate_hz=label_to_rate["ECG"],
        start_day=start_day,
        start_time_text=start_time_text,
    )


def _mat_value_to_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _mat_value_to_scalar(value.reshape(-1)[0])
        if value.dtype.kind in {"U", "S"}:
            return "".join(str(item) for item in value.reshape(-1))
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _load_v5_ecg_mat(ecg_mat_path: Path) -> ZenodoECGRecord:
    mat = loadmat(
        ecg_mat_path,
        variable_names=[
            "signalHeader",
            "recording_startday",
            "recording_starttime",
            "ECG",
            "QRSindex",
            "rr",
            "AF_annotation",
        ],
        squeeze_me=True,
        struct_as_record=False,
    )
    signal_header = np.atleast_1d(mat["signalHeader"]).reshape(-1)
    label_to_rate: dict[str, float] = {}
    for item in signal_header:
        label = str(_mat_value_to_scalar(getattr(item, "signal_labels")))
        sample_rate = float(_mat_value_to_scalar(getattr(item, "samples_in_record")))
        label_to_rate[label] = sample_rate

    start_day = int(str(_mat_value_to_scalar(mat["recording_startday"])).strip())
    start_time_text = str(_mat_value_to_scalar(mat["recording_starttime"])).strip()
    ecg = np.asarray(mat["ECG"], dtype=np.float32).reshape(-1)
    qrs_index = np.asarray(mat["QRSindex"], dtype=np.float32).reshape(-1)
    rr_seconds = np.asarray(mat["rr"], dtype=np.float32).reshape(-1)
    af_annotation = np.asarray(mat["AF_annotation"], dtype=np.float32).reshape(-1)

    return ZenodoECGRecord(
        ecg=ecg,
        qrs_index=qrs_index,
        rr_seconds=rr_seconds,
        af_annotation=af_annotation,
        sample_rate_hz=label_to_rate["ECG"],
        start_day=start_day,
        start_time_text=start_time_text,
    )


def load_zenodo_ecg_mat(ecg_mat_path: Path) -> ZenodoECGRecord:
    if h5py.is_hdf5(ecg_mat_path):
        return _load_hdf5_ecg_mat(ecg_mat_path)
    return _load_v5_ecg_mat(ecg_mat_path)


def load_zenodo_ppg_mat(ppg_mat_path: Path, ecg_record: ZenodoECGRecord | None = None) -> tuple[list[ZenodoPPGSegment], dict[str, list[Any]]]:
    with h5py.File(ppg_mat_path, "r") as file_handle:
        header = _signal_header_to_dict(file_handle, file_handle["signalHeader"])
        signal_labels = header["signal_labels"]
        sample_rates = header["samples_in_record"]
        label_to_rate = {str(label): float(rate) for label, rate in zip(signal_labels, sample_rates)}

        subject_id = ppg_mat_path.stem.replace("_PPG", "")
        segments: list[ZenodoPPGSegment] = []
        if ecg_record is not None:
            ecg_start_sec = _parse_day_time(ecg_record.start_day, ecg_record.start_time_text)
        else:
            ecg_start_sec = 0.0

        segment_count = file_handle["PPG_GREEN"].shape[0]
        last_known_day: int | None = int(ecg_record.start_day) if ecg_record is not None else None
        for segment_index in range(segment_count):
            green = _read_hdf_value(file_handle, file_handle["PPG_GREEN"], segment_index).astype(np.float32).reshape(-1)
            ambient = _read_hdf_value(file_handle, file_handle["PPG_AMBIENT"], segment_index).astype(np.float32).reshape(-1)
            acc_x = _read_hdf_value(file_handle, file_handle["Accelerometer_X"], segment_index).astype(np.float32).reshape(-1)
            acc_y = _read_hdf_value(file_handle, file_handle["Accelerometer_Y"], segment_index).astype(np.float32).reshape(-1)
            acc_z = _read_hdf_value(file_handle, file_handle["Accelerometer_Z"], segment_index).astype(np.float32).reshape(-1)

            start_day_text = _read_char_array(_read_hdf_value(file_handle, file_handle["recording_startday"], segment_index)).strip()
            if start_day_text:
                start_day = int(start_day_text)
                last_known_day = start_day
            elif last_known_day is not None:
                start_day = int(last_known_day)
            else:
                raise ValueError(
                    f"Missing recording_startday for segment {segment_index} in {ppg_mat_path} and no fallback day is available."
                )
            start_time_text = _read_char_array(_read_hdf_value(file_handle, file_handle["recording_starttime"], segment_index))
            if not start_time_text.strip():
                raise ValueError(f"Missing recording_starttime for segment {segment_index} in {ppg_mat_path}.")
            segment_start_sec = _parse_day_time(start_day, start_time_text)
            start_offset_sec = segment_start_sec - ecg_start_sec if ecg_record is not None else segment_start_sec

            segments.append(
                ZenodoPPGSegment(
                    subject_id=subject_id,
                    segment_index=segment_index,
                    ppg_green=green,
                    ppg_ambient=ambient,
                    acc_x=acc_x,
                    acc_y=acc_y,
                    acc_z=acc_z,
                    ppg_sample_rate_hz=label_to_rate["PPG_GREEN"],
                    acc_sample_rate_hz=label_to_rate["Accelerometer_X"],
                    start_day=start_day,
                    start_time_text=start_time_text,
                    start_offset_sec=float(start_offset_sec),
                )
            )

    return segments, header


def summarize_ppg_segments(segments: list[ZenodoPPGSegment]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        duration_sec = float(segment.ppg_green.size / segment.ppg_sample_rate_hz)
        rows.append(
            {
                "segment_index": segment.segment_index,
                "start_day": int(segment.start_day),
                "start_time_text": segment.start_time_text,
                "start_offset_sec": float(segment.start_offset_sec),
                "duration_sec": duration_sec,
                "ppg_samples": int(segment.ppg_green.size),
                "acc_samples": int(segment.acc_x.size),
                "ppg_sample_rate_hz": float(segment.ppg_sample_rate_hz),
                "acc_sample_rate_hz": float(segment.acc_sample_rate_hz),
            }
        )
    return pd.DataFrame(rows)


def build_subject_window_labels(
    ecg_record: ZenodoECGRecord,
    ppg_segments: list[ZenodoPPGSegment],
    window_length_sec: float = 30.0,
    stride_sec: float = 30.0,
    positive_fraction_threshold: float = 0.5,
) -> pd.DataFrame:
    interval_start_sec = ecg_record.qrs_index[:-1] / ecg_record.sample_rate_hz
    interval_end_sec = ecg_record.qrs_index[1:] / ecg_record.sample_rate_hz
    interval_af = ecg_record.af_annotation[: interval_start_sec.size] > 0.5

    rows: list[dict[str, Any]] = []
    for segment in ppg_segments:
        window_samples = int(round(window_length_sec * segment.ppg_sample_rate_hz))
        stride_samples = int(round(stride_sec * segment.ppg_sample_rate_hz))
        if segment.ppg_green.size < window_samples:
            continue

        for start_sample in range(0, segment.ppg_green.size - window_samples + 1, stride_samples):
            segment_window_start_sec = start_sample / segment.ppg_sample_rate_hz
            window_start_sec = segment.start_offset_sec + segment_window_start_sec
            window_end_sec = window_start_sec + window_length_sec

            left_index = int(np.searchsorted(interval_end_sec, window_start_sec, side="right"))
            right_index = int(np.searchsorted(interval_start_sec, window_end_sec, side="left"))
            if right_index <= left_index:
                af_fraction = 0.0
            else:
                current_start = interval_start_sec[left_index:right_index]
                current_end = interval_end_sec[left_index:right_index]
                overlap = np.clip(
                    np.minimum(current_end, window_end_sec) - np.maximum(current_start, window_start_sec),
                    0.0,
                    None,
                )
                af_overlap = float(overlap[interval_af[left_index:right_index]].sum())
                af_fraction = af_overlap / window_length_sec

            if af_fraction >= positive_fraction_threshold:
                label = 1
                use_for_training = 1
            elif af_fraction == 0.0:
                label = 0
                use_for_training = 1
            else:
                label = np.nan
                use_for_training = 0

            rows.append(
                {
                    "record_id": ppg_mat_record_id(ppg_segments),
                    "ppg_segment_index": int(segment.segment_index),
                    "segment_index": int(start_sample // stride_samples),
                    "global_window_id": f"{ppg_mat_record_id(ppg_segments)}_seg{segment.segment_index:02d}_win{int(start_sample // stride_samples):04d}",
                    "start_time_sec": float(window_start_sec),
                    "end_time_sec": float(window_end_sec),
                    "segment_relative_start_sec": float(segment_window_start_sec),
                    "label": label,
                    "use_for_training": int(use_for_training),
                    "label_source": "ecg_af_annotation",
                    "af_fraction": float(af_fraction),
                }
            )

    return pd.DataFrame(rows)


def ppg_mat_record_id(ppg_segments: list[ZenodoPPGSegment]) -> str:
    if not ppg_segments:
        return "unknown_subject"
    return f"zenodo_{ppg_segments[0].subject_id}"
