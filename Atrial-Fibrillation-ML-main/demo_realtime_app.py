#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cgi
import html
import io
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from af_pipeline.features import FEATURE_COLUMNS
from ppg_hybrid_model import RhythmMorphologyFusionNet
from signal_pipeline import default_ppg_config, process_dataframe, process_segment


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "models" / "mimic_ext_sqi_v2" / "model.pt"
TARGET_FS = 125.0


def page(title: str, body: str) -> bytes:
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #eef1f3;
      --panel: #fff;
      --ink: #20262b;
      --muted: #667079;
      --line: #cdd3d8;
      --accent: #204d68;
      --red: #a83232;
      --amber: #9a6300;
      --blue: #315f7a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    }}
    header {{
      background: #fff;
      color: var(--ink);
      padding: 16px clamp(18px, 3vw, 42px);
      border-bottom: 3px solid var(--accent);
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: center;
      flex-wrap: wrap;
    }}
    h1, h2, h3, p {{ margin: 0; }}
    h1 {{ font-size: 1.25rem; line-height: 1.2; letter-spacing: -.01em; }}
    h2 {{ font-size: 1rem; font-weight: 700; }}
    main {{
      width: min(1080px, calc(100vw - 28px));
      margin: 24px auto 42px;
      display: grid;
      gap: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 3px;
      padding: 18px;
    }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); border: 1px solid var(--line); background: #fafbfc; }}
    .metric {{ border-right: 1px solid var(--line); padding: 13px 14px; min-height: 76px; }}
    .metric:last-child {{ border-right: 0; }}
    .metric span {{ display: block; color: var(--muted); font-size: .72rem; font-weight: 700; letter-spacing: .03em; text-transform: uppercase; }}
    .metric strong {{ display: block; margin-top: 7px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 1.25rem; font-weight: 600; overflow-wrap: anywhere; }}
    .chips {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .chip {{ display: inline-flex; align-items: center; gap: 6px; min-height: 25px; padding: 2px 5px; color: var(--muted); font-size: .78rem; font-weight: 650; }}
    .dot {{ width: 9px; height: 9px; border-radius: 999px; display: inline-block; }}
    form {{ display: grid; gap: 12px; }}
    .form-grid {{ display: grid; grid-template-columns: minmax(280px, 1fr) auto; gap: 10px; align-items: end; }}
    label {{ display: grid; gap: 6px; color: var(--muted); font-weight: 750; font-size: .82rem; }}
    input, select, button {{
      width: 100%;
      min-height: 42px;
      border-radius: 2px;
      border: 1px solid var(--line);
      padding: 8px 10px;
      font: inherit;
    }}
    button {{ border-color: var(--accent); background: var(--accent); color: #fff; font-weight: 700; cursor: pointer; padding: 0 18px; }}
    button:hover {{ background: #173d54; }}
    .status {{ display: inline-flex; align-items: center; min-height: 30px; padding: 4px 10px; border-left: 4px solid; font-size: .88rem; font-weight: 700; }}
    .af {{ background: #f8eeee; color: var(--red); border-color: var(--red); }}
    .sr {{ background: #edf4f1; color: #28614d; border-color: #28614d; }}
    .warn {{ background: #fbf5e8; color: var(--amber); border-color: var(--amber); }}
    svg {{ width: 100%; height: 280px; border: 1px solid var(--line); border-radius: 2px; background: #fff; }}
    .tall-svg {{ height: 420px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .88rem; min-width: 760px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #e5eaee; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child, th:last-child, td:last-child {{ text-align: left; }}
    th {{ color: #4e5d66; font-size: .76rem; text-transform: uppercase; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 2px; }}
    .muted {{ color: var(--muted); line-height: 1.45; }}
    .error {{ color: var(--red); font-weight: 750; }}
    details {{ border-top: 1px solid var(--line); margin-top: 14px; padding-top: 12px; }}
    summary {{ color: var(--accent); cursor: pointer; font-weight: 700; }}
    .method {{ margin-top: 10px; color: var(--muted); font-size: .9rem; line-height: 1.55; }}
    @media (max-width: 820px) {{
      .grid, .form-grid {{ grid-template-columns: 1fr; }}
      .metric {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .metric:last-child {{ border-bottom: 0; }}
      header {{ align-items: start; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <p style="color:var(--accent);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.72rem;font-weight:700;letter-spacing:.08em;margin-bottom:4px;">AF-PPG</p>
      <h1>Signal analysis console</h1>
    </div>
  </header>
  <main>{body}</main>
</body>
</html>"""
    return document.encode("utf-8")


def upload_form(message: str = "") -> str:
    alert = f'<div class="panel error">{html.escape(message)}</div>' if message else ""
    return f"""{alert}
<section class="panel">
  <h2>Input</h2>
  <form method="post" enctype="multipart/form-data">
    <div class="form-grid" style="margin-top:14px;">
      <label>CSV file
        <input type="file" name="csv_file" accept=".csv,text/csv" required>
      </label>
      <button type="submit">Run analysis</button>
    </div>
    <p class="muted">Upload a raw PPG CSV. The signal column and sampling rate are detected automatically; files without time data are treated as 125 Hz.</p>
  </form>
</section>"""


def load_checkpoint(model_path: Path) -> tuple[RhythmMorphologyFusionNet, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = RhythmMorphologyFusionNet(
        feature_dim=len(FEATURE_COLUMNS),
        signal_length=int(TARGET_FS * 30),
        active_branches=("time", "spectral", "feature"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def select_ppg_column(dataframe: pd.DataFrame, requested: str) -> str:
    summary_columns = {"record_id", "segment_index", "quality_score", "accepted"}
    if summary_columns.issubset(set(dataframe.columns)) and requested not in dataframe.columns:
        raise ValueError(
            "This looks like a segment summary/features CSV, not a raw PPG waveform CSV. "
            "Upload a *_data.csv file with a PPG column, for example "
            "mimic_perform_af_csv/mimic_perform_af_014_data.csv. "
            "The demo_data files can explain SQI results, but they do not contain the actual PPG samples "
            "needed for filtering, beat detection, and model inference."
        )
    if requested in dataframe.columns:
        return requested
    for candidate in ("PPG", "ppg", "PLETH", "Pleth", "pleth", "signal", "Signal"):
        if candidate in dataframe.columns:
            return candidate
    time_columns = {"Time", "time", "TIME", "timestamp", "Timestamp"}
    numeric_columns = [
        column
        for column in dataframe.select_dtypes(include=[np.number]).columns.tolist()
        if column not in time_columns
    ]
    if len(numeric_columns) == 1:
        return numeric_columns[0]
    available = ", ".join(map(str, dataframe.columns[:12]))
    raise ValueError(f"PPG column '{requested}' was not found. Available columns include: {available}")


def infer_sample_rate(dataframe: pd.DataFrame, fallback: float = TARGET_FS) -> float:
    for column in ("Time", "time", "TIME", "timestamp", "Timestamp"):
        if column not in dataframe.columns:
            continue
        time_values = pd.to_numeric(dataframe[column], errors="coerce").to_numpy(dtype=float)
        time_values = time_values[np.isfinite(time_values)]
        if time_values.size < 2:
            continue
        intervals = np.diff(time_values)
        intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
        if intervals.size == 0:
            continue
        sample_rate = 1.0 / float(np.median(intervals))
        if 10.0 <= sample_rate <= 1000.0:
            return sample_rate
    return fallback


def resample_to_target(dataframe: pd.DataFrame, ppg_column: str, input_fs: float) -> pd.DataFrame:
    signal = pd.to_numeric(dataframe[ppg_column], errors="coerce").to_numpy(dtype=float)
    if np.all(~np.isfinite(signal)):
        raise ValueError("The selected PPG column has no numeric values.")
    finite = np.isfinite(signal)
    if not finite.all():
        valid_idx = np.flatnonzero(finite)
        signal[~finite] = np.interp(np.flatnonzero(~finite), valid_idx, signal[valid_idx])

    if input_fs <= 0:
        raise ValueError("Sample rate must be greater than 0.")
    if math.isclose(input_fs, TARGET_FS, rel_tol=0.0, abs_tol=1e-6):
        return pd.DataFrame({"Time": np.arange(signal.size, dtype=float) / TARGET_FS, "PPG": signal})

    duration = signal.size / input_fs
    target_count = max(int(round(duration * TARGET_FS)), 1)
    source_t = np.arange(signal.size, dtype=float) / input_fs
    target_t = np.arange(target_count, dtype=float) / TARGET_FS
    resampled = np.interp(target_t, source_t, signal)
    return pd.DataFrame({"Time": target_t, "PPG": resampled})


def scale_features(summary_df: pd.DataFrame, checkpoint: dict[str, Any]) -> np.ndarray:
    feature_frame = summary_df[FEATURE_COLUMNS].copy()
    normalization = checkpoint["normalization"]
    medians = np.asarray(normalization["feature_medians"], dtype=np.float32)
    means = np.asarray(normalization["feature_means"], dtype=np.float32)
    stds = np.asarray(normalization["feature_stds"], dtype=np.float32)
    stds[stds == 0.0] = 1.0
    feature_frame = feature_frame.fillna(dict(zip(FEATURE_COLUMNS, medians)))
    values = feature_frame.to_numpy(dtype=np.float32)
    values = (values - means) / stds
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def split_raw_segments(signal_df: pd.DataFrame, segment_samples: int, stride_samples: int) -> np.ndarray:
    signal_values = signal_df["PPG"].to_numpy(dtype=np.float32)
    segments = []
    for start in range(0, signal_values.size - segment_samples + 1, stride_samples):
        segments.append(signal_values[start : start + segment_samples])
    if not segments:
        return np.empty((0, segment_samples), dtype=np.float32)
    return np.stack(segments).astype(np.float32)


def aggregate_probability(probabilities: np.ndarray, quality_scores: np.ndarray | None = None) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    valid_mask = np.isfinite(probabilities)
    if not valid_mask.any():
        return float("nan")
    valid_probs = probabilities[valid_mask]
    if quality_scores is None:
        return float(np.mean(valid_probs))
    valid_quality = np.asarray(quality_scores, dtype=np.float32)[valid_mask]
    valid_quality = np.nan_to_num(valid_quality, nan=0.0, posinf=0.0, neginf=0.0)
    valid_quality = np.clip(valid_quality, 0.0, 1.0)
    if float(valid_quality.sum()) <= 0.0:
        return float(np.mean(valid_probs))
    return float(np.average(valid_probs, weights=valid_quality))


def decision_from_probability(probability: float, threshold: float, no_call: bool = False) -> str:
    if no_call or not np.isfinite(probability):
        return "No-call: all windows failed quality gate"
    return "AF suspected" if probability >= threshold else "Sinus rhythm likely"


def predict_csv(
    data: bytes,
    ppg_column: str,
    sample_rate: float | None,
    model_path: Path,
    sqi_mode: str = "on",
) -> dict[str, Any]:
    model, checkpoint = load_checkpoint(model_path)
    raw_df = pd.read_csv(io.BytesIO(data))
    selected_column = select_ppg_column(raw_df, ppg_column.strip() or "PPG")
    resolved_sample_rate = sample_rate or infer_sample_rate(raw_df)
    signal_df = resample_to_target(raw_df, selected_column, resolved_sample_rate)

    config = default_ppg_config(sample_rate_hz=TARGET_FS)
    segment_samples = int(round(config.segment.length_seconds * config.sample_rate_hz))
    stride_samples = int(round(config.segment.stride_seconds * config.sample_rate_hz))
    raw_segments = split_raw_segments(signal_df, segment_samples, stride_samples)
    summary_df, segments = process_dataframe(
        dataframe=signal_df,
        label=0,
        record_id="uploaded_ppg",
        config=config,
    )
    if summary_df.empty:
        raise ValueError("Not enough samples for a 30-second PPG window after resampling.")

    accepted_mask = summary_df["accepted"].astype(bool).to_numpy()
    features = scale_features(summary_df.reset_index(drop=True), checkpoint)
    waveforms = torch.from_numpy(segments.astype(np.float32))
    feature_tensor = torch.from_numpy(features)
    with torch.no_grad():
        logits = model(waveforms, feature_tensor)
        all_probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    sqi_probabilities = np.full(summary_df.shape[0], np.nan, dtype=np.float32)
    sqi_probabilities[accepted_mask] = all_probabilities[accepted_mask]

    if "best_threshold" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain best_threshold: {model_path}")
    threshold = float(checkpoint["best_threshold"])
    accepted_quality = summary_df.loc[accepted_mask, "quality_score"].to_numpy(dtype=np.float32)
    sqi_record_probability = aggregate_probability(sqi_probabilities[accepted_mask], accepted_quality)
    no_sqi_record_probability = aggregate_probability(all_probabilities)
    sqi_enabled = sqi_mode != "off"
    if sqi_enabled:
        record_prob = sqi_record_probability
        decision = decision_from_probability(record_prob, threshold, no_call=not accepted_mask.any())
        active_probabilities = sqi_probabilities
    else:
        record_prob = no_sqi_record_probability
        decision = decision_from_probability(record_prob, threshold)
        active_probabilities = all_probabilities

    return {
        "sqi_enabled": sqi_enabled,
        "selected_column": selected_column,
        "input_samples": int(raw_df.shape[0]),
        "resampled_samples": int(signal_df.shape[0]),
        "sample_rate": resolved_sample_rate,
        "threshold": threshold,
        "record_probability": record_prob,
        "sqi_record_probability": sqi_record_probability,
        "no_sqi_record_probability": no_sqi_record_probability,
        "decision": decision,
        "summary": summary_df,
        "segments": segments,
        "raw_segments": raw_segments,
        "probabilities": active_probabilities,
        "sqi_probabilities": sqi_probabilities,
        "all_probabilities": all_probabilities,
        "processing": {
            "target_sample_rate": TARGET_FS,
            "window_seconds": config.segment.length_seconds,
            "stride_seconds": config.segment.stride_seconds,
            "bandpass_low_hz": config.bandpass.low_hz,
            "bandpass_high_hz": config.bandpass.high_hz,
            "bandpass_order": config.bandpass.order,
            "quality_rules": {
                "min_heart_band_energy_ratio": config.quality.min_heart_band_energy_ratio,
                "min_template_correlation": config.quality.min_template_correlation,
                "min_peak_count": config.quality.min_peak_count,
                "max_peak_count": config.quality.max_peak_count,
                "min_hr_bpm": config.quality.min_hr_bpm,
                "max_hr_bpm": config.quality.max_hr_bpm,
                "min_snr_sqi": config.quality.min_snr_sqi,
                "min_detector_agreement": config.quality.min_detector_agreement,
            },
        },
    }


def _line_points(values: np.ndarray, width: int, top: float, height: float, pad_x: int, max_points: int = 900) -> str:
    values = np.asarray(values, dtype=float)
    if values.size > max_points:
        idx = np.linspace(0, values.size - 1, max_points).astype(int)
        values = values[idx]
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    min_v, max_v = float(values.min()), float(values.max())
    span = max(max_v - min_v, 1e-6)
    points = []
    for index, value in enumerate(values):
        x = pad_x + index * (width - 2 * pad_x) / max(values.size - 1, 1)
        y = top + height - ((value - min_v) / span) * height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def processing_svg(raw_segment: np.ndarray, processed: Any, threshold: float, probability: float | None) -> str:
    width, height, pad_x = 1000, 390, 30
    band_height = 92
    raw_y, filtered_y, norm_y = 40, 158, 276
    color = "#bd3c32" if probability is not None and probability >= threshold else "#007f73"
    raw_points = _line_points(raw_segment, width, raw_y, band_height, pad_x)
    filtered_points = _line_points(processed.filtered_signal, width, filtered_y, band_height, pad_x)
    normalized_points = _line_points(processed.normalized_signal, width, norm_y, band_height, pad_x)
    peak_marks = []
    signal_length = max(len(processed.normalized_signal) - 1, 1)
    for peak in processed.peaks:
        x = pad_x + int(peak) * (width - 2 * pad_x) / signal_length
        peak_marks.append(f'<line x1="{x:.1f}" y1="{norm_y}" x2="{x:.1f}" y2="{norm_y + band_height}" stroke="#b86b00" stroke-width="1.2" opacity="0.65"/>')
    return f"""<svg class="tall-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Signal processing stages">
  <text x="{pad_x}" y="20" fill="#63717a" font-size="14" font-weight="750">Raw uploaded PPG</text>
  <text x="{pad_x}" y="138" fill="#63717a" font-size="14" font-weight="750">0.5-8 Hz band-pass filtered</text>
  <text x="{pad_x}" y="256" fill="#63717a" font-size="14" font-weight="750">Z-score normalized + detected beats</text>
  <line x1="{pad_x}" y1="{raw_y + band_height / 2:.1f}" x2="{width - pad_x}" y2="{raw_y + band_height / 2:.1f}" stroke="#e5eaee"/>
  <line x1="{pad_x}" y1="{filtered_y + band_height / 2:.1f}" x2="{width - pad_x}" y2="{filtered_y + band_height / 2:.1f}" stroke="#e5eaee"/>
  <line x1="{pad_x}" y1="{norm_y + band_height / 2:.1f}" x2="{width - pad_x}" y2="{norm_y + band_height / 2:.1f}" stroke="#e5eaee"/>
  <polyline fill="none" stroke="#63717a" stroke-width="1.6" points="{raw_points}"/>
  <polyline fill="none" stroke="#2d6b9f" stroke-width="1.8" points="{filtered_points}"/>
  {''.join(peak_marks)}
  <polyline fill="none" stroke="{color}" stroke-width="2.0" points="{normalized_points}"/>
</svg>"""


def format_float(value: Any, digits: int = 3, fallback: str = "N/A") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(numeric):
        return fallback
    return f"{numeric:.{digits}f}"


def processing_details(result: dict[str, Any], preview_index: int, preview_prob: float | None) -> str:
    config = default_ppg_config(sample_rate_hz=TARGET_FS)
    raw_segment = result["raw_segments"][preview_index]
    processed = process_segment(raw_segment, config)
    row = result["summary"].iloc[preview_index]
    if preview_prob is None:
        probability_text = "not used by SQI"
    else:
        probability_text = f"{preview_prob:.3f}"
    accepted_text = "accepted" if bool(row["accepted"]) else "rejected"
    reason = str(row.get("rejection_reason", "")) or "passed all quality checks"
    return f"""
<section class="panel">
  <div style="display:flex;justify-content:space-between;gap:12px;align-items:start;flex-wrap:wrap;">
    <div>
      <h2>Signal trace: window {preview_index}</h2>
      <p class="muted" style="margin-top:6px;">Raw, filtered and normalised PPG used for this window.</p>
    </div>
    <div class="chips">
      <span class="chip"><span class="dot" style="background:#63717a;"></span>raw</span>
      <span class="chip"><span class="dot" style="background:#2d6b9f;"></span>filtered</span>
      <span class="chip"><span class="dot" style="background:#b86b00;"></span>beats</span>
    </div>
  </div>
  <div style="margin-top:12px;">{processing_svg(raw_segment, processed, result['threshold'], preview_prob)}</div>
  <div class="grid" style="margin-top:12px;">
    <div class="metric"><span>SQI status</span><strong>{html.escape(accepted_text)}</strong></div>
    <div class="metric"><span>AF probability</span><strong>{html.escape(probability_text)}</strong></div>
    <div class="metric"><span>Detected beats</span><strong>{len(processed.peaks)}</strong></div>
    <div class="metric"><span>Heart rate (bpm)</span><strong>{format_float(row.get('estimated_hr_bpm'), 1)}</strong></div>
  </div>
  <p class="method"><b>SQI:</b> quality {format_float(row.get('quality_score'))}, template correlation {format_float(row.get('template_correlation'))}, heart-band energy {format_float(row.get('heart_band_energy_ratio'))}. {html.escape(reason)}.</p>
</section>"""


def results_view(result: dict[str, Any]) -> str:
    summary_df: pd.DataFrame = result["summary"].copy()
    probabilities = np.asarray(result["probabilities"], dtype=np.float32)
    all_probabilities = np.asarray(result["all_probabilities"], dtype=np.float32)
    threshold = float(result["threshold"])
    accepted_count = int(summary_df["accepted"].sum())
    total_count = int(summary_df.shape[0])
    record_prob = result["record_probability"]
    sqi_enabled = bool(result["sqi_enabled"])
    if np.isfinite(record_prob):
        badge_class = "af" if record_prob >= threshold else "sr"
        prob_text = f"{record_prob:.3f}"
    else:
        badge_class = "warn"
        prob_text = "N/A"

    if sqi_enabled and accepted_count:
        preview_index = int(np.flatnonzero(summary_df["accepted"].to_numpy(dtype=bool))[0])
    else:
        preview_index = int(np.nanargmax(all_probabilities)) if np.isfinite(all_probabilities).any() else 0
    preview_prob = None if not np.isfinite(probabilities[preview_index]) else float(probabilities[preview_index])
    trace = processing_details(result, preview_index, preview_prob)

    rows = []
    for idx, row in summary_df.iterrows():
        prob = probabilities[idx]
        prob_cell = "" if not np.isfinite(prob) else f"{float(prob):.3f}"
        accepted = bool(row["accepted"])
        status = "accepted" if accepted else "rejected"
        rows.append(
            "<tr>"
            f"<td>{int(row['segment_index'])}</td>"
            f"<td>{float(row['start_time_sec']):.1f}-{float(row['end_time_sec']):.1f}</td>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{float(row['quality_score']):.3f}</td>"
            f"<td>{float(row['template_correlation']):.3f}</td>"
            f"<td>{float(row['heart_band_energy_ratio']):.3f}</td>"
            f"<td>{float(row['estimated_hr_bpm']):.1f}</td>"
            f"<td>{prob_cell}</td>"
            f"<td>{html.escape(str(row.get('rejection_reason', '')))}</td>"
            "</tr>"
        )

    return f"""
<section class="panel">
  <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap;">
    <h2>Record result</h2>
    <span class="status {badge_class}">{html.escape(result['decision'])}</span>
  </div>
  <div class="grid" style="margin-top:14px;">
    <div class="metric"><span>Sampling rate</span><strong>{result['sample_rate']:.1f} Hz</strong></div>
    <div class="metric"><span>Record AF probability</span><strong>{prob_text}</strong></div>
    <div class="metric"><span>Decision threshold</span><strong>{threshold:.3f}</strong></div>
    <div class="metric"><span>Quality accepted</span><strong>{accepted_count}/{total_count}</strong></div>
  </div>
</section>
{trace}
<section class="panel">
  <h2>Window results</h2>
  <p class="muted" style="margin-top:6px;">{result['input_samples']} input samples; {result['resampled_samples']} samples after resampling.</p>
  <div class="table-wrap" style="margin-top:12px;">
    <table>
      <thead><tr><th>Segment</th><th>Time sec</th><th>Status</th><th>Quality</th><th>Template</th><th>Heart band</th><th>HR bpm</th><th>AF probability</th><th>Reason</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</section>
<section class="panel">
  <a href="/" style="color:var(--accent);font-weight:700;">Analyse another file</a>
</section>"""


class DemoHandler(BaseHTTPRequestHandler):
    model_path = DEFAULT_MODEL_PATH

    def do_GET(self) -> None:
        self.respond(page("PPG AF classification", upload_form()))

    def do_POST(self) -> None:
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                },
            )
            file_item = form["csv_file"] if "csv_file" in form else None
            if file_item is None or not getattr(file_item, "file", None):
                raise ValueError("Please choose a CSV file.")
            data = file_item.file.read()
            result = predict_csv(
                data,
                "PPG",
                None,
                self.model_path,
                sqi_mode="on",
            )
            self.respond(page("PPG AF classification", results_view(result)))
        except Exception as exc:
            self.respond(page("PPG AF classification", upload_form(str(exc))), status=400)

    def respond(self, content: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local upload demo for PPG AF screening.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    DemoHandler.model_path = args.model_path
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Serving PPG AF classifier at http://{args.host}:{args.port}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
