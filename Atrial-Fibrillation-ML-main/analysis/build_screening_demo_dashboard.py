#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = ROOT / "artifacts" / "experiments" / "mimic_perform_af_ppg_run1"
DEFAULT_SIGNAL_DIR = ROOT / "artifacts" / "signal_pipeline_perform_af" / "ppg"
DEFAULT_EMBEDDED_LOG = ROOT / "artifacts" / "ch8_embedded_evidence" / "edgeai_runner_info_metrics.txt"
DEFAULT_OUTPUT = ROOT / "artifacts" / "demo_screening_dashboard" / "index.html"
FS = 125.0


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SQI-Aware AF Screening Demo</title>
  <style>
    :root {
      --bg: #f5f7f8;
      --panel: #ffffff;
      --ink: #172026;
      --muted: #65727c;
      --line: #d7dde1;
      --teal: #00897b;
      --amber: #c77700;
      --red: #c24135;
      --blue: #2f6f9f;
      --green-soft: #e3f3ef;
      --amber-soft: #fff2d6;
      --red-soft: #fbe4e1;
      --gray-soft: #eceff1;
      --shadow: 0 12px 30px rgba(17, 31, 44, 0.08);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    header {
      background: #101820;
      color: #ffffff;
      padding: 24px clamp(16px, 3vw, 40px);
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(280px, 430px);
      gap: 20px;
      align-items: end;
    }

    h1, h2, h3, p {
      margin: 0;
    }

    .eyebrow {
      color: #9bd5cf;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }

    h1 {
      font-size: clamp(1.65rem, 3vw, 2.75rem);
      line-height: 1.05;
      max-width: 860px;
      letter-spacing: 0;
    }

    .controls {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: end;
    }

    label {
      display: block;
      color: #c9d4da;
      font-size: 0.78rem;
      font-weight: 700;
      margin-bottom: 6px;
    }

    select, button {
      width: 100%;
      height: 42px;
      border-radius: 8px;
      border: 1px solid #52616b;
      font: inherit;
      font-weight: 700;
    }

    select {
      background: #ffffff;
      color: var(--ink);
      padding: 0 12px;
    }

    button {
      min-width: 130px;
      border: 0;
      color: #ffffff;
      background: var(--teal);
      padding: 0 16px;
      cursor: pointer;
    }

    main {
      padding: 18px clamp(14px, 2.5vw, 34px) 34px;
      display: grid;
      gap: 16px;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(150px, 1fr));
      gap: 12px;
    }

    .metric, .panel, .step, .embedded-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .metric {
      min-height: 94px;
      padding: 14px;
      display: grid;
      align-content: space-between;
      gap: 10px;
    }

    .metric span {
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
    }

    .metric strong {
      font-size: clamp(1.2rem, 2vw, 1.85rem);
      line-height: 1;
      overflow-wrap: anywhere;
    }

    .metric small {
      color: var(--muted);
      font-weight: 650;
    }

    .workflow {
      display: grid;
      grid-template-columns: repeat(5, minmax(145px, 1fr));
      gap: 10px;
    }

    .step {
      padding: 12px;
      min-height: 78px;
      display: grid;
      gap: 7px;
      border-left: 5px solid var(--line);
    }

    .step.done {
      border-left-color: var(--teal);
    }

    .step.warn {
      border-left-color: var(--amber);
    }

    .step.stop {
      border-left-color: var(--red);
    }

    .step span {
      color: var(--muted);
      font-size: 0.74rem;
      font-weight: 800;
      text-transform: uppercase;
    }

    .step strong {
      font-size: 0.95rem;
      overflow-wrap: anywhere;
    }

    .two-col {
      display: grid;
      grid-template-columns: minmax(300px, 1.35fr) minmax(280px, 0.9fr);
      gap: 16px;
    }

    .panel {
      padding: 14px;
      min-width: 0;
    }

    .panel-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 10px;
    }

    h2 {
      font-size: 1rem;
      line-height: 1.2;
      letter-spacing: 0;
    }

    .meta {
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 700;
      text-align: right;
    }

    svg {
      width: 100%;
      display: block;
      border: 1px solid #edf0f2;
      border-radius: 8px;
      background: #fbfcfc;
    }

    #waveformSvg {
      height: 330px;
    }

    #probSvg {
      height: 330px;
    }

    .table-wrap {
      overflow: auto;
      max-height: 360px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
      font-size: 0.86rem;
    }

    thead {
      position: sticky;
      top: 0;
      background: #eef3f5;
      z-index: 1;
    }

    th, td {
      padding: 9px 10px;
      border-bottom: 1px solid #e4e9ec;
      text-align: right;
      white-space: nowrap;
    }

    th:first-child, td:first-child,
    th:nth-child(2), td:nth-child(2),
    th:last-child, td:last-child {
      text-align: left;
    }

    th {
      color: #4a5963;
      font-size: 0.74rem;
      text-transform: uppercase;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 800;
      border: 1px solid transparent;
    }

    .badge.accepted {
      color: #00695c;
      background: var(--green-soft);
      border-color: #b9dfd8;
    }

    .badge.rejected {
      color: #9f2f25;
      background: var(--red-soft);
      border-color: #efc1bb;
    }

    .badge.warning {
      color: #8a5600;
      background: var(--amber-soft);
      border-color: #f1d08f;
    }

    .embedded-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }

    .embedded-card {
      min-height: 84px;
      padding: 12px;
      display: grid;
      align-content: space-between;
      box-shadow: none;
    }

    .embedded-card span {
      color: var(--muted);
      font-size: 0.74rem;
      font-weight: 800;
      text-transform: uppercase;
    }

    .embedded-card strong {
      font-size: 1.25rem;
    }

    .log {
      margin-top: 12px;
      background: #101820;
      color: #d8f3ee;
      border-radius: 8px;
      padding: 12px;
      overflow: auto;
      font: 0.82rem ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      max-height: 190px;
      white-space: pre;
    }

    footer {
      color: var(--muted);
      font-size: 0.78rem;
      line-height: 1.5;
      padding: 2px 4px 0;
    }

    @media (max-width: 1080px) {
      header, .two-col {
        grid-template-columns: 1fr;
      }

      .summary-grid {
        grid-template-columns: repeat(2, minmax(150px, 1fr));
      }

      .workflow, .embedded-grid {
        grid-template-columns: repeat(2, minmax(130px, 1fr));
      }

      .controls {
        max-width: 620px;
      }
    }

    @media (max-width: 620px) {
      .summary-grid, .workflow, .embedded-grid, .controls {
        grid-template-columns: 1fr;
      }

      button {
        min-width: 0;
      }

      #waveformSvg, #probSvg {
        height: 280px;
      }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <p class="eyebrow">Demonstration</p>
      <h1>From PPG Record to Screening Output</h1>
    </div>
    <div class="controls">
      <div>
        <label for="caseSelect">Record</label>
        <select id="caseSelect"></select>
      </div>
      <button id="runButton" type="button">Run pipeline</button>
    </div>
  </header>

  <main>
    <section class="summary-grid" aria-label="summary">
      <div class="metric">
        <span>Input record</span>
        <strong id="recordId">-</strong>
        <small id="recordLabel">-</small>
      </div>
      <div class="metric">
        <span>SQI status</span>
        <strong id="sqiStatus">-</strong>
        <small id="sqiDetail">-</small>
      </div>
      <div class="metric">
        <span>Record probability</span>
        <strong id="recordProbability">-</strong>
        <small id="thresholdText">-</small>
      </div>
      <div class="metric">
        <span>Final output</span>
        <strong id="decisionText">-</strong>
        <small id="confidenceText">-</small>
      </div>
      <div class="metric">
        <span>Segments used</span>
        <strong id="segmentCount">-</strong>
        <small id="aggregationText">-</small>
      </div>
    </section>

    <section class="workflow" aria-label="pipeline">
      <div class="step done" id="stepInput">
        <span>1. Input</span>
        <strong>Record loaded</strong>
      </div>
      <div class="step" id="stepSqi">
        <span>2. SQI</span>
        <strong>-</strong>
      </div>
      <div class="step" id="stepPeaks">
        <span>3. Peaks</span>
        <strong>-</strong>
      </div>
      <div class="step" id="stepModel">
        <span>4. Model</span>
        <strong>-</strong>
      </div>
      <div class="step" id="stepDecision">
        <span>5. Output</span>
        <strong>-</strong>
      </div>
    </section>

    <section class="two-col">
      <div class="panel">
        <div class="panel-head">
          <h2>PPG waveform preview</h2>
          <div class="meta" id="waveformMeta">-</div>
        </div>
        <svg id="waveformSvg" viewBox="0 0 900 330" preserveAspectRatio="none" role="img"></svg>
      </div>

      <div class="panel">
        <div class="panel-head">
          <h2>Segment-level AF probabilities</h2>
          <div class="meta" id="probMeta">-</div>
        </div>
        <svg id="probSvg" viewBox="0 0 620 330" preserveAspectRatio="none" role="img"></svg>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Record aggregation</h2>
        <div class="meta" id="tableMeta">-</div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Segment</th>
              <th>Status</th>
              <th>Start (s)</th>
              <th>SQI</th>
              <th>AF prob.</th>
              <th>Peaks</th>
              <th>HR bpm</th>
              <th>CV IBI</th>
              <th>Use</th>
            </tr>
          </thead>
          <tbody id="segmentTable"></tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Embedded smoke test</h2>
        <div class="meta">Compact 17-feature inference path</div>
      </div>
      <div class="embedded-grid" id="embeddedGrid"></div>
      <pre class="log" id="embeddedLog"></pre>
    </section>

    <footer id="sourceFooter"></footer>
  </main>

  <script id="demo-data" type="application/json">__DEMO_DATA__</script>
  <script>
    const demoData = JSON.parse(document.getElementById("demo-data").textContent);
    const svgNS = "http://www.w3.org/2000/svg";

    function fmt(value, digits = 3) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toFixed(digits);
    }

    function percent(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "Not evaluated";
      return Math.round(Number(value) * 1000) / 10 + "%";
    }

    function clearSvg(svg) {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
    }

    function addSvg(svg, tag, attrs) {
      const el = document.createElementNS(svgNS, tag);
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
      svg.appendChild(el);
      return el;
    }

    function addText(svg, x, y, text, attrs = {}) {
      const el = addSvg(svg, "text", {
        x,
        y,
        fill: attrs.fill || "#65727c",
        "font-size": attrs.size || "12",
        "font-weight": attrs.weight || "700",
        "text-anchor": attrs.anchor || "start"
      });
      el.textContent = text;
      return el;
    }

    function renderWaveform(record) {
      const svg = document.getElementById("waveformSvg");
      clearSvg(svg);
      const w = 900;
      const h = 330;
      const pad = { left: 48, right: 18, top: 22, bottom: 40 };
      const values = record.waveform || [];
      if (!values.length) return;
      const minY = Math.min(...values);
      const maxY = Math.max(...values);
      const span = Math.max(maxY - minY, 1e-6);
      const innerW = w - pad.left - pad.right;
      const innerH = h - pad.top - pad.bottom;
      const xFor = i => pad.left + (i / Math.max(values.length - 1, 1)) * innerW;
      const yFor = v => pad.top + (1 - ((v - minY) / span)) * innerH;
      const zeroY = yFor(Math.max(Math.min(0, maxY), minY));

      addSvg(svg, "line", { x1: pad.left, x2: w - pad.right, y1: zeroY, y2: zeroY, stroke: "#d7dde1", "stroke-width": 1 });
      for (let i = 0; i <= 6; i++) {
        const x = pad.left + (i / 6) * innerW;
        addSvg(svg, "line", { x1: x, x2: x, y1: pad.top, y2: h - pad.bottom, stroke: "#edf0f2", "stroke-width": 1 });
        addText(svg, x, h - 14, String(Math.round(i * record.durationSeconds / 6)), { anchor: "middle", size: "11" });
      }
      addText(svg, pad.left, 16, "PPG (normalised)", { size: "12", weight: "800" });
      addText(svg, w - pad.right, h - 14, "Time (s)", { anchor: "end", size: "11" });

      const step = Math.max(1, Math.ceil(values.length / 1100));
      const points = [];
      for (let i = 0; i < values.length; i += step) {
        points.push(xFor(i).toFixed(1) + "," + yFor(values[i]).toFixed(1));
      }
      addSvg(svg, "polyline", {
        points: points.join(" "),
        fill: "none",
        stroke: record.noCall ? "#c24135" : "#2f6f9f",
        "stroke-width": 2.2,
        "stroke-linejoin": "round",
        "stroke-linecap": "round"
      });

      const peakColor = record.noCall ? "#c77700" : "#00897b";
      (record.peaks || []).forEach(index => {
        if (index >= 0 && index < values.length) {
          addSvg(svg, "circle", {
            cx: xFor(index),
            cy: yFor(values[index]),
            r: 2.8,
            fill: peakColor,
            opacity: 0.9
          });
        }
      });
    }

    function renderProbabilities(record) {
      const svg = document.getElementById("probSvg");
      clearSvg(svg);
      const rows = record.segmentRows || [];
      const w = 620;
      const h = 330;
      const pad = { left: 44, right: 18, top: 24, bottom: 42 };
      const innerW = w - pad.left - pad.right;
      const innerH = h - pad.top - pad.bottom;
      const threshold = record.threshold;
      const barGap = 2;
      const barW = Math.max(4, (innerW - barGap * Math.max(rows.length - 1, 0)) / Math.max(rows.length, 1));
      const yFor = p => pad.top + (1 - Math.max(0, Math.min(1, p))) * innerH;

      for (let i = 0; i <= 4; i++) {
        const p = i / 4;
        const y = yFor(p);
        addSvg(svg, "line", { x1: pad.left, x2: w - pad.right, y1: y, y2: y, stroke: "#edf0f2", "stroke-width": 1 });
        addText(svg, 10, y + 4, fmt(p, 2), { size: "11" });
      }

      if (threshold !== null && threshold !== undefined) {
        const y = yFor(threshold);
        addSvg(svg, "line", {
          x1: pad.left,
          x2: w - pad.right,
          y1: y,
          y2: y,
          stroke: "#c77700",
          "stroke-width": 1.6,
          "stroke-dasharray": "5 5"
        });
        addText(svg, w - pad.right - 2, y - 6, "threshold " + fmt(threshold, 2), { anchor: "end", fill: "#8a5600", size: "11" });
      }

      rows.forEach((row, i) => {
        const x = pad.left + i * (barW + barGap);
        if (row.prob === null || row.prob === undefined) {
          const qualityHeight = Math.max(3, Number(row.sqi || 0) * innerH);
          addSvg(svg, "rect", {
            x,
            y: h - pad.bottom - qualityHeight,
            width: barW,
            height: qualityHeight,
            fill: "#c24135",
            opacity: 0.55
          });
          return;
        }
        const p = Math.max(0, Math.min(1, Number(row.prob)));
        const y = yFor(p);
        const fill = row.used ? (p >= threshold ? "#00897b" : "#2f6f9f") : "#c24135";
        addSvg(svg, "rect", {
          x,
          y,
          width: barW,
          height: h - pad.bottom - y,
          rx: 2,
          fill,
          opacity: row.used ? 0.88 : 0.45
        });
      });

      addText(svg, pad.left, h - 14, "Segments", { size: "11" });
      addText(svg, 10, 16, "Probability", { size: "12", weight: "800" });
    }

    function renderSegmentTable(record) {
      const tbody = document.getElementById("segmentTable");
      tbody.innerHTML = "";
      (record.segmentRows || []).forEach(row => {
        const tr = document.createElement("tr");
        const statusClass = row.status === "Accepted" ? "accepted" : "rejected";
        const prob = row.prob === null || row.prob === undefined ? "Skipped" : fmt(row.prob, 3);
        const useText = row.used ? "yes" : (row.status === "Accepted" ? "down-weight" : "no");
        tr.innerHTML = `
          <td>${row.segment}</td>
          <td><span class="badge ${statusClass}">${row.status}</span></td>
          <td>${fmt(row.startSeconds, 1)}</td>
          <td>${fmt(row.sqi, 3)}</td>
          <td>${prob}</td>
          <td>${row.peakCount ?? "-"}</td>
          <td>${fmt(row.hrBpm, 1)}</td>
          <td>${fmt(row.cvIbi, 3)}</td>
          <td>${useText}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    function setStep(id, state, text) {
      const el = document.getElementById(id);
      el.className = "step " + state;
      el.querySelector("strong").textContent = text;
    }

    function renderEmbedded() {
      const grid = document.getElementById("embeddedGrid");
      grid.innerHTML = "";
      demoData.embedded.metrics.forEach(item => {
        const card = document.createElement("div");
        card.className = "embedded-card";
        card.innerHTML = `<span>${item.label}</span><strong>${item.value}</strong>`;
        grid.appendChild(card);
      });
      document.getElementById("embeddedLog").textContent = demoData.embedded.logExcerpt.join("\n");
    }

    function renderCase(key) {
      const record = demoData.cases[key];
      document.getElementById("recordId").textContent = record.recordId;
      document.getElementById("recordLabel").textContent = record.caseTitle + " | reference " + record.referenceLabel;
      document.getElementById("sqiStatus").textContent = record.sqiStatus;
      document.getElementById("sqiDetail").textContent = record.sqiDetail;
      document.getElementById("recordProbability").textContent = percent(record.recordProbability);
      document.getElementById("thresholdText").textContent = record.noCall ? "classification skipped" : "threshold " + fmt(record.threshold, 3);
      document.getElementById("decisionText").textContent = record.finalDecision;
      document.getElementById("confidenceText").textContent = record.confidence;
      document.getElementById("segmentCount").textContent = record.usedSegments + " / " + record.totalSegments;
      document.getElementById("aggregationText").textContent = record.aggregationText;
      document.getElementById("waveformMeta").textContent = record.waveformMeta;
      document.getElementById("probMeta").textContent = record.probabilityMeta;
      document.getElementById("tableMeta").textContent = record.tableMeta;

      setStep("stepSqi", record.noCall ? "stop" : "done", record.sqiStatus);
      setStep("stepPeaks", record.noCall ? "warn" : "done", record.peakCount + " peaks detected");
      setStep("stepModel", record.noCall ? "stop" : "done", record.noCall ? "Skipped after SQI gate" : "Segment scores loaded");
      setStep("stepDecision", record.noCall ? "stop" : "done", record.finalDecision);

      renderWaveform(record);
      renderProbabilities(record);
      renderSegmentTable(record);
    }

    const select = document.getElementById("caseSelect");
    demoData.caseOrder.forEach(key => {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = demoData.cases[key].caseTitle;
      select.appendChild(option);
    });
    select.value = demoData.defaultCase;
    select.addEventListener("change", () => renderCase(select.value));
    document.getElementById("runButton").addEventListener("click", () => renderCase(select.value));
    document.getElementById("sourceFooter").textContent = demoData.sourceNote;

    renderEmbedded();
    renderCase(demoData.defaultCase);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML demo for the SQI-aware AF screening workflow."
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--signal-dir", type=Path, default=DEFAULT_SIGNAL_DIR)
    parser.add_argument("--embedded-log", type=Path, default=DEFAULT_EMBEDDED_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fs", type=float, default=FS)
    return parser.parse_args()


def read_threshold(metrics_path: Path) -> float:
    if not metrics_path.exists():
        return 0.5
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = metrics.get("best_val_threshold")
    return float(threshold) if threshold is not None else 0.5


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def finite_int(value: Any) -> int | None:
    number = finite_float(value)
    if number is None:
        return None
    return int(round(number))


def rounded_signal(values: np.ndarray, digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in values]


def detect_peaks(waveform: np.ndarray, fs: float) -> list[int]:
    std = float(np.std(waveform))
    peaks, _ = find_peaks(
        waveform,
        distance=max(1, int(0.35 * fs)),
        prominence=max(0.15 * std, 1e-3),
    )
    return [int(p) for p in peaks]


def confidence_label(probability: float | None, threshold: float, no_call: bool) -> str:
    if no_call or probability is None:
        return "No-call"
    if probability >= threshold:
        margin = (probability - threshold) / max(1.0 - threshold, 1e-6)
    else:
        margin = (threshold - probability) / max(threshold, 1e-6)
    if margin >= 0.55:
        return "High margin"
    if margin >= 0.25:
        return "Moderate margin"
    return "Low margin"


def load_prediction_tables(experiment_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    record_path = experiment_dir / "test_record_predictions.csv"
    segment_path = experiment_dir / "test_segment_predictions.csv"
    metrics_path = experiment_dir / "metrics.json"
    if not record_path.exists():
        raise FileNotFoundError(record_path)
    if not segment_path.exists():
        raise FileNotFoundError(segment_path)
    return pd.read_csv(record_path), pd.read_csv(segment_path), read_threshold(metrics_path)


def load_signal_tables(signal_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    accepted_summary_path = signal_dir / "ppg_accepted_segment_summary.csv"
    accepted_segments_path = signal_dir / "ppg_accepted_segments.npz"
    full_summary_path = signal_dir / "ppg_segment_summary.csv"
    full_segments_path = signal_dir / "ppg_segments.npz"
    for path in [accepted_summary_path, accepted_segments_path, full_summary_path, full_segments_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    accepted_summary = pd.read_csv(accepted_summary_path)
    full_summary = pd.read_csv(full_summary_path)
    accepted_segments = np.load(accepted_segments_path)["segments"]
    full_segments = np.load(full_segments_path)["segments"]
    return accepted_summary, full_summary, accepted_segments, full_segments


def record_segments(
    record_id: str,
    accepted_summary: pd.DataFrame,
    segment_predictions: pd.DataFrame,
) -> pd.DataFrame:
    summary = accepted_summary[accepted_summary["record_id"] == record_id].copy()
    summary = summary.reset_index().rename(columns={"index": "source_index"})
    predictions = segment_predictions[segment_predictions["record_id"] == record_id].reset_index(drop=True)
    if summary.empty:
        raise RuntimeError(f"No accepted segments found for {record_id}")
    if len(summary) != len(predictions):
        raise RuntimeError(
            f"Prediction/segment count mismatch for {record_id}: "
            f"{len(predictions)} predictions vs {len(summary)} accepted segments"
        )
    summary["prob"] = predictions["prob"].to_numpy(dtype=float)
    return summary


def choose_preview_segment(segments: pd.DataFrame, mode: str) -> pd.Series:
    sortable = segments.copy()
    if mode == "clean_sr":
        sortable["rank_cv"] = sortable["cv_ibi"].fillna(np.inf)
        return sortable.sort_values(
            ["rank_cv", "quality_score", "prob"],
            ascending=[True, False, True],
        ).iloc[0]
    if mode == "af":
        sortable["rank_cv"] = sortable["cv_ibi"].fillna(-np.inf)
        return sortable.sort_values(
            ["prob", "rank_cv", "quality_score"],
            ascending=[False, False, False],
        ).iloc[0]
    return sortable.sort_values(["quality_score"], ascending=[False]).iloc[0]


def segment_rows(segments: pd.DataFrame, threshold: float, include_prob: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, row in segments.reset_index(drop=True).iterrows():
        accepted = bool(row.get("accepted", True))
        prob = finite_float(row.get("prob")) if include_prob else None
        rows.append(
            {
                "segment": int(row.get("segment_index", offset)) + 1,
                "status": "Accepted" if accepted else "Rejected",
                "startSeconds": finite_float(row.get("start_time_sec")),
                "sqi": finite_float(row.get("quality_score")),
                "prob": prob,
                "peakCount": finite_int(row.get("peak_count")),
                "hrBpm": finite_float(row.get("estimated_hr_bpm")),
                "cvIbi": finite_float(row.get("cv_ibi")),
                "used": bool(accepted and prob is not None and finite_float(row.get("quality_score")) is not None),
                "aboveThreshold": bool(prob is not None and prob >= threshold),
            }
        )
    return rows


def quality_weighted_probability(rows: list[dict[str, Any]]) -> float | None:
    usable = [row for row in rows if row["used"] and row["prob"] is not None and row["sqi"] is not None]
    if not usable:
        return None
    weights = np.asarray([max(float(row["sqi"]), 0.0) for row in usable], dtype=float)
    probs = np.asarray([float(row["prob"]) for row in usable], dtype=float)
    if float(weights.sum()) <= 0.0:
        return float(probs.mean())
    return float(np.average(probs, weights=weights))


def make_prediction_case(
    *,
    key: str,
    title: str,
    record_id: str,
    mode: str,
    record_predictions: pd.DataFrame,
    segment_predictions: pd.DataFrame,
    accepted_summary: pd.DataFrame,
    accepted_segments: np.ndarray,
    threshold: float,
    fs: float,
) -> dict[str, Any]:
    matches = record_predictions[record_predictions["record_id"] == record_id]
    if matches.empty:
        raise RuntimeError(f"No record prediction found for {record_id}")
    record = matches.iloc[0]
    segments = record_segments(record_id, accepted_summary, segment_predictions)
    preview = choose_preview_segment(segments, mode)
    waveform = accepted_segments[int(preview["source_index"])].astype(float)
    peaks = detect_peaks(waveform, fs)
    rows = segment_rows(segments, threshold)
    weighted_prob = quality_weighted_probability(rows)
    probability = float(record["prob"])
    label = int(record["label"])
    decision = "AF-like screening evidence" if probability >= threshold else "SR-compatible"
    reference = "AF" if label == 1 else "SR"
    quality_mean = finite_float(record.get("quality_mean"))
    total_segments = len(rows)
    used_segments = sum(1 for row in rows if row["used"])
    aggregation = (
        f"quality-weighted p={weighted_prob:.3f}" if weighted_prob is not None else "no accepted segments"
    )
    peak_count = len(peaks)

    return {
        "key": key,
        "caseTitle": title,
        "recordId": record_id,
        "referenceLabel": reference,
        "sqiStatus": "Accepted",
        "sqiDetail": f"mean SQI {quality_mean:.3f}" if quality_mean is not None else "accepted windows",
        "recordProbability": probability,
        "threshold": threshold,
        "finalDecision": decision,
        "confidence": confidence_label(probability, threshold, no_call=False),
        "usedSegments": used_segments,
        "totalSegments": total_segments,
        "aggregationText": aggregation,
        "durationSeconds": round(len(waveform) / fs, 3),
        "waveformMeta": f"segment {int(preview['segment_index']) + 1}, SQI {float(preview['quality_score']):.3f}",
        "probabilityMeta": f"{total_segments} accepted windows",
        "tableMeta": f"record p={probability:.3f}",
        "peakCount": peak_count,
        "noCall": False,
        "waveform": rounded_signal(waveform),
        "peaks": peaks,
        "segmentRows": rows,
    }


def make_low_quality_case(
    *,
    title: str,
    full_summary: pd.DataFrame,
    full_segments: np.ndarray,
    threshold: float,
    fs: float,
) -> dict[str, Any]:
    rejected = full_summary[full_summary["accepted"].astype(bool) == False].copy()  # noqa: E712
    if rejected.empty:
        rejected = full_summary.nsmallest(1, "quality_score").copy()
    rejected = rejected.reset_index().rename(columns={"index": "source_index"})
    row = rejected.sort_values(["quality_score"], ascending=[True]).iloc[0]
    waveform = full_segments[int(row["source_index"])].astype(float)
    peaks = detect_peaks(waveform, fs)
    single = row.to_frame().T
    single["prob"] = np.nan
    rows = segment_rows(single, threshold, include_prob=False)
    reason = str(row.get("rejection_reason", "") or "quality gate")
    reference = "AF" if int(row.get("label", 0)) == 1 else "SR"
    sqi = finite_float(row.get("quality_score"))

    return {
        "key": "low_quality",
        "caseTitle": title,
        "recordId": str(row["record_id"]),
        "referenceLabel": reference,
        "sqiStatus": "Rejected",
        "sqiDetail": f"SQI {sqi:.3f}, {reason}" if sqi is not None else reason,
        "recordProbability": None,
        "threshold": threshold,
        "finalDecision": "No-call",
        "confidence": "Insufficient signal quality",
        "usedSegments": 0,
        "totalSegments": 1,
        "aggregationText": "classification not forced",
        "durationSeconds": round(len(waveform) / fs, 3),
        "waveformMeta": f"rejected segment, {reason}",
        "probabilityMeta": "SQI gate stopped inference",
        "tableMeta": "window removed before aggregation",
        "peakCount": len(peaks),
        "noCall": True,
        "waveform": rounded_signal(waveform),
        "peaks": peaks,
        "segmentRows": rows,
    }


def metric_from_log(pattern: str, text: str, default: str = "-") -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else default


def embedded_summary(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {
            "metrics": [{"label": "Status", "value": "missing log"}],
            "logExcerpt": [f"Missing embedded evidence log: {log_path}"],
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics = [
        {"label": "Input features", "value": metric_from_log(r"Unique inputs count:\s+(.+)", text)},
        {"label": "Model flash", "value": metric_from_log(r"Model flash usage:\s+(.+)", text)},
        {"label": "Samples", "value": metric_from_log(r"Processed\s+(\d+\s+samples)", text)},
        {"label": "Accuracy", "value": metric_from_log(r"ACCURACY:\s+([0-9.]+)", text)},
        {"label": "Balanced acc.", "value": metric_from_log(r"BALANCED ACCURACY:\s+([0-9.]+)", text)},
    ]
    excerpt = [
        "$ nrf_edgeai_inference_runner_linux info",
        f"Unique inputs count: {metrics[0]['value']}",
        f"Model flash usage: {metrics[1]['value']}",
        "",
        "$ nrf_edgeai_inference_runner_linux metrics vitaldb_ppg_holdout.csv -t label",
        f"Processed {metrics[2]['value']}",
        f"ACCURACY: {metrics[3]['value']}",
        f"BALANCED ACCURACY: {metrics[4]['value']}",
    ]
    return {"metrics": metrics, "logExcerpt": excerpt}


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def build_demo_data(args: argparse.Namespace) -> dict[str, Any]:
    record_predictions, segment_predictions, threshold = load_prediction_tables(args.experiment_dir)
    accepted_summary, full_summary, accepted_segments, full_segments = load_signal_tables(args.signal_dir)

    cases = {
        "clean_sr": make_prediction_case(
            key="clean_sr",
            title="Clean SR case",
            record_id="mimic_perform_non_af_003",
            mode="clean_sr",
            record_predictions=record_predictions,
            segment_predictions=segment_predictions,
            accepted_summary=accepted_summary,
            accepted_segments=accepted_segments,
            threshold=threshold,
            fs=args.fs,
        ),
        "af_case": make_prediction_case(
            key="af_case",
            title="AF case",
            record_id="mimic_perform_af_014",
            mode="af",
            record_predictions=record_predictions,
            segment_predictions=segment_predictions,
            accepted_summary=accepted_summary,
            accepted_segments=accepted_segments,
            threshold=threshold,
            fs=args.fs,
        ),
        "low_quality": make_low_quality_case(
            title="Low-quality / no-call case",
            full_summary=full_summary,
            full_segments=full_segments,
            threshold=threshold,
            fs=args.fs,
        ),
    }

    return {
        "caseOrder": ["clean_sr", "af_case", "low_quality"],
        "defaultCase": "clean_sr",
        "cases": cases,
        "embedded": embedded_summary(args.embedded_log),
        "sourceNote": (
            f"Sources: {display_path(args.experiment_dir)} predictions, "
            f"{display_path(args.signal_dir)} waveform/SQI artifacts, "
            f"{display_path(args.embedded_log)} embedded evidence."
        ),
    }


def write_dashboard(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__DEMO_DATA__", payload)
    output_path.write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    data = build_demo_data(args)
    write_dashboard(data, args.output)
    print(f"Saved {args.output}")
    print("Open the HTML file in a browser for the interactive demo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
