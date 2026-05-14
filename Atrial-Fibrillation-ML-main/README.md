# Atrial-Fibrillation-ML

This repository is now organized around a single supported workflow for
PPG-based atrial fibrillation detection using the MIMIC PERform AF dataset,
with initial support for inspecting the Zenodo long-term ECG/PPG AF dataset.

## Supported pipeline

1. Build filtered and quality-gated segment datasets
   - `build_signal_dataset.py`
   - `signal_pipeline.py`
2. Build synchronized physiology-aware multimodal datasets
   - `build_physio_multimodal_dataset.py`
   - `signal_pipeline.py`
3. Export and use per-window label templates
   - `export_window_label_template.py`
4. Inspect Zenodo long-term ECG/PPG subjects and export ECG-derived window labels
   - `inspect_zenodo_subject.py`
   - `zenodo_longterm_loader.py`
5. Build Zenodo datasets in the same bundle format as the MIMIC pipeline
   - `build_zenodo_datasets.py`
5. Visualize preprocessing and ECG/PPG alignment
   - `visualize_filter_effect.py`
   - `compare_ecg_ppg.py`
6. Train the PPG-only hybrid AF classifier
   - `ppg_hybrid_model.py`
   - `train_ppg_hybrid.py`
7. Train the multimodal teacher / PPG-only student distillation model
   - `physio_distill_model.py`
   - `train_physio_distill.py`

## Core files

- `signal_pipeline.py`
  - band-pass filtering
  - normalization
  - peak detection
  - SQI gating
  - HRV feature extraction
  - dataset export to CSV and NPZ
- `build_signal_dataset.py`
  - processes the MIMIC PERform CSV files and writes artifacts under `artifacts/signal_pipeline`
- `visualize_filter_effect.py`
  - saves raw vs filtered comparison figures
- `compare_ecg_ppg.py`
  - compares simultaneous PPG and ECG timing on the same segment
- `ppg_hybrid_model.py`
  - hybrid deep model for AF classification from PPG
- `train_ppg_hybrid.py`
  - patient-wise training, augmentation, mixup, focal loss, TTA, checkpointing
- `build_physio_multimodal_dataset.py`
  - builds synchronized PPG/ECG/resp 30-second segments with timing and respiration features
- `export_window_label_template.py`
  - exports one row per 30-second window, including original MIMIC IDs from the WFDB headers so ECG-based window labels can be merged in later
- `zenodo_longterm_loader.py`
  - reads the Zenodo subject MAT files, decodes segment-level PPG object references, and aligns PPG segment time with the continuous ECG timeline
- `inspect_zenodo_subject.py`
  - summarizes one Zenodo subject and exports ECG-derived 30-second window labels for the corresponding PPG segments
- `build_zenodo_datasets.py`
  - builds Zenodo PPG and multimodal bundles using ECG beat-level AF annotations, resampled to the same segment length used by the MIMIC models
- `physio_distill_model.py`
  - physiology-aware teacher/student architecture for PPG-only inference
- `train_physio_distill.py`
  - multimodal distillation training with ECG/resp supervision available only during training

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Imperial GPU Cluster

If you are running this repo on the Department of Computing GPU cluster, use the repo-local guide in
[`docs/imperial_gpu_cluster.md`](docs/imperial_gpu_cluster.md).

The shortest sanity check after logging into a cluster head node is:

```bash
cd /homes/${USER}/FYP/Atrial-Fibrillation-ML-main
PROJECT_DIR=/homes/${USER}/FYP/Atrial-Fibrillation-ML-main \
VENV_PATH=/vol/bitbucket/${USER}/myvenv \
sbatch run_ppg_gpu.slurm
```

## Build the segment dataset

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache .venv/bin/python build_signal_dataset.py --signal-type both
```

To rebuild the dataset using ECG-reviewed or externally merged per-window labels:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache .venv/bin/python export_window_label_template.py \
  --output-csv artifacts/labeling/window_label_template.csv

PYTHONPYCACHEPREFIX=/tmp/pycache .venv/bin/python build_signal_dataset.py \
  --signal-type both \
  --window-label-csv artifacts/labeling/window_label_template.csv
```

The window-label CSV is expected to contain at least:

- `record_id`
- `segment_index`
- `label`

Optional columns such as `use_for_training`, `label_source`, `notes`, or annotation overlap metadata are preserved into the output summary CSVs.

Artifacts are written to:

- `artifacts/signal_pipeline/ppg`
- `artifacts/signal_pipeline/ecg`

## Train the PPG classifier

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache .venv/bin/python train_ppg_hybrid.py \
  --epochs 25 \
  --patience 8 \
  --output-dir artifacts/experiments/ppg_hybrid_run1
```

## Build the multimodal physiology-aware dataset

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache .venv/bin/python build_physio_multimodal_dataset.py \
  --output-dir artifacts/physio_distill
```

You can also rebuild the multimodal dataset with the same per-window label CSV:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache .venv/bin/python build_physio_multimodal_dataset.py \
  --output-dir artifacts/physio_distill \
  --window-label-csv artifacts/labeling/window_label_template.csv
```

## Train the multimodal distillation model

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache .venv/bin/python train_physio_distill.py \
  --epochs 20 \
  --patience 6 \
  --output-dir artifacts/experiments/physio_distill_run1
```

## Inspect a Zenodo long-term ECG/PPG subject

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache .venv/bin/python inspect_zenodo_subject.py \
  --ecg-mat zenodo_longterm_af/001_ECG.mat \
  --ppg-mat zenodo_longterm_af/001_PPG.mat
```

This writes:

- `artifacts/zenodo_inspect/001/001_ppg_segment_summary.csv`
- `artifacts/zenodo_inspect/001/001_window_labels.csv`

## Build Zenodo datasets

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache .venv/bin/python build_zenodo_datasets.py \
  --dataset-root zenodo_longterm_af \
  --output-root artifacts/zenodo \
  --mode both
```

For a quick smoke run on one subject:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache .venv/bin/python build_zenodo_datasets.py \
  --dataset-root zenodo_longterm_af \
  --output-root artifacts/zenodo_smoke \
  --subject-ids 001 \
  --mode both \
  --max-windows-per-subject 200
```

## Train with multiple datasets

Both training scripts now accept multiple summary / NPZ pairs, so you can combine MIMIC and Zenodo directly.

PPG classifier:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache .venv/bin/python train_ppg_hybrid.py \
  --segments-path artifacts/signal_pipeline/ppg/ppg_accepted_segments.npz artifacts/zenodo/signal_pipeline/ppg/ppg_accepted_segments.npz \
  --summary-path artifacts/signal_pipeline/ppg/ppg_accepted_segment_summary.csv artifacts/zenodo/signal_pipeline/ppg/ppg_accepted_segment_summary.csv
```

Multimodal distillation:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache .venv/bin/python train_physio_distill.py \
  --segments-path artifacts/physio_distill/physio_multimodal_accepted_segments.npz artifacts/zenodo/physio_distill/physio_multimodal_accepted_segments.npz \
  --summary-path artifacts/physio_distill/physio_multimodal_accepted_segment_summary.csv artifacts/zenodo/physio_distill/physio_multimodal_accepted_segment_summary.csv
```

## Dataset notes

- The MIMIC PERform records used here contain simultaneous `PPG`, `ECG`, and `resp`.
- The ECG in this dataset is single-lead `lead II`, not 12-lead ECG.
- ECG is used as a synchronized reference signal, while the classifier itself uses PPG as input.
- The default builders still support folder-based AF labels, but both builders now also accept a per-window label CSV for stricter ECG-based evaluation.
