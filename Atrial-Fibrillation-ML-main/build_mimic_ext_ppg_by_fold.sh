#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/vol/bitbucket/mc1920/fypenv/bin/python}"
REPO_ROOT="${REPO_ROOT:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
DATASET_ROOT="${DATASET_ROOT:-/vol/bitbucket/mc1920/FYP/1.1.0}"
COHORT_CSV="${COHORT_CSV:-$REPO_ROOT/artifacts/mimic_ext_ppg_p00_cohort/af_sr_cohort.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/vol/bitbucket/mc1920/mimic_ext_ppg_by_fold}"
SUBSET_PREFIX="${SUBSET_PREFIX:-p00/}"
SIGNAL_TYPE="${SIGNAL_TYPE:-ppg}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"

mkdir -p "$OUTPUT_ROOT"

for fold in $FOLDS; do
  out_dir="$OUTPUT_ROOT/fold_${fold}"
  accepted_segments_path="$out_dir/ppg/ppg_accepted_segments.npz"
  accepted_summary_path="$out_dir/ppg/ppg_accepted_segment_summary.csv"
  if [[ -s "$accepted_segments_path" && -s "$accepted_summary_path" ]]; then
    echo "[build] skip subset=${SUBSET_PREFIX} fold=${fold} existing_output=${out_dir}"
    continue
  fi
  echo "[build] subset=${SUBSET_PREFIX} fold=${fold} output=${out_dir}"
  "$PYTHON_BIN" "$REPO_ROOT/build_mimic_ext_ppg_dataset.py" \
    --dataset-root "$DATASET_ROOT" \
    --cohort-csv "$COHORT_CSV" \
    --output-dir "$out_dir" \
    --subset-prefix "$SUBSET_PREFIX" \
    --strat-folds "$fold" \
    --signal-type "$SIGNAL_TYPE"
done
