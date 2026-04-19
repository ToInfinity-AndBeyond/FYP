#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/vol/bitbucket/mc1920/fypenv/bin/python}"
REPO_ROOT="${REPO_ROOT:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
DATASET_ROOT="${DATASET_ROOT:-/vol/bitbucket/mc1920/FYP/1.1.0}"
COHORT_CSV="${COHORT_CSV:-$REPO_ROOT/artifacts/mimic_ext_ppg_p00_cohort/af_sr_cohort.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/vol/bitbucket/mc1920/mimic_ext_ppg_p00_by_fold}"
SIGNAL_TYPE="${SIGNAL_TYPE:-ppg}"

mkdir -p "$OUTPUT_ROOT"

for fold in 0 1 2 3 4 5 6 7 8 9; do
  out_dir="$OUTPUT_ROOT/fold_${fold}"
  echo "[build] fold=${fold} output=${out_dir}"
  "$PYTHON_BIN" "$REPO_ROOT/build_mimic_ext_ppg_dataset.py" \
    --dataset-root "$DATASET_ROOT" \
    --cohort-csv "$COHORT_CSV" \
    --output-dir "$out_dir" \
    --subset-prefix "p00/" \
    --strat-folds "$fold" \
    --signal-type "$SIGNAL_TYPE"
done
