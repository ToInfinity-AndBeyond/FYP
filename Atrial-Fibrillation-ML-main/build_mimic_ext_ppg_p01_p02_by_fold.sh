#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
GENERIC_SCRIPT="${GENERIC_SCRIPT:-$REPO_ROOT/build_mimic_ext_ppg_by_fold.sh}"
PYTHON_BIN="${PYTHON_BIN:-/vol/bitbucket/mc1920/fypenv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/vol/bitbucket/mc1920/FYP/physionet.org/files/mimic-iii-ext-ppg/1.1.0}"
SIGNAL_TYPE="${SIGNAL_TYPE:-ppg}"

for shard in p01 p02; do
  cohort_csv="$REPO_ROOT/artifacts/mimic_ext_ppg_${shard}_cohort/af_sr_cohort.csv"
  output_root="/vol/bitbucket/mc1920/mimic_ext_ppg_${shard}_by_fold"
  echo "[cohort] shard=${shard} cohort=${cohort_csv}"
  if [[ ! -f "$cohort_csv" ]]; then
    echo "[error] missing cohort CSV for ${shard}: ${cohort_csv}" >&2
    exit 1
  fi
  echo "[start] shard=${shard} dataset_root=${DATASET_ROOT} output_root=${output_root}"
  PYTHON_BIN="$PYTHON_BIN" \
  REPO_ROOT="$REPO_ROOT" \
  DATASET_ROOT="$DATASET_ROOT" \
  COHORT_CSV="$cohort_csv" \
  OUTPUT_ROOT="$output_root" \
  SUBSET_PREFIX="${shard}/" \
  SIGNAL_TYPE="$SIGNAL_TYPE" \
  "$GENERIC_SCRIPT"
  echo "[done] shard=${shard}"
done
