#!/bin/bash
#SBATCH --job-name=build_mimic_ext_sqi_v2
#SBATCH --partition=a16
#SBATCH --array=0-29
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=slurm_logs/build_mimic_ext_sqi_v2_%A_%a.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
VENV_PATH="${VENV_PATH:-/vol/bitbucket/mc1920/fypenv}"
DATASET_ROOT="${DATASET_ROOT:-/vol/bitbucket/mc1920/FYP/1.1.0}"
OUTPUT_BASE="${OUTPUT_BASE:-/vol/bitbucket/mc1920}"

PREFIXES=(p00 p01 p02)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PREFIX_INDEX=$((TASK_ID / 10))
FOLD=$((TASK_ID % 10))
PREFIX="${PREFIXES[$PREFIX_INDEX]}"

COHORT_CSV="${PROJECT_DIR}/artifacts/mimic_ext_ppg_${PREFIX}_cohort/af_sr_cohort.csv"
OUTPUT_ROOT="${OUTPUT_BASE}/mimic_ext_ppg_sqi_v2_${PREFIX}_by_fold"
OUT_DIR="${OUTPUT_ROOT}/fold_${FOLD}"
export OUT_DIR

mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${OUT_DIR}"
cd "${PROJECT_DIR}"

echo "=== SQI v2 preprocess ==="
hostname
date
echo "prefix=${PREFIX} fold=${FOLD}"
echo "cohort=${COHORT_CSV}"
echo "output=${OUT_DIR}"

if [ -f "${VENV_PATH}/bin/activate" ] && [ -x "${VENV_PATH}/bin/python" ]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
  echo "Using venv at ${VENV_PATH}"
else
  echo "ERROR: missing venv at ${VENV_PATH}"
  exit 1
fi

python --version

PYTHONPYCACHEPREFIX=/tmp/pycache \
python build_mimic_ext_ppg_dataset.py \
  --dataset-root "${DATASET_ROOT}" \
  --cohort-csv "${COHORT_CSV}" \
  --output-dir "${OUT_DIR}" \
  --subset-prefix "${PREFIX}/" \
  --strat-folds "${FOLD}" \
  --signal-type ppg

python - <<'PY'
from pathlib import Path
import pandas as pd
import os

out_dir = Path(os.environ["OUT_DIR"]) / "ppg"
summary_path = out_dir / "ppg_segment_summary.csv"
accepted_path = out_dir / "ppg_accepted_segment_summary.csv"

summary = pd.read_csv(summary_path)
accepted = pd.read_csv(accepted_path)
print("=== SQI v2 summary ===", flush=True)
print({
    "rows": int(summary.shape[0]),
    "accepted": int(accepted.shape[0]),
    "rejected": int(summary.shape[0] - accepted.shape[0]),
    "accepted_fraction": float(accepted.shape[0] / max(summary.shape[0], 1)),
}, flush=True)
if "label" in summary.columns:
    print("label counts all:", summary["label"].value_counts().sort_index().to_dict(), flush=True)
    print("label counts accepted:", accepted["label"].value_counts().sort_index().to_dict(), flush=True)
rejected = summary.loc[~summary["accepted"].astype(bool)]
if not rejected.empty and "rejection_reason" in rejected.columns:
    reasons = {}
    for value in rejected["rejection_reason"].fillna(""):
        for reason in str(value).split(";"):
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
    print("rejection reasons:", dict(sorted(reasons.items(), key=lambda x: (-x[1], x[0]))), flush=True)
PY
