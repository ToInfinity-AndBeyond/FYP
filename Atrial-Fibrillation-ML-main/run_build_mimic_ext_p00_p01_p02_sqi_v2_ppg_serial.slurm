#!/bin/bash
#SBATCH --job-name=build_mimic_ext_sqi_v2_serial
#SBATCH --partition=a16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs/build_mimic_ext_sqi_v2_serial_%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
VENV_PATH="${VENV_PATH:-/vol/bitbucket/mc1920/fypenv}"
DATASET_ROOT="${DATASET_ROOT:-/vol/bitbucket/mc1920/FYP/1.1.0}"
OUTPUT_BASE="${OUTPUT_BASE:-/vol/bitbucket/mc1920}"

mkdir -p "${PROJECT_DIR}/slurm_logs"
cd "${PROJECT_DIR}"

echo "=== SQI v2 serial preprocess ==="
hostname
date

if [ -f "${VENV_PATH}/bin/activate" ] && [ -x "${VENV_PATH}/bin/python" ]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
  echo "Using venv at ${VENV_PATH}"
else
  echo "ERROR: missing venv at ${VENV_PATH}"
  exit 1
fi

python --version

for PREFIX in p00 p01 p02; do
  COHORT_CSV="${PROJECT_DIR}/artifacts/mimic_ext_ppg_${PREFIX}_cohort/af_sr_cohort.csv"
  OUTPUT_ROOT="${OUTPUT_BASE}/mimic_ext_ppg_sqi_v2_${PREFIX}_by_fold"
  for FOLD in 0 1 2 3 4 5 6 7 8 9; do
    OUT_DIR="${OUTPUT_ROOT}/fold_${FOLD}"
    export OUT_DIR

    if [ -s "${OUT_DIR}/ppg/ppg_accepted_segments.npz" ] && [ -s "${OUT_DIR}/ppg/ppg_accepted_segment_summary.csv" ]; then
      echo "[skip] prefix=${PREFIX} fold=${FOLD} output already exists: ${OUT_DIR}"
      continue
    fi

    mkdir -p "${OUT_DIR}"
    echo "=== build prefix=${PREFIX} fold=${FOLD} ==="
    date

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
summary = pd.read_csv(out_dir / "ppg_segment_summary.csv")
accepted = pd.read_csv(out_dir / "ppg_accepted_segment_summary.csv")
print("=== SQI v2 fold summary ===", flush=True)
print({
    "rows": int(summary.shape[0]),
    "accepted": int(accepted.shape[0]),
    "rejected": int(summary.shape[0] - accepted.shape[0]),
    "accepted_fraction": float(accepted.shape[0] / max(summary.shape[0], 1)),
}, flush=True)
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
  done
done

echo "=== SQI v2 serial preprocess complete ==="
date
