#!/bin/bash
#SBATCH --job-name=build_mimic_ext_sqi_v2_workers
#SBATCH --partition=a16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=slurm_logs/build_mimic_ext_sqi_v2_workers_%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
VENV_PATH="${VENV_PATH:-/vol/bitbucket/mc1920/fypenv}"
DATASET_ROOT="${DATASET_ROOT:-/vol/bitbucket/mc1920/FYP/1.1.0}"
OUTPUT_BASE="${OUTPUT_BASE:-/vol/bitbucket/mc1920}"
WORKERS="${WORKERS:-4}"
SUBMIT_TRAIN_ON_SUCCESS="${SUBMIT_TRAIN_ON_SUCCESS:-0}"
TRAIN_SLURM="${TRAIN_SLURM:-${PROJECT_DIR}/run_mimic_ext_p00_p01_p02_sqi_v2_ppg_train.slurm}"

mkdir -p "${PROJECT_DIR}/slurm_logs"
cd "${PROJECT_DIR}"

echo "=== SQI v2 worker preprocess ==="
hostname
date
echo "workers=${WORKERS}"

if [ -f "${VENV_PATH}/bin/activate" ] && [ -x "${VENV_PATH}/bin/python" ]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
  echo "Using venv at ${VENV_PATH}"
else
  echo "ERROR: missing venv at ${VENV_PATH}"
  exit 1
fi

python --version

TASK_FILE="$(mktemp /tmp/mimic_ext_sqi_v2_tasks.XXXXXX)"
TASK_LOCK="${TASK_FILE}.lock"
LOCK_ROOT="${OUTPUT_BASE}/mimic_ext_ppg_sqi_v2_locks"
mkdir -p "${LOCK_ROOT}"
trap 'rm -f "${TASK_FILE}" "${TASK_LOCK}"' EXIT

for PREFIX in p00 p01 p02; do
  for FOLD in 0 1 2 3 4 5 6 7 8 9; do
    OUT_DIR="${OUTPUT_BASE}/mimic_ext_ppg_sqi_v2_${PREFIX}_by_fold/fold_${FOLD}"
    if [ -s "${OUT_DIR}/ppg/ppg_accepted_segments.npz" ] && [ -s "${OUT_DIR}/ppg/ppg_accepted_segment_summary.csv" ]; then
      echo "[skip-done] prefix=${PREFIX} fold=${FOLD}"
      continue
    fi
    printf '%s %s\n' "${PREFIX}" "${FOLD}" >> "${TASK_FILE}"
  done
done

run_task() {
  local prefix="$1"
  local fold="$2"
  local cohort_csv="${PROJECT_DIR}/artifacts/mimic_ext_ppg_${prefix}_cohort/af_sr_cohort.csv"
  local output_root="${OUTPUT_BASE}/mimic_ext_ppg_sqi_v2_${prefix}_by_fold"
  local out_dir="${output_root}/fold_${fold}"
  local lock_dir="${LOCK_ROOT}/${prefix}_fold_${fold}.lock"

  if ! mkdir "${lock_dir}" 2>/dev/null; then
    echo "[skip-locked] prefix=${prefix} fold=${fold}"
    return 0
  fi
  trap 'rm -rf "${lock_dir:-}"' RETURN

  if [ -s "${out_dir}/ppg/ppg_accepted_segments.npz" ] && [ -s "${out_dir}/ppg/ppg_accepted_segment_summary.csv" ]; then
    echo "[skip-done] prefix=${prefix} fold=${fold}"
    rm -rf "${lock_dir}"
    trap - RETURN
    return 0
  fi

  mkdir -p "${out_dir}"
  echo "=== build prefix=${prefix} fold=${fold} ==="
  date

  PYTHONPYCACHEPREFIX=/tmp/pycache \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  python build_mimic_ext_ppg_dataset.py \
    --dataset-root "${DATASET_ROOT}" \
    --cohort-csv "${cohort_csv}" \
    --output-dir "${out_dir}" \
    --subset-prefix "${prefix}/" \
    --strat-folds "${fold}" \
    --signal-type ppg

  OUT_DIR="${out_dir}" python - <<'PY'
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
PY
  rm -rf "${lock_dir}"
  trap - RETURN
}

export -f run_task
export PROJECT_DIR DATASET_ROOT OUTPUT_BASE LOCK_ROOT

worker_loop() {
  local worker_id="$1"
  local line prefix fold
  while true; do
    line="$(flock "${TASK_LOCK}" python - "${TASK_FILE}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
try:
    with path.open("r+") as handle:
        lines = handle.readlines()
        if not lines:
            raise SystemExit(1)
        first, rest = lines[0], lines[1:]
        handle.seek(0)
        handle.truncate()
        handle.writelines(rest)
        print(first.strip())
except FileNotFoundError:
    raise SystemExit(1)
PY
)" || break
    prefix="${line%% *}"
    fold="${line##* }"
    echo "[worker-${worker_id}] claimed prefix=${prefix} fold=${fold}"
    run_task "${prefix}" "${fold}"
  done
  echo "[worker-${worker_id}] done"
}

for worker_id in $(seq 1 "${WORKERS}"); do
  worker_loop "${worker_id}" &
done

wait

echo "=== SQI v2 worker preprocess complete ==="
date

if [ "${SUBMIT_TRAIN_ON_SUCCESS}" = "1" ]; then
  echo "=== submitting dependent train job ==="
  sbatch "${TRAIN_SLURM}"
fi
