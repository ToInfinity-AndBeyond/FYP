#!/bin/bash
#SBATCH --job-name=build_mimic_paired
#SBATCH --partition=a16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --array=0-29%6
#SBATCH --output=slurm_logs/build_mimic_paired_%A_%a.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
VENV_PATH="${VENV_PATH:-/vol/bitbucket/mc1920/fypenv}"
DATASET_ROOT="${DATASET_ROOT:-/vol/bitbucket/mc1920/FYP/1.1.0}"
OUTPUT_BASE="${OUTPUT_BASE:-/vol/bitbucket/mc1920/mimic_ext_paired_physio_from_ppg_cache}"
PPG_CACHE_BASE="${PPG_CACHE_BASE:-/vol/bitbucket/mc1920}"
REQUIRE_ECG_ACCEPTED="${REQUIRE_ECG_ACCEPTED:-0}"

PREFIXES=(p00 p01 p02)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PREFIX_INDEX=$((TASK_ID / 10))
FOLD=$((TASK_ID % 10))
PREFIX="${PREFIXES[$PREFIX_INDEX]}"

PPG_ROOT="${PPG_CACHE_BASE}/mimic_ext_ppg_sqi_v2_${PREFIX}_by_fold/fold_${FOLD}/ppg"
OUT_DIR="${OUTPUT_BASE}/${PREFIX}/fold_${FOLD}/physio_distill"

mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${OUT_DIR}"
cd "${PROJECT_DIR}"

echo "=== Build paired MIMIC PPG/ECG cache ==="
hostname
date
echo "prefix=${PREFIX} fold=${FOLD}"
echo "ppg_root=${PPG_ROOT}"
echo "output=${OUT_DIR}"

if [ -f "${VENV_PATH}/bin/activate" ] && [ -x "${VENV_PATH}/bin/python" ]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
else
  echo "ERROR: missing venv at ${VENV_PATH}"
  exit 1
fi

ARGS=()
if [ "${REQUIRE_ECG_ACCEPTED}" = "1" ]; then
  ARGS+=(--require-ecg-accepted)
fi

PYTHONPYCACHEPREFIX=/tmp/pycache \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python build_mimic_ext_paired_physio_from_ppg_cache.py \
  --dataset-root "${DATASET_ROOT}" \
  --ppg-summary "${PPG_ROOT}/ppg_accepted_segment_summary.csv" \
  --ppg-segments "${PPG_ROOT}/ppg_accepted_segments.npz" \
  --output-dir "${OUT_DIR}" \
  "${ARGS[@]}"
