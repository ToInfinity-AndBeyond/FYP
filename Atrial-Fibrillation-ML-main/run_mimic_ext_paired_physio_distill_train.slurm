#!/bin/bash
#SBATCH --job-name=mimic_paired_distill
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/mimic_paired_distill_%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
VENV_PATH="${VENV_PATH:-/vol/bitbucket/mc1920/fypenv}"
DATA_BASE="${DATA_BASE:-/vol/bitbucket/mc1920/mimic_ext_paired_physio_from_ppg_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/vol/bitbucket/mc1920/mimic_ext_paired_physio_distill_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-3e-4}"
REQUIRE_FOLD_COUNT="${REQUIRE_FOLD_COUNT:-30}"

mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_DIR}"

echo "=== MIMIC p00-p02 paired ECG-supervised distillation ==="
hostname
date
echo "data_base=${DATA_BASE}"
echo "output_root=${OUTPUT_ROOT}"

if [ -f "${VENV_PATH}/bin/activate" ] && [ -x "${VENV_PATH}/bin/python" ]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
  echo "Using venv at ${VENV_PATH}"
else
  echo "ERROR: missing venv at ${VENV_PATH}"
  exit 1
fi

mapfile -t SEGMENT_PATHS < <(find "${DATA_BASE}" -path '*/physio_distill/physio_multimodal_accepted_segments.npz' | sort)
mapfile -t SUMMARY_PATHS < <(find "${DATA_BASE}" -path '*/physio_distill/physio_multimodal_accepted_segment_summary.csv' | sort)

echo "segment files=${#SEGMENT_PATHS[@]}"
echo "summary files=${#SUMMARY_PATHS[@]}"
if [ "${#SEGMENT_PATHS[@]}" -lt "${REQUIRE_FOLD_COUNT}" ] || [ "${#SUMMARY_PATHS[@]}" -lt "${REQUIRE_FOLD_COUNT}" ]; then
  echo "ERROR: expected at least ${REQUIRE_FOLD_COUNT} paired fold files before training."
  exit 1
fi

PYTHONPYCACHEPREFIX=/tmp/pycache \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python train_physio_distill.py \
  --segments-path "${SEGMENT_PATHS[@]}" \
  --summary-path "${SUMMARY_PATHS[@]}" \
  --output-dir "${OUTPUT_ROOT}" \
  --split-mode metadata_folds \
  --train-folds 0,1,2,3,4,5,6,7 \
  --val-folds 8 \
  --test-folds 9 \
  --split-group-column record_id \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay 1e-4 \
  --balanced-sampler \
  --threshold-objective f1
