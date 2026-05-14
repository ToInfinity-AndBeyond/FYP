#!/bin/bash
#SBATCH --job-name=mimic_ppg_ssl
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
VENV_PATH="${VENV_PATH:-/vol/bitbucket/mc1920/fypenv}"
OUTPUT_DIR="${OUTPUT_DIR:-/vol/bitbucket/mc1920/mimic_ppg_ssl_simclr_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOTS=(
  "${DATA_ROOT_P00:-/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p00_by_fold}"
  "${DATA_ROOT_P01:-/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p01_by_fold}"
  "${DATA_ROOT_P02:-/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p02_by_fold}"
)

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

hostname
date
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

if [ -f "${VENV_PATH}/bin/activate" ] && [ -x "${VENV_PATH}/bin/python" ]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
else
  echo "ERROR: missing venv at ${VENV_PATH}"
  exit 1
fi

SEGMENT_ARGS=()
SUMMARY_ARGS=()
for data_root in "${DATA_ROOTS[@]}"; do
  for fold in 0 1 2 3 4 5 6 7 8 9; do
    SEGMENT_ARGS+=("${data_root}/fold_${fold}/ppg/ppg_accepted_segments.npz")
    SUMMARY_ARGS+=("${data_root}/fold_${fold}/ppg/ppg_accepted_segment_summary.csv")
  done
done

PYTHONPYCACHEPREFIX=/tmp/pycache \
python train_ppg_ssl_simclr.py \
  --segments-path "${SEGMENT_ARGS[@]}" \
  --summary-path "${SUMMARY_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --train-folds 0,1,2,3,4,5,6,7 \
  --val-folds 8 \
  --epochs "${EPOCHS:-30}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --lr "${LR:-1e-3}" \
  --weight-decay "${WEIGHT_DECAY:-1e-6}" \
  --temperature "${TEMPERATURE:-0.10}" \
  --patience "${PATIENCE:-8}" \
  --disable-amp
