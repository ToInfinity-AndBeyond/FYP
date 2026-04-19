#!/bin/bash
#SBATCH --job-name=mimic_ext_p00_ppg_v2
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=slurm_logs/mimic_ext_p00_ppg_v2_%j.out

set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
VENV_PATH="/vol/bitbucket/mc1920/fypenv"
OUTPUT_DIR="/vol/bitbucket/mc1920/mimic_ext_p00_ppg_experiment_v2"
DATA_ROOT="/vol/bitbucket/mc1920/mimic_ext_ppg_p00_by_fold"

mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

echo "=== Host/GPU Info ==="
hostname
date
/usr/bin/nvidia-smi

if [ -f "${VENV_PATH}/bin/activate" ] && [ -x "${VENV_PATH}/bin/python" ]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
  echo "Using venv at ${VENV_PATH}"
else
  echo "ERROR: missing venv at ${VENV_PATH}"
  exit 1
fi

python --version

SEGMENT_ARGS=()
SUMMARY_ARGS=()
for fold in 0 1 2 3 4 5 6 7 8 9; do
  segments_path="${DATA_ROOT}/fold_${fold}/ppg/ppg_accepted_segments.npz"
  summary_path="${DATA_ROOT}/fold_${fold}/ppg/ppg_accepted_segment_summary.csv"
  if [ ! -f "${segments_path}" ]; then
    echo "ERROR: missing segments NPZ at ${segments_path}"
    exit 1
  fi
  if [ ! -f "${summary_path}" ]; then
    echo "ERROR: missing summary CSV at ${summary_path}"
    exit 1
  fi
  SEGMENT_ARGS+=("${segments_path}")
  SUMMARY_ARGS+=("${summary_path}")
done

PYTHONPYCACHEPREFIX=/tmp/pycache \
python train_ppg_hybrid.py \
  --segments-path "${SEGMENT_ARGS[@]}" \
  --summary-path "${SUMMARY_ARGS[@]}" \
  --split-mode metadata_folds \
  --train-folds 0,1,2,3,4,5,6,7 \
  --val-folds 8 \
  --test-folds 9 \
  --threshold-objective f1 \
  --balanced-sampler \
  --disable-amp \
  --mixup-alpha 0.0 \
  --epochs 25 \
  --patience 8 \
  --lr 5e-4 \
  --batch-size 32 \
  --output-dir "${OUTPUT_DIR}"
