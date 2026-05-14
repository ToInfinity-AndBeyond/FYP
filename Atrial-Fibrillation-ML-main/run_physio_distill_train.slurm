#!/bin/bash
#SBATCH --job-name=physio_distill_train
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
VENV_PATH="${VENV_PATH:-/vol/bitbucket/${USER}/myvenv}"
STARTER_VENV="${STARTER_VENV:-/vol/bitbucket/starter}"
SEGMENTS_PATH="${SEGMENTS_PATH:-${PROJECT_DIR}/artifacts/physio_distill/physio_multimodal_accepted_segments.npz}"
SUMMARY_PATH="${SUMMARY_PATH:-${PROJECT_DIR}/artifacts/physio_distill/physio_multimodal_accepted_segment_summary.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/artifacts/experiments/physio_distill_mimic_rerun}"

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
  if python -c "import numpy, pandas, sklearn, torch" >/dev/null 2>&1; then
    echo "Using user venv at ${VENV_PATH}"
  elif [ -f "${STARTER_VENV}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${STARTER_VENV}/bin/activate"
    echo "User venv missing key packages; using fallback starter venv at ${STARTER_VENV}"
  else
    echo "ERROR: user venv is missing required packages and starter venv is unavailable."
    exit 1
  fi
elif [ -f "${STARTER_VENV}/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "${STARTER_VENV}/bin/activate"
  echo "Using fallback starter venv at ${STARTER_VENV}"
else
  echo "ERROR: no usable virtual environment found."
  exit 1
fi

python --version

if [ ! -f "${SEGMENTS_PATH}" ]; then
  echo "ERROR: missing segments NPZ at ${SEGMENTS_PATH}"
  exit 1
fi

if [ ! -f "${SUMMARY_PATH}" ]; then
  echo "ERROR: missing summary CSV at ${SUMMARY_PATH}"
  exit 1
fi

PYTHONPYCACHEPREFIX=/tmp/pycache \
python train_physio_distill.py \
  --segments-path "${SEGMENTS_PATH}" \
  --summary-path "${SUMMARY_PATH}" \
  --epochs 20 \
  --patience 6 \
  --batch-size 32 \
  --val-records 5 \
  --test-records 5 \
  --output-dir "${OUTPUT_DIR}"
