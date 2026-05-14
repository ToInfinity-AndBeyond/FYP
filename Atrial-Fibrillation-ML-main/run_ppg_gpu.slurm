#!/bin/bash
#SBATCH --job-name=ppg_gpu_sanity
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
VENV_PATH="${VENV_PATH:-/vol/bitbucket/${USER}/myvenv}"
STARTER_VENV="${STARTER_VENV:-/vol/bitbucket/starter}"

mkdir -p "${PROJECT_DIR}/slurm_logs"
cd "${PROJECT_DIR}"

echo "=== Host/GPU Info ==="
hostname
date
/usr/bin/nvidia-smi

if [ -f "${VENV_PATH}/bin/activate" ] && [ -x "${VENV_PATH}/bin/python" ]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
  if python -c "import numpy, pandas" >/dev/null 2>&1; then
    echo "Using user venv at ${VENV_PATH}"
  elif [ -f "${STARTER_VENV}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${STARTER_VENV}/bin/activate"
    echo "User venv missing key packages; using fallback starter venv at ${STARTER_VENV}"
  else
    echo "WARNING: user venv is missing packages and starter venv not found."
    echo "Proceeding with current python; imports may fail."
  fi
elif [ -f "${STARTER_VENV}/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "${STARTER_VENV}/bin/activate"
  echo "Using fallback starter venv at ${STARTER_VENV}"
else
  echo "WARNING: venv not found at ${VENV_PATH}"
  echo "WARNING: starter venv not found at ${STARTER_VENV}"
  echo "Proceeding with system python; imports may fail."
fi

python --version
python sanity_check.py --skip-data
