#!/bin/bash
#SBATCH --job-name=prepare_full_p03_p09_manifest
#SBATCH --partition=a16
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/prepare_full_p03_p09_manifest_%j.out

set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
VENV_PATH="/vol/bitbucket/mc1920/fypenv"

mkdir -p "${PROJECT_DIR}/slurm_logs"
cd "${PROJECT_DIR}"

if [ -f "${VENV_PATH}/bin/activate" ] && [ -x "${VENV_PATH}/bin/python" ]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
else
  echo "ERROR: missing venv at ${VENV_PATH}"
  exit 1
fi

python build_mimic_ext_full_segment_manifest.py
