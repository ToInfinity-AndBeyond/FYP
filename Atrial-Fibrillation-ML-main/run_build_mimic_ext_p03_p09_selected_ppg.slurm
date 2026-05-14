#!/bin/bash
#SBATCH --job-name=build_mimic_ext_p03_p09_selected_ppg
#SBATCH --partition=a16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/build_mimic_ext_p03_p09_selected_ppg_%j.out

set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
VENV_PATH="/vol/bitbucket/mc1920/fypenv"
DATASET_ROOT="/vol/bitbucket/mc1920/FYP/physionet.org/files/mimic-iii-ext-ppg/1.1.0"
INPUT_MANIFEST="${PROJECT_DIR}/artifacts/mimic_ext_subject_selection/selected_segments.csv"
COHORT_CSV="${PROJECT_DIR}/artifacts/mimic_ext_subject_selection/selected_segments_build_cohort.csv"
OUTPUT_DIR="/vol/bitbucket/mc1920/mimic_ext_ppg_p03_p09_selected_trainonly"

mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

echo "=== Host Info ==="
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

python prepare_selected_mimic_ext_cohort.py \
  --input-csv "${INPUT_MANIFEST}" \
  --output-csv "${COHORT_CSV}" \
  --train-fold 0

PYTHONPYCACHEPREFIX=/tmp/pycache \
python build_mimic_ext_ppg_dataset.py \
  --dataset-root "${DATASET_ROOT}" \
  --cohort-csv "${COHORT_CSV}" \
  --subset-prefix "p" \
  --signal-type ppg \
  --output-dir "${OUTPUT_DIR}"
