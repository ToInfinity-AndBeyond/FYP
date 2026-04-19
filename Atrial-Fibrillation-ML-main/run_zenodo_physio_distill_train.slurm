#!/bin/bash
#SBATCH --job-name=zenodo_physio
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/zenodo_physio_%j.out

set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
VENV_PATH="/vol/bitbucket/mc1920/fypenv"
SEGMENTS_PATH="/vol/bitbucket/mc1920/zenodo_build_full/physio_distill/physio_multimodal_accepted_segments.npz"
SUMMARY_PATH="/vol/bitbucket/mc1920/zenodo_build_full/physio_distill/physio_multimodal_accepted_segment_summary.csv"
OUTPUT_DIR="/vol/bitbucket/mc1920/zenodo_physio_distill_experiment"

mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

echo "=== Host/GPU Info ==="
hostname
date
/usr/bin/nvidia-smi

if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "ERROR: expected venv at ${VENV_PATH}"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"
echo "Using venv at ${VENV_PATH}"
python --version

if [ ! -s "${SEGMENTS_PATH}" ]; then
  echo "ERROR: missing accepted multimodal NPZ at ${SEGMENTS_PATH}"
  exit 1
fi

if [ ! -s "${SUMMARY_PATH}" ]; then
  echo "ERROR: missing accepted multimodal summary CSV at ${SUMMARY_PATH}"
  exit 1
fi

echo "=== Input Artifacts ==="
ls -lh "${SEGMENTS_PATH}" "${SUMMARY_PATH}"

PYTHONPYCACHEPREFIX=/tmp/pycache \
python train_physio_distill.py \
  --segments-path "${SEGMENTS_PATH}" \
  --summary-path "${SUMMARY_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 20 \
  --patience 6 \
  --batch-size 32 \
  --lr 5e-4 \
  --val-records 5 \
  --test-records 5 \
  --split-group-column subject_id \
  --threshold-objective f1 \
  --balanced-sampler \
  --disable-amp
