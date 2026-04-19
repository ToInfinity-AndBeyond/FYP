#!/bin/bash
#SBATCH --job-name=zenodo_physio_ssl
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/zenodo_physio_ssl_distill_%j.out

set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
VENV_PATH="/vol/bitbucket/mc1920/fypenv"
INPUT_ROOT="/vol/bitbucket/mc1920/zenodo_build_by_subject"
SSL_OUTPUT_DIR="/vol/bitbucket/mc1920/zenodo_physio_ssl_experiment"
OUTPUT_DIR="/vol/bitbucket/mc1920/zenodo_physio_distill_experiment_subjects_ssl"

mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

echo "=== Host/GPU Info ==="
hostname
date
/usr/bin/nvidia-smi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"
echo "Using venv at ${VENV_PATH}"
python --version

if [ ! -f "${SSL_OUTPUT_DIR}/best_model.pt" ]; then
  echo "ERROR: expected SSL checkpoint at ${SSL_OUTPUT_DIR}/best_model.pt"
  exit 1
fi

mapfile -t SEGMENT_PATHS < <(find "${INPUT_ROOT}" -path '*/physio_distill/physio_multimodal_accepted_segments.npz' -size +0c | sort)
mapfile -t SUMMARY_PATHS < <(find "${INPUT_ROOT}" -path '*/physio_distill/physio_multimodal_accepted_segment_summary.csv' -size +0c | sort)

if [ "${#SEGMENT_PATHS[@]}" -eq 0 ]; then
  echo "ERROR: no accepted multimodal NPZ files found under ${INPUT_ROOT}"
  exit 1
fi

if [ "${#SEGMENT_PATHS[@]}" -ne "${#SUMMARY_PATHS[@]}" ]; then
  echo "ERROR: segments/summary file count mismatch under ${INPUT_ROOT}"
  exit 1
fi

echo "=== Input Subject Bundles ==="
echo "segment_count=${#SEGMENT_PATHS[@]}"
printf 'segments %s\n' "${SEGMENT_PATHS[@]}"

PYTHONPYCACHEPREFIX=/tmp/pycache \
python train_physio_distill.py \
  --segments-path "${SEGMENT_PATHS[@]}" \
  --summary-path "${SUMMARY_PATHS[@]}" \
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
  --disable-amp \
  --ssl-checkpoint "${SSL_OUTPUT_DIR}/best_model.pt"
