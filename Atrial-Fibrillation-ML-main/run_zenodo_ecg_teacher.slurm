#!/bin/bash
#SBATCH --job-name=zenodo_ecg_teacher
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/zenodo_ecg_teacher_%j.out

set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
VENV_PATH="/vol/bitbucket/mc1920/fypenv"
INPUT_ROOT="/vol/bitbucket/mc1920/zenodo_build_by_subject"
OUTPUT_DIR="/vol/bitbucket/mc1920/zenodo_ecg_teacher_experiment"
SSL_CHECKPOINT="/vol/bitbucket/mc1920/zenodo_physio_ssl_experiment/best_model.pt"

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

mapfile -t SEGMENT_PATHS < <(find "${INPUT_ROOT}" -path '*/physio_distill/physio_multimodal_accepted_segments.npz' -size +0c | sort)
mapfile -t SUMMARY_PATHS < <(find "${INPUT_ROOT}" -path '*/physio_distill/physio_multimodal_accepted_segment_summary.csv' -size +0c | sort)

EXTRA_ARGS=()
if [ -f "${SSL_CHECKPOINT}" ]; then
  EXTRA_ARGS+=(--ssl-checkpoint "${SSL_CHECKPOINT}")
fi

PYTHONPYCACHEPREFIX=/tmp/pycache \
python train_zenodo_ecg_teacher.py \
  --segments-path "${SEGMENT_PATHS[@]}" \
  --summary-path "${SUMMARY_PATHS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 20 \
  --patience 6 \
  --batch-size 32 \
  --lr 5e-4 \
  --val-subjects 5 \
  --test-subjects 5 \
  --balanced-sampler \
  --disable-amp \
  "${EXTRA_ARGS[@]}"
