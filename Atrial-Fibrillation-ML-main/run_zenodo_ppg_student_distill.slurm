#!/bin/bash
#SBATCH --job-name=zenodo_ppg_distill
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/zenodo_ppg_distill_%j.out

set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
VENV_PATH="/vol/bitbucket/mc1920/fypenv"
INPUT_ROOT="/vol/bitbucket/mc1920/zenodo_build_by_subject"
TEACHER_CHECKPOINT="/vol/bitbucket/mc1920/zenodo_ecg_teacher_experiment/best_model.pt"
OUTPUT_DIR="/vol/bitbucket/mc1920/zenodo_ppg_student_distill_experiment"

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

if [ ! -f "${TEACHER_CHECKPOINT}" ]; then
  echo "ERROR: missing teacher checkpoint at ${TEACHER_CHECKPOINT}"
  exit 1
fi

mapfile -t SEGMENT_PATHS < <(find "${INPUT_ROOT}" -path '*/physio_distill/physio_multimodal_accepted_segments.npz' -size +0c | sort)
mapfile -t SUMMARY_PATHS < <(find "${INPUT_ROOT}" -path '*/physio_distill/physio_multimodal_accepted_segment_summary.csv' -size +0c | sort)

PYTHONPYCACHEPREFIX=/tmp/pycache \
python train_zenodo_ppg_student_distill.py \
  --segments-path "${SEGMENT_PATHS[@]}" \
  --summary-path "${SUMMARY_PATHS[@]}" \
  --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 20 \
  --patience 8 \
  --batch-size 32 \
  --lr 3e-4 \
  --weight-decay 5e-5 \
  --balanced-sampler \
  --disable-amp \
  --alpha-kd 0.7 \
  --temperature 2.0
