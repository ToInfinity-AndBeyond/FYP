#!/bin/bash
#SBATCH --job-name=mimic_p00_p01_p02_zenodo_init
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=slurm_logs/mimic_p00_p01_p02_zenodo_init_%j.out

set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
VENV_PATH="/vol/bitbucket/mc1920/fypenv"
OUTPUT_DIR="/vol/bitbucket/mc1920/mimic_ext_p00_p01_p02_ppg_experiment_zenodo_init"
INIT_CHECKPOINT="/vol/bitbucket/mc1920/zenodo_ppg_student_distill_experiment/best_model.pt"
DATA_ROOTS=(
  "/vol/bitbucket/mc1920/mimic_ext_ppg_p00_by_fold"
  "/vol/bitbucket/mc1920/mimic_ext_ppg_p01_by_fold"
  "/vol/bitbucket/mc1920/mimic_ext_ppg_p02_by_fold"
)

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

if [ ! -f "${INIT_CHECKPOINT}" ]; then
  echo "ERROR: missing init checkpoint at ${INIT_CHECKPOINT}"
  exit 1
fi

SEGMENT_ARGS=()
SUMMARY_ARGS=()
for data_root in "${DATA_ROOTS[@]}"; do
  for fold in 0 1 2 3 4 5 6 7 8 9; do
    segments_path="${data_root}/fold_${fold}/ppg/ppg_accepted_segments.npz"
    summary_path="${data_root}/fold_${fold}/ppg/ppg_accepted_segment_summary.csv"
    SEGMENT_ARGS+=("${segments_path}")
    SUMMARY_ARGS+=("${summary_path}")
  done
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
  --epochs 30 \
  --patience 10 \
  --lr 2e-4 \
  --weight-decay 5e-5 \
  --batch-size 32 \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}"
