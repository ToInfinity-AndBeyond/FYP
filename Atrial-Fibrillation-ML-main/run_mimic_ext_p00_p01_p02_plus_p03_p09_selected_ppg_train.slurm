#!/bin/bash
#SBATCH --job-name=mimic_ext_p00_p01_p02_plus_selected_ppg
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=slurm_logs/mimic_ext_p00_p01_p02_plus_selected_ppg_%j.out

set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
VENV_PATH="/vol/bitbucket/mc1920/fypenv"
OUTPUT_DIR="/vol/bitbucket/mc1920/mimic_ext_p00_p01_p02_plus_p03_p09_selected_ppg_experiment"
EXTRA_ROOT="/vol/bitbucket/mc1920/mimic_ext_ppg_p03_p09_selected_trainonly"
INIT_CHECKPOINT="/vol/bitbucket/mc1920/mimic_ext_p00_p01_p02_ppg_experiment/best_model.pt"
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
for data_root in "${DATA_ROOTS[@]}"; do
  for fold in 0 1 2 3 4 5 6 7 8 9; do
    segments_path="${data_root}/fold_${fold}/ppg/ppg_accepted_segments.npz"
    summary_path="${data_root}/fold_${fold}/ppg/ppg_accepted_segment_summary.csv"
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
done

EXTRA_SEGMENTS="${EXTRA_ROOT}/ppg/ppg_accepted_segments.npz"
EXTRA_SUMMARY="${EXTRA_ROOT}/ppg/ppg_accepted_segment_summary.csv"
if [ ! -f "${EXTRA_SEGMENTS}" ] || [ ! -f "${EXTRA_SUMMARY}" ]; then
  echo "ERROR: missing extra selected-subset bundle under ${EXTRA_ROOT}"
  exit 1
fi
SEGMENT_ARGS+=("${EXTRA_SEGMENTS}")
SUMMARY_ARGS+=("${EXTRA_SUMMARY}")

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
  --epochs 24 \
  --patience 8 \
  --lr 2e-4 \
  --weight-decay 5e-5 \
  --batch-size 32 \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}"
