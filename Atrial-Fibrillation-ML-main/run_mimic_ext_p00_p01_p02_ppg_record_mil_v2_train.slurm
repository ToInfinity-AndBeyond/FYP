#!/bin/bash
#SBATCH --job-name=mimic_ext_p00_p01_p02_ppg_mil_v2
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
VENV_PATH="${VENV_PATH:-/vol/bitbucket/mc1920/fypenv}"
OUTPUT_DIR="${OUTPUT_DIR:-/vol/bitbucket/mc1920/mimic_ext_p00_p01_p02_ppg_record_mil_v2_experiment}"
DATA_ROOTS=(
  "${DATA_ROOT_P00:-/vol/bitbucket/mc1920/mimic_ext_ppg_p00_by_fold}"
  "${DATA_ROOT_P01:-/vol/bitbucket/mc1920/mimic_ext_ppg_p01_by_fold}"
  "${DATA_ROOT_P02:-/vol/bitbucket/mc1920/mimic_ext_ppg_p02_by_fold}"
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

PYTHONPYCACHEPREFIX=/tmp/pycache \
python train_ppg_record_mil.py \
  --segments-path "${SEGMENT_ARGS[@]}" \
  --summary-path "${SUMMARY_ARGS[@]}" \
  --split-mode metadata_folds \
  --train-folds 0,1,2,3,4,5,6,7 \
  --val-folds 8 \
  --test-folds 9 \
  --threshold-objective f1 \
  --balanced-sampler \
  --disable-amp \
  --epochs 36 \
  --patience 12 \
  --lr 2e-4 \
  --weight-decay 3e-5 \
  --batch-size 8 \
  --max-segments-per-record 48 \
  --eval-max-segments-per-record 96 \
  --quality-inference-strength 0.0 \
  --decision-score prob \
  --segment-aux-weight 0.1 \
  --hard-negative-scale 1.75 \
  --hard-positive-scale 0.75 \
  --segment-quality-power 2.0 \
  --attention-quality-floor 0.35 \
  --attention-quality-power 2.0 \
  --progress-every-batches 25 \
  --output-dir "${OUTPUT_DIR}"
