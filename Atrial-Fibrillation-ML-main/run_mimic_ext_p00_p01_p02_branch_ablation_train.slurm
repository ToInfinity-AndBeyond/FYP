#!/bin/bash
#SBATCH --job-name=mimic_ext_branch_ablation
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
VENV_PATH="${VENV_PATH:-/vol/bitbucket/mc1920/fypenv}"
MODEL_VARIANT="${MODEL_VARIANT:-full_fusion}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/vol/bitbucket/mc1920/mimic_ext_p00_p01_p02_branch_ablation_${RUN_STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${MODEL_VARIANT}}"
BALANCED_SAMPLER="${BALANCED_SAMPLER:-1}"
SAMPLER_POSITIVE_FRACTION="${SAMPLER_POSITIVE_FRACTION:-0.333333}"
POS_WEIGHT_SCALE="${POS_WEIGHT_SCALE:-1.0}"
DATA_ROOTS=(
  "${DATA_ROOT_P00:-/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p00_by_fold}"
  "${DATA_ROOT_P01:-/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p01_by_fold}"
  "${DATA_ROOT_P02:-/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p02_by_fold}"
)

mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

echo "=== Branch Ablation Job ==="
echo "model_variant=${MODEL_VARIANT}"
echo "output_dir=${OUTPUT_DIR}"
hostname
date
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi unavailable on this host"
fi

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

SAMPLING_ARGS=(--pos-weight-scale "${POS_WEIGHT_SCALE}")
if [ "${BALANCED_SAMPLER}" = "1" ]; then
  SAMPLING_ARGS+=(--balanced-sampler --sampler-positive-fraction "${SAMPLER_POSITIVE_FRACTION}")
fi

PYTHONPYCACHEPREFIX=/tmp/pycache \
python train_ppg_hybrid.py \
  --segments-path "${SEGMENT_ARGS[@]}" \
  --summary-path "${SUMMARY_ARGS[@]}" \
  --split-mode metadata_folds \
  --train-folds 0,1,2,3,4,5,6,7 \
  --val-folds 8 \
  --test-folds 9 \
  --threshold-objective f1 \
  --fusion-mode sqi_conditioned \
  --model-variant "${MODEL_VARIANT}" \
  --disable-amp \
  --mixup-alpha 0.0 \
  --epochs 30 \
  --patience 10 \
  --lr 3e-4 \
  --weight-decay 5e-5 \
  --batch-size 32 \
  --output-dir "${OUTPUT_DIR}" \
  "${SAMPLING_ARGS[@]}"
