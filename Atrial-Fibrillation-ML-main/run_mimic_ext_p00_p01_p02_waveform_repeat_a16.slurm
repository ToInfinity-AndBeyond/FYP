#!/bin/bash
#SBATCH --job-name=mimic_ext_waveform_repeat_a16
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
VENV_PATH="/vol/bitbucket/mc1920/fypenv"
RUN_STAMP="20260510_123042"
OUTPUT_BASE="/vol/bitbucket/mc1920/mimic_ext_p00_p01_p02_branch_ablation_repeat_${RUN_STAMP}"
OUTPUT_DIR="${OUTPUT_BASE}/waveform_only_seed123_a16"
SEED="123"
BALANCED_SAMPLER="1"
SAMPLER_POSITIVE_FRACTION="0.333333"
POS_WEIGHT_SCALE="1.0"
DATA_ROOTS=(
  "/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p00_by_fold"
  "/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p01_by_fold"
  "/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p02_by_fold"
)

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

echo "=== Waveform-only Repeat a16 Job ==="
echo "model_variant=waveform_only"
echo "seed=${SEED}"
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
  --model-variant waveform_only \
  --seed "${SEED}" \
  --disable-amp \
  --mixup-alpha 0.0 \
  --epochs 30 \
  --patience 10 \
  --lr 3e-4 \
  --weight-decay 5e-5 \
  --batch-size 32 \
  --output-dir "${OUTPUT_DIR}" \
  "${SAMPLING_ARGS[@]}"
