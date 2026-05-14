#!/bin/bash
#SBATCH --job-name=qa_asl_array
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --array=0-2%1
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
VENV_PATH="${VENV_PATH:-/vol/bitbucket/mc1920/fypenv}"
QA_CHECKPOINT="${QA_CHECKPOINT:-/vol/bitbucket/mc1920/mimic_ext_p00_p01_p02_qa_beatformer_20260513_181156/best_model.pt}"
OUTPUT_BASE="${OUTPUT_BASE:-/vol/bitbucket/mc1920/qa_beatformer_asl_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOTS=(
  "${DATA_ROOT_P00:-/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p00_by_fold}"
  "${DATA_ROOT_P01:-/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p01_by_fold}"
  "${DATA_ROOT_P02:-/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p02_by_fold}"
)

CONFIG_NAMES=("asl_g2" "asl_g4_softf1" "asl_aggressive")
GAMMA_NEG=(2.0 4.0 4.0)
GAMMA_POS=(1.0 1.0 0.0)
CLIP=(0.05 0.05 0.10)
SOFT_F1=(0.0 0.30 0.30)

CONFIG_NAME="${CONFIG_NAMES[${SLURM_ARRAY_TASK_ID}]}"
OUTPUT_DIR="${OUTPUT_BASE}/${CONFIG_NAME}"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

hostname
date
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

if [ -f "${VENV_PATH}/bin/activate" ] && [ -x "${VENV_PATH}/bin/python" ]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
else
  echo "ERROR: missing venv at ${VENV_PATH}"
  exit 1
fi

SEGMENT_ARGS=()
SUMMARY_ARGS=()
for data_root in "${DATA_ROOTS[@]}"; do
  for fold in 0 1 2 3 4 5 6 7 8 9; do
    SEGMENT_ARGS+=("${data_root}/fold_${fold}/ppg/ppg_accepted_segments.npz")
    SUMMARY_ARGS+=("${data_root}/fold_${fold}/ppg/ppg_accepted_segment_summary.csv")
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
  --model-architecture qa_beatformer \
  --beatformer-max-beats 64 \
  --beatformer-beat-length 128 \
  --beatformer-d-model 128 \
  --beatformer-layers 2 \
  --beatformer-heads 4 \
  --beatformer-dropout 0.1 \
  --init-checkpoint "${QA_CHECKPOINT}" \
  --loss-type asymmetric \
  --asl-gamma-neg "${GAMMA_NEG[${SLURM_ARRAY_TASK_ID}]}" \
  --asl-gamma-pos "${GAMMA_POS[${SLURM_ARRAY_TASK_ID}]}" \
  --asl-clip "${CLIP[${SLURM_ARRAY_TASK_ID}]}" \
  --asl-soft-f1-weight "${SOFT_F1[${SLURM_ARRAY_TASK_ID}]}" \
  --disable-amp \
  --mixup-alpha 0.0 \
  --epochs 8 \
  --patience 4 \
  --lr 3e-5 \
  --weight-decay 5e-5 \
  --batch-size 32 \
  --balanced-sampler \
  --sampler-positive-fraction 0.333333 \
  --pos-weight-scale 1.0 \
  --output-dir "${OUTPUT_DIR}"
