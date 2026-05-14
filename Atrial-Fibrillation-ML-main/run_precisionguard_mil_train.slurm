#!/bin/bash
#SBATCH --job-name=precisionguard_mil
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
VENV_PATH="${VENV_PATH:-/vol/bitbucket/mc1920/fypenv}"
OUTPUT_BASE="${OUTPUT_BASE:-/vol/bitbucket/mc1920/precisionguard_mil_$(date +%Y%m%d_%H%M%S)}"
BRANCH_BASE="${BRANCH_BASE:-/vol/bitbucket/mc1920/mimic_ext_p00_p01_p02_branch_ablation_20260506_155101}"
QA_DIR="${QA_DIR:-/vol/bitbucket/mc1920/mimic_ext_p00_p01_p02_qa_beatformer_20260513_181156}"
DATA_ROOTS=(
  "${DATA_ROOT_P00:-/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p00_by_fold}"
  "${DATA_ROOT_P01:-/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p01_by_fold}"
  "${DATA_ROOT_P02:-/vol/bitbucket/mc1920/mimic_ext_ppg_sqi_v2_p02_by_fold}"
)

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_BASE}"

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

export_model_predictions() {
  local name="$1"
  local checkpoint="$2"
  local out_dir="${OUTPUT_BASE}/stage1_${name}"
  mkdir -p "${out_dir}"
  if [ ! -f "${out_dir}/train_segment_predictions.csv" ]; then
    PYTHONPYCACHEPREFIX=/tmp/pycache \
    python export_ppg_stage1_predictions.py \
      --checkpoint-path "${checkpoint}" \
      --segments-path "${SEGMENT_ARGS[@]}" \
      --summary-path "${SUMMARY_ARGS[@]}" \
      --output-dir "${out_dir}" \
      --batch-size 256
  fi
}

export_model_predictions "waveform" "${BRANCH_BASE}/waveform_only/best_model.pt"
export_model_predictions "gated" "${BRANCH_BASE}/full_fusion/best_model.pt"
export_model_predictions "qa" "${QA_DIR}/best_model.pt"

PYTHONPYCACHEPREFIX=/tmp/pycache \
python train_precisionguard_mil.py \
  --train-prediction "waveform=${OUTPUT_BASE}/stage1_waveform/train_segment_predictions.csv" \
  --train-prediction "gated=${OUTPUT_BASE}/stage1_gated/train_segment_predictions.csv" \
  --train-prediction "qa=${OUTPUT_BASE}/stage1_qa/train_segment_predictions.csv" \
  --val-prediction "waveform=${OUTPUT_BASE}/stage1_waveform/val_segment_predictions.csv" \
  --val-prediction "gated=${OUTPUT_BASE}/stage1_gated/val_segment_predictions.csv" \
  --val-prediction "qa=${OUTPUT_BASE}/stage1_qa/val_segment_predictions.csv" \
  --test-prediction "waveform=${OUTPUT_BASE}/stage1_waveform/test_segment_predictions.csv" \
  --test-prediction "gated=${OUTPUT_BASE}/stage1_gated/test_segment_predictions.csv" \
  --test-prediction "qa=${OUTPUT_BASE}/stage1_qa/test_segment_predictions.csv" \
  --output-dir "${OUTPUT_BASE}/precisionguard_mil" \
  --epochs 200 \
  --patience 25 \
  --batch-size 128 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --hidden-dim 128 \
  --dropout 0.2 \
  --hard-negative-threshold 0.5 \
  --hard-negative-weight 3.0 \
  --balanced-sampler
