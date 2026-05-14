#!/bin/bash
#SBATCH --job-name=mimic_ext_p00_p01_p02_hier_ppg
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
VENV_PATH="${VENV_PATH:-/vol/bitbucket/mc1920/fypenv}"
OUTPUT_DIR="${OUTPUT_DIR:-/vol/bitbucket/mc1920/mimic_ext_p00_p01_p02_ppg_hierarchical_experiment}"
DATA_ROOTS=(
  "${DATA_ROOT_P00:-/vol/bitbucket/mc1920/mimic_ext_ppg_p00_by_fold}"
  "${DATA_ROOT_P01:-/vol/bitbucket/mc1920/mimic_ext_ppg_p01_by_fold}"
  "${DATA_ROOT_P02:-/vol/bitbucket/mc1920/mimic_ext_ppg_p02_by_fold}"
)

mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

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
  --epochs 24 \
  --patience 8 \
  --lr 2e-4 \
  --weight-decay 3e-5 \
  --batch-size 8 \
  --max-segments-per-record 48 \
  --eval-max-segments-per-record 96 \
  --quality-inference-strength 0.0 \
  --decision-score prob \
  --segment-aux-weight 0.08 \
  --hard-negative-scale 1.5 \
  --hard-positive-scale 0.85 \
  --segment-quality-power 2.0 \
  --attention-quality-floor 0.30 \
  --attention-quality-power 2.0 \
  --model-type hierarchical \
  --token-dim 192 \
  --transformer-layers 2 \
  --transformer-heads 8 \
  --progress-every-batches 25 \
  --output-dir "${OUTPUT_DIR}"
