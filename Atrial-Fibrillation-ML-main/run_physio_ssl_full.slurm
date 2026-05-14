#!/bin/bash
#SBATCH --job-name=physio_ssl_full
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main}"
VENV_PATH="${VENV_PATH:-/vol/bitbucket/mc1920/fypenv}"
ZENODO_BASE="${ZENODO_BASE:-/vol/bitbucket/mc1920/zenodo_build_by_subject}"
OUTPUT_DIR="${OUTPUT_DIR:-/vol/bitbucket/mc1920/physio_ssl_full_$(date +%Y%m%d_%H%M%S)}"

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

mapfile -t SEGMENT_ARGS < <(find "${ZENODO_BASE}" -mindepth 3 -maxdepth 3 -path '*/physio_distill/physio_multimodal_accepted_segments.npz' | sort)
mapfile -t SUMMARY_ARGS < <(find "${ZENODO_BASE}" -mindepth 3 -maxdepth 3 -path '*/physio_distill/physio_multimodal_accepted_segment_summary.csv' | sort)

if [ "${#SEGMENT_ARGS[@]}" -eq 0 ] || [ "${#SEGMENT_ARGS[@]}" -ne "${#SUMMARY_ARGS[@]}" ]; then
  echo "ERROR: could not discover matched Zenodo multimodal bundles under ${ZENODO_BASE}"
  echo "segments=${#SEGMENT_ARGS[@]} summaries=${#SUMMARY_ARGS[@]}"
  exit 1
fi

echo "Using ${#SEGMENT_ARGS[@]} subject bundles"

PYTHONPYCACHEPREFIX=/tmp/pycache \
python train_physio_ssl.py \
  --segments-path "${SEGMENT_ARGS[@]}" \
  --summary-path "${SUMMARY_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS:-30}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --lr "${LR:-3e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --temperature "${TEMPERATURE:-0.10}" \
  --alpha-cross "${ALPHA_CROSS:-1.0}" \
  --alpha-ppg "${ALPHA_PPG:-0.25}" \
  --alpha-ecg "${ALPHA_ECG:-0.25}" \
  --val-subjects "${VAL_SUBJECTS:-4}" \
  --test-subjects "${TEST_SUBJECTS:-4}" \
  --patience "${PATIENCE:-8}"
