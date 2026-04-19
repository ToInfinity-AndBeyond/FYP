#!/bin/bash
#SBATCH --job-name=zenodo_ssl
#SBATCH --gres=gpu:1
#SBATCH --partition=a16
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/zenodo_ssl_%j.out

set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
VENV_PATH="/vol/bitbucket/mc1920/fypenv"
INPUT_ROOT="/vol/bitbucket/mc1920/zenodo_build_by_subject"
OUTPUT_DIR="/vol/bitbucket/mc1920/zenodo_physio_ssl_experiment"

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

mapfile -t SEGMENT_PATHS < <(find "${INPUT_ROOT}" -path '*/physio_distill/physio_multimodal_accepted_segments.npz' -size +0c | sort)
mapfile -t SUMMARY_PATHS < <(find "${INPUT_ROOT}" -path '*/physio_distill/physio_multimodal_accepted_segment_summary.csv' -size +0c | sort)

if [ "${#SEGMENT_PATHS[@]}" -lt 10 ]; then
  echo "ERROR: expected at least 10 subject bundles under ${INPUT_ROOT}, found ${#SEGMENT_PATHS[@]}"
  exit 1
fi

if [ "${#SEGMENT_PATHS[@]}" -ne "${#SUMMARY_PATHS[@]}" ]; then
  echo "ERROR: segments/summary file count mismatch under ${INPUT_ROOT}"
  exit 1
fi

echo "=== Input Subject Bundles ==="
echo "segment_count=${#SEGMENT_PATHS[@]}"
printf 'segments %s\n' "${SEGMENT_PATHS[@]}"

PYTHONPYCACHEPREFIX=/tmp/pycache \
python train_physio_ssl.py \
  --segments-path "${SEGMENT_PATHS[@]}" \
  --summary-path "${SUMMARY_PATHS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 30 \
  --patience 8 \
  --batch-size 64 \
  --lr 3e-4 \
  --val-subjects 4 \
  --test-subjects 4 \
  --disable-amp
