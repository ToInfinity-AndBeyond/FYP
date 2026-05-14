#!/bin/bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./submit_gpu_job.sh [slurm-script]

Examples:
  ./submit_gpu_job.sh
  ./submit_gpu_job.sh run_ppg_hybrid_train.slurm
  GPU_SUBMIT_HOST=my-gpu-alias ./submit_gpu_job.sh run_physio_distill_train.slurm

Environment overrides:
  GPU_SUBMIT_HOST   SSH host or alias for the GPU head node
                    default: gpucluster2.doc.ic.ac.uk
  REMOTE_PROJECT_DIR
                    default: /homes/${USER}/FYP/Atrial-Fibrillation-ML-main
  REMOTE_VENV_PATH  default: /vol/bitbucket/${USER}/myvenv
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SLURM_SCRIPT="${1:-run_ppg_gpu.slurm}"
GPU_SUBMIT_HOST="${GPU_SUBMIT_HOST:-gpucluster2.doc.ic.ac.uk}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/homes/${USER}/FYP/Atrial-Fibrillation-ML-main}"
REMOTE_VENV_PATH="${REMOTE_VENV_PATH:-/vol/bitbucket/${USER}/myvenv}"

if [[ "${SLURM_SCRIPT}" == */* ]]; then
  REMOTE_SLURM_PATH="${SLURM_SCRIPT}"
else
  REMOTE_SLURM_PATH="${REMOTE_PROJECT_DIR}/${SLURM_SCRIPT}"
fi

echo "Submitting ${REMOTE_SLURM_PATH} via ${GPU_SUBMIT_HOST}"

ssh "${GPU_SUBMIT_HOST}" \
  "set -euo pipefail; \
   cd '${REMOTE_PROJECT_DIR}'; \
   test -f '${REMOTE_SLURM_PATH}'; \
   PROJECT_DIR='${REMOTE_PROJECT_DIR}' \
   VENV_PATH='${REMOTE_VENV_PATH}' \
   sbatch '${REMOTE_SLURM_PATH}'"

echo
echo "Queue status:"
ssh "${GPU_SUBMIT_HOST}" "squeue --me"
