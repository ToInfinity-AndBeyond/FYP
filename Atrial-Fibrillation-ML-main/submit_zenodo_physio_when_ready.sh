#!/bin/bash

set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
SEGMENTS_PATH="/vol/bitbucket/mc1920/zenodo_build_full/physio_distill/physio_multimodal_accepted_segments.npz"
SUMMARY_PATH="/vol/bitbucket/mc1920/zenodo_build_full/physio_distill/physio_multimodal_accepted_segment_summary.csv"
STATE_FILE="/vol/bitbucket/mc1920/zenodo_physio_submit.state"
LOG_FILE="/vol/bitbucket/mc1920/zenodo_physio_submit.log"
CHECK_INTERVAL_SEC=300

mkdir -p "$(dirname "${STATE_FILE}")"
touch "${LOG_FILE}"

if [ -f "${STATE_FILE}" ]; then
  echo "$(date -Iseconds) submit state already exists at ${STATE_FILE}; nothing to do." | tee -a "${LOG_FILE}"
  exit 0
fi

echo "$(date -Iseconds) waiting for Zenodo physio artifacts..." | tee -a "${LOG_FILE}"

while true; do
  if [ -s "${SEGMENTS_PATH}" ] && [ -s "${SUMMARY_PATH}" ]; then
    echo "$(date -Iseconds) artifacts detected, submitting Slurm job..." | tee -a "${LOG_FILE}"
    cd "${PROJECT_DIR}"
    SBATCH_OUTPUT="$(sbatch run_zenodo_physio_distill_train.slurm)"
    printf '%s\n' "${SBATCH_OUTPUT}" | tee -a "${LOG_FILE}"
    {
      echo "submitted_at=$(date -Iseconds)"
      echo "sbatch_output=${SBATCH_OUTPUT}"
    } > "${STATE_FILE}"
    exit 0
  fi

  echo "$(date -Iseconds) still waiting: ${SEGMENTS_PATH}" | tee -a "${LOG_FILE}"
  sleep "${CHECK_INTERVAL_SEC}"
done
