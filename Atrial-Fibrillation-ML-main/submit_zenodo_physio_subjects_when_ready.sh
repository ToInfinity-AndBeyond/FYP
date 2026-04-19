#!/bin/bash

set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
DATASET_ROOT="/vol/bitbucket/mc1920/zenodo_11242869"
INPUT_ROOT="/vol/bitbucket/mc1920/zenodo_build_by_subject"
STATE_FILE="/vol/bitbucket/mc1920/zenodo_physio_subjects_submit.state"
LOG_FILE="/vol/bitbucket/mc1920/zenodo_physio_subjects_submit.log"
CHECK_INTERVAL_SEC=300

mkdir -p "$(dirname "${STATE_FILE}")"
touch "${LOG_FILE}"

if [ -f "${STATE_FILE}" ]; then
  echo "$(date -Iseconds) submit state already exists at ${STATE_FILE}; nothing to do." | tee -a "${LOG_FILE}"
  exit 0
fi

expected_count="$(find "${DATASET_ROOT}" -maxdepth 1 -name '*_ECG.mat' | wc -l)"
echo "$(date -Iseconds) waiting for ${expected_count} per-subject Zenodo physio bundles..." | tee -a "${LOG_FILE}"

while true; do
  segment_count="$(find "${INPUT_ROOT}" -path '*/physio_distill/physio_multimodal_accepted_segments.npz' -size +0c 2>/dev/null | wc -l)"
  summary_count="$(find "${INPUT_ROOT}" -path '*/physio_distill/physio_multimodal_accepted_segment_summary.csv' -size +0c 2>/dev/null | wc -l)"
  echo "$(date -Iseconds) progress segment_count=${segment_count} summary_count=${summary_count} expected=${expected_count}" | tee -a "${LOG_FILE}"

  if [ "${segment_count}" -eq "${expected_count}" ] && [ "${summary_count}" -eq "${expected_count}" ]; then
    echo "$(date -Iseconds) all subject bundles ready, submitting Slurm job..." | tee -a "${LOG_FILE}"
    cd "${PROJECT_DIR}"
    SBATCH_OUTPUT="$(sbatch run_zenodo_physio_distill_subjects.slurm)"
    printf '%s\n' "${SBATCH_OUTPUT}" | tee -a "${LOG_FILE}"
    {
      echo "submitted_at=$(date -Iseconds)"
      echo "sbatch_output=${SBATCH_OUTPUT}"
    } > "${STATE_FILE}"
    exit 0
  fi

  sleep "${CHECK_INTERVAL_SEC}"
done
