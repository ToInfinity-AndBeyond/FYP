#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
DATASET_ROOT="/vol/bitbucket/mc1920/zenodo_11242869"
OUTPUT_ROOT="/vol/bitbucket/mc1920/zenodo_build_by_subject"
BUILD_LOG="/vol/bitbucket/mc1920/zenodo_subject_build.log"
SUPERVISOR_LOG="/vol/bitbucket/mc1920/zenodo_subject_supervisor.log"
RETRY_DELAY_SEC="${RETRY_DELAY_SEC:-15}"

mkdir -p "${OUTPUT_ROOT}"
touch "${BUILD_LOG}" "${SUPERVISOR_LOG}"

expected_count="$(find "${DATASET_ROOT}" -maxdepth 1 -name '*_ECG.mat' | wc -l)"

completed_count() {
  find "${OUTPUT_ROOT}" \
    -path '*/physio_distill/physio_multimodal_accepted_segments.npz' \
    -size +0c 2>/dev/null | wc -l
}

worker_count() {
  pgrep -af "build_zenodo_datasets.py .*--dataset-root ${DATASET_ROOT} .*--mode physio" | wc -l
}

log() {
  printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "${SUPERVISOR_LOG}"
}

log "supervisor start expected_subjects=${expected_count}"

while true; do
  completed="$(completed_count)"
  if [[ "${completed}" -ge "${expected_count}" ]]; then
    log "complete completed_subjects=${completed}"
    exit 0
  fi

  active_workers="$(worker_count)"
  if [[ "${active_workers}" -gt 0 ]]; then
    log "worker already running count=${active_workers} completed_subjects=${completed}; sleeping=${RETRY_DELAY_SEC}s"
    sleep "${RETRY_DELAY_SEC}"
    continue
  fi

  log "launch build completed_subjects=${completed} remaining=$((expected_count - completed))"
  set +e
  (
    cd "${PROJECT_DIR}"
    stdbuf -oL -eL ./build_zenodo_physio_by_subject.sh >> "${BUILD_LOG}" 2>&1
  )
  status=$?
  set -e

  completed_after="$(completed_count)"
  if [[ "${completed_after}" -ge "${expected_count}" ]]; then
    log "complete completed_subjects=${completed_after} after_status=${status}"
    exit 0
  fi

  log "build exited status=${status} completed_subjects=${completed_after}; retrying_in=${RETRY_DELAY_SEC}s"
  sleep "${RETRY_DELAY_SEC}"
done
