#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
STATE_FILE="/vol/bitbucket/mc1920/zenodo_physio_subjects_submit.state"
WATCHER_LOG="/vol/bitbucket/mc1920/zenodo_physio_subjects_submit.log"
SUPERVISOR_LOG="/vol/bitbucket/mc1920/zenodo_physio_subjects_submit_supervisor.log"
RETRY_DELAY_SEC="${RETRY_DELAY_SEC:-30}"

touch "${WATCHER_LOG}" "${SUPERVISOR_LOG}"

log() {
  printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "${SUPERVISOR_LOG}"
}

log "watcher supervisor start"

while true; do
  if [[ -f "${STATE_FILE}" ]]; then
    log "state file present; watcher work complete"
    exit 0
  fi

  log "launch watcher"
  set +e
  (
    cd "${PROJECT_DIR}"
    ./submit_zenodo_physio_subjects_when_ready.sh >/dev/null 2>&1
  )
  status=$?
  set -e

  if [[ -f "${STATE_FILE}" ]]; then
    log "state file created after watcher exit status=${status}"
    exit 0
  fi

  log "watcher exited status=${status} without state file; retrying_in=${RETRY_DELAY_SEC}s"
  sleep "${RETRY_DELAY_SEC}"
done
