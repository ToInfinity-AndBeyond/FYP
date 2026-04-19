#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
PYTHON_BIN="/vol/bitbucket/mc1920/fypenv/bin/python"
DATASET_ROOT="/vol/bitbucket/mc1920/FYP/physionet.org/files/mimic-iii-ext-ppg/1.1.0"
COHORT_CSV="${PROJECT_DIR}/artifacts/mimic_ext_ppg_p02_cohort/af_sr_cohort.csv"
OUTPUT_ROOT="/vol/bitbucket/mc1920/mimic_ext_ppg_p02_by_fold"
BUILD_LOG="/vol/bitbucket/mc1920/build_p01_p02_by_fold.log"
SUPERVISOR_LOG="/vol/bitbucket/mc1920/build_p02_supervisor.log"
FOLDS="${FOLDS:-3 4 5 6 7 8 9}"
RETRY_DELAY_SEC="${RETRY_DELAY_SEC:-15}"

mkdir -p "${OUTPUT_ROOT}"
touch "${BUILD_LOG}" "${SUPERVISOR_LOG}"

completed_count() {
  local count=0
  local fold
  for fold in ${FOLDS}; do
    local out_dir="${OUTPUT_ROOT}/fold_${fold}"
    local seg="${out_dir}/ppg/ppg_accepted_segments.npz"
    local summ="${out_dir}/ppg/ppg_accepted_segment_summary.csv"
    if [[ -s "${seg}" && -s "${summ}" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "${count}"
}

fold_total="$(wc -w <<< "${FOLDS}")"

log() {
  printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "${SUPERVISOR_LOG}"
}

log "supervisor start folds=\"${FOLDS}\""

while true; do
  completed="$(completed_count)"
  if [[ "${completed}" -ge "${fold_total}" ]]; then
    log "complete completed_folds=${completed}/${fold_total}"
    exit 0
  fi

  log "launch build completed_folds=${completed}/${fold_total}"
  set +e
  (
    cd "${PROJECT_DIR}"
    stdbuf -oL -eL env \
      PYTHON_BIN="${PYTHON_BIN}" \
      REPO_ROOT="${PROJECT_DIR}" \
      DATASET_ROOT="${DATASET_ROOT}" \
      COHORT_CSV="${COHORT_CSV}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      SUBSET_PREFIX="p02/" \
      SIGNAL_TYPE="ppg" \
      FOLDS="${FOLDS}" \
      ./build_mimic_ext_ppg_by_fold.sh >> "${BUILD_LOG}" 2>&1
  )
  status=$?
  set -e

  completed_after="$(completed_count)"
  if [[ "${completed_after}" -ge "${fold_total}" ]]; then
    log "complete completed_folds=${completed_after}/${fold_total} after_status=${status}"
    exit 0
  fi

  log "build exited status=${status} completed_folds=${completed_after}/${fold_total}; retrying_in=${RETRY_DELAY_SEC}s"
  sleep "${RETRY_DELAY_SEC}"
done
