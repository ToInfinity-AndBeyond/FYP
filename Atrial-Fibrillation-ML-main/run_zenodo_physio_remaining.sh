#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
PYTHON_BIN="/vol/bitbucket/mc1920/fypenv/bin/python"
DATASET_ROOT="/vol/bitbucket/mc1920/zenodo_11242869"
OUTPUT_ROOT="/vol/bitbucket/mc1920/zenodo_build_by_subject"
LOG_FILE="/vol/bitbucket/mc1920/zenodo_remaining_runner.log"
RETRY_DELAY_SEC="${RETRY_DELAY_SEC:-30}"

mkdir -p "${OUTPUT_ROOT}"
touch "${LOG_FILE}"
cd "${PROJECT_DIR}"

log() {
  printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "${LOG_FILE}"
}

completed_subjects() {
  find "${OUTPUT_ROOT}" \
    -path '*/physio_distill/physio_multimodal_accepted_segments.npz' \
    -size +0c 2>/dev/null | while read -r path; do
      basename "$(dirname "$(dirname "${path}")")"
    done | sort
}

mapfile -t SUBJECT_IDS < <(find "${DATASET_ROOT}" -maxdepth 1 -name '*_ECG.mat' -printf '%f\n' | sed 's/_ECG\.mat$//' | sort)
TOTAL_SUBJECTS="${#SUBJECT_IDS[@]}"

log "zenodo remaining runner start total_subjects=${TOTAL_SUBJECTS}"

while true; do
  mapfile -t DONE < <(completed_subjects)
  unset DONE_MAP
  declare -A DONE_MAP=()
  for subject_id in "${DONE[@]:-}"; do
    DONE_MAP["${subject_id}"]=1
  done

  next_subject=""
  for subject_id in "${SUBJECT_IDS[@]}"; do
    if [[ -z "${DONE_MAP[${subject_id}]:-}" ]]; then
      next_subject="${subject_id}"
      break
    fi
  done

  if [[ -z "${next_subject}" ]]; then
    log "all subjects complete completed_count=${#DONE[@]}"
    exit 0
  fi

  completed_count="${#DONE[@]}"
  log "starting subject=${next_subject} completed_count=${completed_count}/${TOTAL_SUBJECTS}"

  set +e
  "${PYTHON_BIN}" -u build_zenodo_datasets.py \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${OUTPUT_ROOT}/${next_subject}" \
    --mode physio \
    --subject-ids "${next_subject}" \
    --progress-every 250 >> "${LOG_FILE}" 2>&1
  status=$?
  set -e

  accepted_segments_path="${OUTPUT_ROOT}/${next_subject}/physio_distill/physio_multimodal_accepted_segments.npz"
  accepted_summary_path="${OUTPUT_ROOT}/${next_subject}/physio_distill/physio_multimodal_accepted_segment_summary.csv"

  if [[ "${status}" -eq 0 && -s "${accepted_segments_path}" && -s "${accepted_summary_path}" ]]; then
    log "completed subject=${next_subject} status=${status}"
    continue
  fi

  log "subject failed subject=${next_subject} status=${status}; retrying_in=${RETRY_DELAY_SEC}s"
  sleep "${RETRY_DELAY_SEC}"
done
