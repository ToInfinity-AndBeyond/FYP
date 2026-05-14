#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_DIR="${SCRIPT_DIR}/artifacts/mimic_ext_subject_selection/full_p03_p09_by_prefix"
PREFIXES=(p03 p04 p05 p06 p07 p08 p09)
LOG_DIR="${SCRIPT_DIR}/artifacts/mimic_ext_subject_selection/full_download_logs"

mkdir -p "$LOG_DIR"

RETRY_SLEEP="${RETRY_SLEEP:-30}"

if [[ -z "${PHYSIONET_PASS:-}" ]]; then
  if [[ -f "${HOME}/.netrc" ]] && grep -q "machine physionet.org" "${HOME}/.netrc"; then
    :
  else
    read -rsp "PhysioNet password: " PHYSIONET_PASS
    echo
  fi
fi

for prefix in "${PREFIXES[@]}"; do
  manifest="${MANIFEST_DIR}/${prefix}_eligible_segments.csv"
  attempt=1
  while true; do
    log_path="${LOG_DIR}/${prefix}_download_$(date +%Y%m%d_%H%M%S).log"
    echo
    echo "=== Starting ${prefix} (attempt ${attempt}) ==="
    echo "manifest=${manifest}"
    echo "log=${log_path}"
    set +e
    "${SCRIPT_DIR}/download_selected_mimic_ext.sh" "$manifest" | tee "$log_path"
    status=${PIPESTATUS[0]}
    set -e
    if [[ $status -eq 0 ]]; then
      echo "=== Completed ${prefix} ==="
      break
    fi
    echo "=== ${prefix} failed with status ${status}; retrying in ${RETRY_SLEEP}s ==="
    attempt=$((attempt + 1))
    sleep "$RETRY_SLEEP"
  done
done
