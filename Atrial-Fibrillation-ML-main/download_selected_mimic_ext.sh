#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main/artifacts/mimic_ext_subject_selection/selected_segments.csv}"
LIMIT="${2:-}"
DATASET_ROOT="${DATASET_ROOT:-/vol/bitbucket/mc1920/FYP/physionet.org/files/mimic-iii-ext-ppg/1.1.0}"
BASE_URL="${BASE_URL:-https://physionet.org/files/mimic-iii-ext-ppg/1.1.0}"
PHYSIONET_USER="${PHYSIONET_USER:-mc1920}"
RETRIES="${RETRIES:-5}"
RETRY_SLEEP="${RETRY_SLEEP:-5}"

format_duration() {
  local total_seconds="$1"
  local hours minutes seconds
  hours=$((total_seconds / 3600))
  minutes=$(((total_seconds % 3600) / 60))
  seconds=$((total_seconds % 60))
  printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

print_progress() {
  local status="$1"
  local rel_path="$2"
  local elapsed remaining avg eta_seconds percent
  elapsed=$(( $(date +%s) - START_TIME ))
  if (( completed > 0 )); then
    avg=$(( elapsed / completed ))
    remaining=$(( total_files - completed ))
    eta_seconds=$(( avg * remaining ))
  else
    eta_seconds=0
  fi
  percent=$(awk -v c="$completed" -v t="$total_files" 'BEGIN { if (t == 0) { printf "100.0" } else { printf "%.1f", (100.0*c)/t } }')
  echo "[${completed}/${total_files} ${percent}% | elapsed $(format_duration "$elapsed") | eta $(format_duration "$eta_seconds")] ${status}: ${rel_path}"
}

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

USE_EXISTING_NETRC=0
if [[ -z "${PHYSIONET_PASS:-}" ]]; then
  if [[ -f "${HOME}/.netrc" ]] && grep -q "machine physionet.org" "${HOME}/.netrc"; then
    USE_EXISTING_NETRC=1
  else
    read -rsp "PhysioNet password: " PHYSIONET_PASS
    echo
  fi
fi

TMPHOME="$(mktemp -d)"
trap 'rm -rf "$TMPHOME"; unset PHYSIONET_PASS' EXIT

if [[ "$USE_EXISTING_NETRC" -eq 1 ]]; then
  cp "${HOME}/.netrc" "$TMPHOME/.netrc"
else
  cat > "$TMPHOME/.netrc" <<EOF
machine physionet.org
login $PHYSIONET_USER
password $PHYSIONET_PASS
EOF
fi
chmod 600 "$TMPHOME/.netrc"

FAILED_LOG="${TMPHOME}/failed_urls.txt"
touch "$FAILED_LOG"

if [[ -n "$LIMIT" ]]; then
  mapfile -t RECORDS < <(
    python - "$MANIFEST" "$LIMIT" <<'PY'
import csv
import itertools
import sys

manifest = sys.argv[1]
limit = int(sys.argv[2])
with open(manifest, newline="", encoding="utf-8") as handle:
    for row in itertools.islice(csv.DictReader(handle), limit):
        print(row["wfdb_record_path"])
PY
  )
else
  mapfile -t RECORDS < <(
    python - "$MANIFEST" <<'PY'
import csv
import sys

manifest = sys.argv[1]
with open(manifest, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        print(row["wfdb_record_path"])
PY
  )
fi

total_records="${#RECORDS[@]}"
total_files=$(( total_records * 2 ))
existing_files=0

for rel in "${RECORDS[@]}"; do
  for ext in hea dat; do
    out="${DATASET_ROOT}/${rel}.${ext}"
    if [[ -s "$out" ]]; then
      existing_files=$((existing_files + 1))
    fi
  done
done

echo "Planned records: ${total_records}"
echo "Planned files: ${total_files}"
echo "Already present: ${existing_files}"
echo "Remaining files: $(( total_files - existing_files ))"
echo

downloaded=0
skipped=0
failed=0
completed=0
START_TIME="$(date +%s)"

for rel in "${RECORDS[@]}"; do
  for ext in hea dat; do
    out="${DATASET_ROOT}/${rel}.${ext}"
    url="${BASE_URL}/${rel}.${ext}"
    mkdir -p "$(dirname "$out")"

    if [[ -s "$out" ]]; then
      skipped=$((skipped + 1))
      completed=$((completed + 1))
      print_progress "skip" "${rel}.${ext}"
      continue
    fi

    success=0
    for attempt in $(seq 1 "$RETRIES"); do
      echo "download ${rel}.${ext} (attempt ${attempt}/${RETRIES})"
      stderr_log="${TMPHOME}/wget.stderr"
      if HOME="$TMPHOME" wget -c --tries=1 --timeout=30 -O "$out" "$url" 2>"$stderr_log"; then
        if [[ -s "$out" ]]; then
          downloaded=$((downloaded + 1))
          completed=$((completed + 1))
          print_progress "downloaded" "${rel}.${ext}"
          success=1
          break
        fi
      fi

      if grep -Eq '401 Unauthorized|403 Forbidden' "$stderr_log"; then
        echo "Authentication failed while downloading $url" >&2
        exit 1
      fi

      if [[ -f "$out" && ! -s "$out" ]]; then
        rm -f "$out"
      fi

      if (( attempt < RETRIES )); then
        sleep "$RETRY_SLEEP"
      fi
    done

    if (( success == 0 )); then
      echo "$url" >> "$FAILED_LOG"
      echo "failed after ${RETRIES} attempts: $url" >&2
      failed=$((failed + 1))
      completed=$((completed + 1))
      print_progress "failed" "${rel}.${ext}"
    fi
  done
done

echo
echo "Download summary"
echo "downloaded=$downloaded"
echo "skipped=$skipped"
echo "failed=$failed"

if [[ -s "$FAILED_LOG" ]]; then
  echo "failed_urls_log=$FAILED_LOG"
else
  echo "failed_urls_log=none"
fi
