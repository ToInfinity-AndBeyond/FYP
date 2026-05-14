#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-/homes/mc1920/FYP/Atrial-Fibrillation-ML-main/artifacts/mimic_ext_subject_selection/selected_segments.csv}"
LIMIT="${2:-}"
DATASET_ROOT="${DATASET_ROOT:-/vol/bitbucket/mc1920/FYP/physionet.org/files/mimic-iii-ext-ppg/1.1.0}"
BASE_URL="${BASE_URL:-https://physionet.org/files/mimic-iii-ext-ppg/1.1.0}"
PHYSIONET_USER="${PHYSIONET_USER:-mc1920}"
RETRIES="${RETRIES:-5}"
RETRY_SLEEP="${RETRY_SLEEP:-5}"
JOBS="${JOBS:-4}"

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
TMPLIST="$(mktemp)"
trap 'rm -rf "$TMPHOME" "$TMPLIST"; unset PHYSIONET_PASS' EXIT

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

if [[ -n "$LIMIT" ]]; then
  python - "$MANIFEST" "$LIMIT" > "$TMPLIST" <<'PY'
import csv
import itertools
import sys

manifest = sys.argv[1]
limit = int(sys.argv[2])
with open(manifest, newline="", encoding="utf-8") as handle:
    for row in itertools.islice(csv.DictReader(handle), limit):
        rel = row["wfdb_record_path"]
        print(f"{rel}.hea")
        print(f"{rel}.dat")
PY
else
  python - "$MANIFEST" > "$TMPLIST" <<'PY'
import csv
import sys

manifest = sys.argv[1]
with open(manifest, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        rel = row["wfdb_record_path"]
        print(f"{rel}.hea")
        print(f"{rel}.dat")
PY
fi

total_files="$(wc -l < "$TMPLIST")"
existing_files=0
while IFS= read -r rel_ext; do
  out="${DATASET_ROOT}/${rel_ext}"
  if [[ -s "$out" ]]; then
    existing_files=$((existing_files + 1))
  fi
done < "$TMPLIST"

echo "Planned files: ${total_files}"
echo "Already present: ${existing_files}"
echo "Remaining files: $(( total_files - existing_files ))"
echo "Parallel jobs: ${JOBS}"
echo

export DATASET_ROOT BASE_URL TMPHOME RETRIES RETRY_SLEEP

download_one() {
  local rel_ext="$1"
  local out="${DATASET_ROOT}/${rel_ext}"
  local url="${BASE_URL}/${rel_ext}"
  local stderr_log

  mkdir -p "$(dirname "$out")"
  if [[ -s "$out" ]]; then
    echo "skip ${rel_ext}"
    return 0
  fi

  for attempt in $(seq 1 "$RETRIES"); do
    stderr_log="$(mktemp)"
    if HOME="$TMPHOME" wget -c --tries=1 --timeout=30 -O "$out" "$url" 2>"$stderr_log"; then
      if [[ -s "$out" ]]; then
        rm -f "$stderr_log"
        echo "downloaded ${rel_ext}"
        return 0
      fi
    fi

    if grep -Eq '401 Unauthorized|403 Forbidden' "$stderr_log"; then
      rm -f "$stderr_log"
      echo "auth_failed ${rel_ext}" >&2
      return 20
    fi

    rm -f "$stderr_log"
    if [[ -f "$out" && ! -s "$out" ]]; then
      rm -f "$out"
    fi
    if (( attempt < RETRIES )); then
      sleep "$RETRY_SLEEP"
    fi
  done

  echo "failed ${rel_ext}" >&2
  return 10
}

export -f download_one

parallel --jobs "$JOBS" --line-buffer --halt now,fail=1 download_one :::: "$TMPLIST"
