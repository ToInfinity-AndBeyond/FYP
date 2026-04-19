#!/bin/bash

set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
PYTHON_BIN="/vol/bitbucket/mc1920/fypenv/bin/python"
DATASET_ROOT="/vol/bitbucket/mc1920/zenodo_11242869"
OUTPUT_ROOT="/vol/bitbucket/mc1920/zenodo_build_by_subject"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_DIR}"

mapfile -t SUBJECT_IDS < <(find "${DATASET_ROOT}" -maxdepth 1 -name '*_ECG.mat' -printf '%f\n' | sed 's/_ECG\.mat$//' | sort)

echo "[zenodo-subject-build] dataset_root=${DATASET_ROOT}"
echo "[zenodo-subject-build] output_root=${OUTPUT_ROOT}"
echo "[zenodo-subject-build] subject_count=${#SUBJECT_IDS[@]}"

for subject_id in "${SUBJECT_IDS[@]}"; do
  subject_output_dir="${OUTPUT_ROOT}/${subject_id}"
  accepted_segments_path="${subject_output_dir}/physio_distill/physio_multimodal_accepted_segments.npz"
  accepted_summary_path="${subject_output_dir}/physio_distill/physio_multimodal_accepted_segment_summary.csv"

  if [ -s "${accepted_segments_path}" ] && [ -s "${accepted_summary_path}" ]; then
    echo "[zenodo-subject-build] skip subject=${subject_id} existing_output=${subject_output_dir}"
    continue
  fi

  echo "[zenodo-subject-build] build subject=${subject_id} output=${subject_output_dir}"
  "${PYTHON_BIN}" build_zenodo_datasets.py \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${subject_output_dir}" \
    --mode physio \
    --subject-ids "${subject_id}" \
    --progress-every 250
done

echo "[zenodo-subject-build] complete"
