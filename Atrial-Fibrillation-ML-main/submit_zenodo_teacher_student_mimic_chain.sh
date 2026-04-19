#!/bin/bash
set -euo pipefail

PROJECT_DIR="/homes/mc1920/FYP/Atrial-Fibrillation-ML-main"
STATE_FILE="/vol/bitbucket/mc1920/zenodo_teacher_student_mimic_chain.state"

cd "${PROJECT_DIR}"

teacher_submit=$(sbatch run_zenodo_ecg_teacher.slurm)
teacher_jobid=$(awk '{print $4}' <<<"${teacher_submit}")
echo "teacher_jobid=${teacher_jobid}"

student_submit=$(sbatch --dependency=afterok:${teacher_jobid} run_zenodo_ppg_student_distill.slurm)
student_jobid=$(awk '{print $4}' <<<"${student_submit}")
echo "student_jobid=${student_jobid}"

mimic_submit=$(sbatch --dependency=afterok:${student_jobid} run_mimic_ext_p00_p01_p02_ppg_train_zenodo_init.slurm)
mimic_jobid=$(awk '{print $4}' <<<"${mimic_submit}")
echo "mimic_jobid=${mimic_jobid}"

{
  echo "teacher_jobid=${teacher_jobid}"
  echo "student_jobid=${student_jobid}"
  echo "mimic_jobid=${mimic_jobid}"
} > "${STATE_FILE}"

echo "Saved chain state to ${STATE_FILE}"
