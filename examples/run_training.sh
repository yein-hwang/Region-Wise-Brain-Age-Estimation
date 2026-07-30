#!/bin/bash
# Train one independent brain-age model per region, 4-fold CV.
#
# Each region is a separate run: its own CNN instance, its own checkpoints and
# its own results directory. Regions are independent, so they can be launched in
# parallel on separate GPUs instead of the sequential loop below.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

COHORT_CSV="${COHORT_CSV:-/path/to/cohort_train.csv}"     # columns: subjectID, imgs, age
REGION_CONFIG="${REGION_CONFIG:-${REPO_DIR}/configs/regions_mni_structural.json}"
ATLAS="${ATLAS:-/path/to/MNI-maxprob-thr0-1mm.nii.gz}"
OUT_ROOT="${OUT_ROOT:-/path/to/output}"

REGIONS=(global caudate cerebellum frontal_lobe insula occipital_lobe parietal_lobe putamen temporal_lobe thalamus)

for region in "${REGIONS[@]}"; do
    echo "===== region: ${region} ====="
    "${PYTHON_BIN}" "${REPO_DIR}/scripts/train_cv.py" \
        --cohort_csv "${COHORT_CSV}" \
        --region "${region}" \
        --region_config "${REGION_CONFIG}" \
        --atlas "${ATLAS}" \
        --mode train \
        --model_dir "${OUT_ROOT}/models/${region}" \
        --results_dir "${OUT_ROOT}/results/${region}" \
        --n_splits 4 \
        --seed 7 \
        --epochs 40 \
        --batch_size 8 \
        --lr 1e-4 \
        --lr_scheduler_choice 0 \
        --patience 0 \
        --n_workers 8
done
