#!/bin/bash
# Fine-tune one region's pretrained models on a target cohort (4 externally
# defined folds), then run inference and compute the brain-age gaps.
#
# Fold CSVs are expected as {train,val} pairs per fold. Split at the SUBJECT
# level (one row per subject, never the same subject in train and val).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

REGION="${REGION:-temporal_lobe}"
REGION_CONFIG="${REGION_CONFIG:-${REPO_DIR}/configs/regions_mni_structural.json}"
ATLAS="${ATLAS:-/path/to/MNI-maxprob-thr0-1mm.nii.gz}"
FOLD_DIR="${FOLD_DIR:-/path/to/folds}"              # cv_fold_{F}_{train,val}.csv
PRETRAINED_DIR="${PRETRAINED_DIR:-/path/to/pretrained/${REGION}}"   # cv-{F}-{load_epoch}.pth.tar
                                                                    # (falls back to cv-{F}.pth.tar)
INFERENCE_CSV="${INFERENCE_CSV:-/path/to/cohort_all.csv}"
OUT_ROOT="${OUT_ROOT:-/path/to/output}"

MODEL_DIR="${OUT_ROOT}/models_finetuned/${REGION}"
RESULTS_DIR="${OUT_ROOT}/results_finetuned/${REGION}"

# ---- 1. fine-tune each fold (all layers trainable; fold F starts from the
#         pretrained checkpoint of fold F of the same region) ----
for fold in 0 1 2 3; do
    "${PYTHON_BIN}" "${REPO_DIR}/scripts/finetune_fold.py" \
        --train_csv "${FOLD_DIR}/cv_fold_${fold}_train.csv" \
        --val_csv "${FOLD_DIR}/cv_fold_${fold}_val.csv" \
        --region "${REGION}" \
        --region_config "${REGION_CONFIG}" \
        --atlas "${ATLAS}" \
        --fold "${fold}" \
        --model_load_folder "${PRETRAINED_DIR}" \
        --load_epoch 40 \
        --model_dir "${MODEL_DIR}" \
        --results_dir "${RESULTS_DIR}/fold_${fold}" \
        --epochs 40 --batch_size 8 --lr 1e-4 --lr_scheduler_choice 0 \
        --patience 0 --seed 7 --n_workers 8
done

# ---- 2. inference over the full cohort ----
#  If the cohort table has a `prediction_mode` column ('oof_fold_K' for a
#  subject's own validation fold, 'ensemble' otherwise), it is honoured;
#  otherwise every row is scored with the 4-fold ensemble.
"${PYTHON_BIN}" "${REPO_DIR}/scripts/predict.py" \
    --cohort_csv "${INFERENCE_CSV}" \
    --region "${REGION}" \
    --region_config "${REGION_CONFIG}" \
    --atlas "${ATLAS}" \
    --model_dir "${MODEL_DIR}" \
    --n_folds 4 \
    --output_csv "${OUT_ROOT}/predictions/${REGION}.csv" \
    --keep_cols group sex \
    --batch_size 16 --n_workers 8

# ---- 3. brain-age gaps: raw + bias-corrected + INT ----
#  The bias-correction coefficients are fit on the CONTROL out-of-fold
#  predictions (rows whose prediction_mode starts with 'oof') and applied
#  unchanged to every other row.
"${PYTHON_BIN}" "${REPO_DIR}/scripts/compute_bag.py" \
    --predictions_csv "${OUT_ROOT}/predictions/${REGION}.csv" \
    --calibration_filter "prediction_mode=oof*" \
    --int \
    --output_csv "${OUT_ROOT}/bag/${REGION}_bag.csv"
