#!/bin/bash
# SAIGE Step 1 (null model) -- UK Biobank region-wise brain-age-gap GWAS.
#
# One null model per region. The phenotype is the inverse-normal-transformed,
# bias-corrected brain-age gap produced by the RegionBAE model
# (column {region}_corrected_delta_age_int in $UKB_PHENO_FILE).
#
# --invNormalize=TRUE is applied on an already-INT column: rank-preserving, so
# it is a no-op here. It is kept because it is what the published run used.
#
# Usage: bash saige_step1.sh [region ...]      (default: all UK Biobank regions)
set -uo pipefail
. "$(dirname "$0")/../config/common.sh"
gwas_require GWAS_UKB_DIR UKB_GRM_PLINK UKB_PHENO_FILE

COVARS="${UKB_COVARS:-Sex,Age,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10}"

if [ "$#" -ge 1 ]; then regions=("$@"); else mapfile -t regions < <(gwas_regions ukb); fi

i=0
for region in "${regions[@]}"; do
    out_dir="$GWAS_UKB_DIR/$region"
    log_dir="$out_dir/logs"
    mkdir -p "$out_dir" "$log_dir"
    node="$(gwas_node $i)"

    echo "===== Step 1: $region ${node:+on $node} ====="
    cmd="$(gwas_saige_cmd step1_fitNULLGLMM.R \
        --plinkFile="$UKB_GRM_PLINK" \
        --phenoFile="$UKB_PHENO_FILE" \
        --phenoCol="${region}_corrected_delta_age_int" \
        --covarColList="$COVARS" \
        --sampleIDColinphenoFile=IID \
        --traitType=quantitative \
        --invNormalize=TRUE \
        --outputPrefix="${out_dir}/step1" \
        --nThreads="${SAIGE_NTHREADS:-4}" \
        --LOCO=TRUE \
        --IsOverwriteVarianceRatioFile=TRUE)"
    gwas_run "$node" "$cmd" > "$log_dir/step1_${region}.log" 2>&1 &
    i=$((i + 1))
done
wait
echo "All Step 1 jobs finished. Check $GWAS_UKB_DIR/*/logs/step1_*.log and */step1.rda"
