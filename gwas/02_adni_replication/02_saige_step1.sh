#!/bin/bash
# SAIGE Step 1 (null model) -- ADNI replication.
#
# Same design as the UK Biobank run: phenoCol={region}_corrected_delta_age_int,
# quantitative, LOCO, Sex + Age + PC1-10.
#
# Two deliberate differences from the UK Biobank script:
#   * --invNormalize=FALSE. The phenotype column is already inverse-normal
#     transformed on the exact GWAS sample (see build_pheno.py), so this is
#     numerically identical to the UK Biobank run's TRUE, without a second INT.
#   * ADNI_EXTRA_COVARS adds genotyping-batch dummies (ADNI was genotyped in
#     several waves); the reference wave is the omitted category.
#
# Usage: bash 02_saige_step1.sh [region ...]     (default: all ADNI regions)
set -uo pipefail
. "$(dirname "$0")/../config/common.sh"
gwas_require GWAS_ADNI_DIR ADNI_PHENO_FILE

: "${ADNI_EXTRA_COVARS:=is_GO2_set1,is_GO2_set2,is_ADNI3_set1,is_ADNI3_set2}"
COVARS="Sex,Age,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10${ADNI_EXTRA_COVARS:+,$ADNI_EXTRA_COVARS}"

PLINK_PREFIX="$GWAS_ADNI_DIR/grm/adni_grm"
LOGDIR="$GWAS_ADNI_DIR/logs"
mkdir -p "$LOGDIR"

if [ "$#" -ge 1 ]; then regions=("$@"); else mapfile -t regions < <(gwas_regions adni); fi

i=0
for region in "${regions[@]}"; do
    out_dir="$GWAS_ADNI_DIR/$region"
    mkdir -p "$out_dir"
    node="$(gwas_node $i)"

    echo "===== Step 1: $region ${node:+on $node} ====="
    cmd="$(gwas_saige_cmd step1_fitNULLGLMM.R \
        --plinkFile="$PLINK_PREFIX" \
        --phenoFile="$ADNI_PHENO_FILE" \
        --phenoCol="${region}_corrected_delta_age_int" \
        --covarColList="$COVARS" \
        --sampleIDColinphenoFile=IID \
        --traitType=quantitative \
        --invNormalize=FALSE \
        --outputPrefix="${out_dir}/step1" \
        --nThreads="${SAIGE_NTHREADS:-8}" \
        --LOCO=TRUE \
        --IsOverwriteVarianceRatioFile=TRUE)"
    gwas_run "$node" "$cmd" > "$LOGDIR/step1_${region}.log" 2>&1 &
    i=$((i + 1))
done
wait
echo "All Step 1 jobs finished. Check $LOGDIR/step1_*.log and */step1.rda"
