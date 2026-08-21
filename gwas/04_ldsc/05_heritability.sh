#!/bin/bash
# LDSC SNP heritability for one trait.
#
# This standalone --h2 run is the source of the published h2_SNP table.
# The h2 printed inside an --rg log is a different estimate (it is re-fitted on
# the two-trait intersection) and is not what the table reports.
#
# Usage: bash 05_heritability.sh <trait>
#        for r in $(../config/regions.tsv codes); do bash 05_heritability.sh pad_${r}; done
set -uo pipefail
. "$(dirname "$0")/../config/common.sh"
gwas_require LDSC_DIR LDSC_WORK_DIR LDSC_REF_LD_CHR LDSC_W_LD_CHR

TRAIT="${1:?Usage: $0 <trait>}"
mkdir -p "$LDSC_WORK_DIR/results" "$LDSC_WORK_DIR/logs"

"${LDSC_PYTHON:-python}" "$LDSC_DIR/ldsc.py" \
    --h2 "$LDSC_WORK_DIR/sumstats/${TRAIT}.sumstats.gz" \
    --ref-ld-chr "$LDSC_REF_LD_CHR" \
    --w-ld-chr "$LDSC_W_LD_CHR" \
    --out "$LDSC_WORK_DIR/results/heritability_${TRAIT}" \
    > "$LDSC_WORK_DIR/logs/heritability_${TRAIT}.out" 2>&1
echo "-> $LDSC_WORK_DIR/results/heritability_${TRAIT}.log"
