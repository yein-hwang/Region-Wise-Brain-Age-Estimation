#!/bin/bash
# LDSC munge_sumstats.py, one job per chromosome, for one trait.
#
# --N is the GWAS sample size and must match the summary statistics being
# munged. It is passed explicitly rather than read from the file, which is how
# the published run was done -- see the note in gwas/README.md about the two
# UK Biobank GWAS releases (N=45,076 and N=41,067) and which tables used which.
#
# Usage: bash 02_munge.sh <trait> <N>
#        bash 02_munge.sh caudate 45076
set -uo pipefail
. "$(dirname "$0")/../config/common.sh"
gwas_require LDSC_DIR LDSC_WORK_DIR

TRAIT="${1:?Usage: $0 <trait> <N>}"
SAMPLE_SIZE="${2:?Usage: $0 <trait> <N>}"

IN_DIR="$LDSC_WORK_DIR/files/$TRAIT"
OUT_DIR="$LDSC_WORK_DIR/munge_output/$TRAIT"
mkdir -p "$OUT_DIR" "$LDSC_WORK_DIR/logs"

for chrom in $(seq 1 22); do
    echo "Processing chromosome: $chrom"
    "${LDSC_PYTHON:-python}" "$LDSC_DIR/munge_sumstats.py" \
        --sumstats "${IN_DIR}/chr${chrom}_${TRAIT}_GWAS_SummaryStatistics.txt" \
        --N "${SAMPLE_SIZE}" \
        --snp MarkerID \
        --a1 Allele2 \
        --a2 Allele1 \
        --out "${OUT_DIR}/${TRAIT}_chr${chrom}_munge" \
        --merge-alleles "${IN_DIR}/chr${chrom}_${TRAIT}_GWAS_SummaryStatistics_snplist.txt" \
        > "${OUT_DIR}/${TRAIT}_GWASchr${chrom}.out" 2>&1 &
done
wait
echo "munge done for $TRAIT -> $OUT_DIR"
