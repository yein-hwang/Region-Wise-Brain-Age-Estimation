#!/bin/bash
# LDSC cross-trait genetic correlation for one pair of traits.
#
# Usage: bash 06_genetic_correlation.sh <sumstats1> <sumstats2> <out_prefix>
set -uo pipefail
. "$(dirname "$0")/../config/common.sh"
gwas_require LDSC_DIR LDSC_WORK_DIR LDSC_REF_LD_CHR LDSC_W_LD_CHR

IN1="${1:?Usage: $0 <sumstats1> <sumstats2> <out_prefix>}"
IN2="${2:?Usage: $0 <sumstats1> <sumstats2> <out_prefix>}"
OUT="${3:?Usage: $0 <sumstats1> <sumstats2> <out_prefix>}"
mkdir -p "$(dirname "$OUT")" "$LDSC_WORK_DIR/logs"

"${LDSC_PYTHON:-python}" "$LDSC_DIR/ldsc.py" \
    --rg "${IN1},${IN2}" \
    --w-ld-chr "$LDSC_W_LD_CHR" \
    --ref-ld-chr "$LDSC_REF_LD_CHR" \
    --out "$OUT" \
    > "$LDSC_WORK_DIR/logs/$(basename "$OUT").out" 2>&1
echo "-> ${OUT}.log"
