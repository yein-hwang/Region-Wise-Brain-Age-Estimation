#!/bin/bash
# ADNI step 0b: convert the merged dosage pgen to per-chromosome BGEN v1.2 for
# SAIGE step 2.
#
# Two points that matter for effect-allele orientation and must not be changed
# independently of each other:
#   * export is ref-first, so BGEN allele[0] is REF;
#   * SAIGE step 2 is run with --AlleleOrder=alt-first, which treats allele[0]
#     as the effect allele.
# Together these make the effect allele REF, matching the UK Biobank run.
# Verify once per dataset via AF_Allele2 ~ (1 - pvar ALT AF) before trusting it.
#
# The source pgen is fully phased; SAIGE needs unphased BGEN, hence the
# erase-phase pass (keeps dosage, drops HDS).
#
# Usage: bash 01_convert_bgen.sh          # all chr 1-22
#        bash 01_convert_bgen.sh 22       # single chr
set -uo pipefail
. "$(dirname "$0")/../config/common.sh"
gwas_require PLINK2 BGENIX ADNI_PFILE ADNI_KEEP_IIDS GWAS_ADNI_DIR

OUT="$GWAS_ADNI_DIR/bgen"
LOGDIR="$GWAS_ADNI_DIR/logs"
mkdir -p "$OUT" "$LOGDIR"

if [ "$#" -ge 1 ]; then chrs=("$@"); else chrs=($(seq 1 22)); fi

# plink2 --keep wants a header line then IDs; ADNI_KEEP_IIDS is a single IID column.
KEEPF="$OUT/keep_iids.txt"
awk 'BEGIN{print "#IID"}{print $1}' "$ADNI_KEEP_IIDS" > "$KEEPF"

convert_one() {
  local chr=$1 node=$2
  gwas_run "$node" "
    $PLINK2 --pfile $ADNI_PFILE --chr $chr --keep $KEEPF \
            --make-pgen erase-phase --out $OUT/tmp_chr$chr && \
    $PLINK2 --pfile $OUT/tmp_chr$chr \
            --export bgen-1.2 bits=8 ref-first \
            --out $OUT/chr$chr && \
    rm -f $OUT/tmp_chr$chr.pgen $OUT/tmp_chr$chr.pvar $OUT/tmp_chr$chr.psam $OUT/tmp_chr$chr.log && \
    awk 'NR>2{print \$2}' $OUT/chr$chr.sample > $OUT/chr$chr.samplelist && \
    $BGENIX -g $OUT/chr$chr.bgen -index -clobber && \
    echo \"chr$chr DONE bgen=\$(stat -c%s $OUT/chr$chr.bgen) bgi=\$(stat -c%s $OUT/chr$chr.bgen.bgi)\"
  " > "$LOGDIR/convert_chr$chr.log" 2>&1 &
}

i=0
for chr in "${chrs[@]}"; do
  node="$(gwas_node $i)"
  echo "dispatch chr$chr ${node:+-> $node}"
  convert_one "$chr" "$node"
  i=$((i+1))
done
wait
echo "=== conversion done; check $LOGDIR/convert_chr*.log ==="
grep -h "DONE\|Error\|error" "$LOGDIR"/convert_chr*.log 2>/dev/null | tail
