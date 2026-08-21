#!/bin/bash
# SAIGE Step 2 (single-variant association) -- ADNI replication.
#
# Same flags as the UK Biobank run: BGEN, --AlleleOrder=alt-first,
# --is_imputed_data=TRUE, minMAF=0, minMAC=10, Firth (pCutoff 0.05), LOCO,
# --is_output_moreDetails=TRUE. ADNI chromosome labels are unpadded.
#
# Resumable: skips a chromosome whose result exists and whose log reports
# "Analysis done".
#
# Usage: bash 03_saige_step2.sh [chr ...]                (default 1-22)
#        REGIONS=hippocampus bash 03_saige_step2.sh 22
set -uo pipefail
. "$(dirname "$0")/../config/common.sh"
gwas_require GWAS_ADNI_DIR

BGEN="$GWAS_ADNI_DIR/bgen"
if [ -n "${REGIONS:-}" ]; then regions=($REGIONS); else mapfile -t regions < <(gwas_regions adni); fi
if [ "$#" -ge 1 ]; then chrs=("$@"); else chrs=($(seq 1 22)); fi

launch() {
  local region=$1 chr=$2 node=$3
  local res="$GWAS_ADNI_DIR/$region/results" log="$GWAS_ADNI_DIR/$region/logs"
  mkdir -p "$res" "$log"
  local cmd
  cmd="$(gwas_saige_cmd step2_SPAtests.R \
      --bgenFile="$BGEN/chr${chr}.bgen" \
      --bgenFileIndex="$BGEN/chr${chr}.bgen.bgi" \
      --sampleFile="$BGEN/chr${chr}.samplelist" \
      --AlleleOrder=alt-first \
      --SAIGEOutputFile="$res/chr${chr}.txt" \
      --chrom="${chr}" --minMAF=0 --minMAC=10 --LOCO=TRUE \
      --GMMATmodelFile="$GWAS_ADNI_DIR/$region/step1.rda" \
      --varianceRatioFile="$GWAS_ADNI_DIR/$region/step1.varianceRatio.txt" \
      --is_Firth_beta=TRUE --is_imputed_data=TRUE \
      --pCutoffforFirth=0.05 --is_output_moreDetails=TRUE)"
  gwas_run "$node" "$cmd" > "$log/step2_chr${chr}.log" 2>&1
}

i=0; launched=0; skipped=0
for region in "${regions[@]}"; do
  for chr in "${chrs[@]}"; do
    out="$GWAS_ADNI_DIR/$region/results/chr${chr}.txt"
    if [ -s "$out" ] && grep -q "Analysis done" "$GWAS_ADNI_DIR/$region/logs/step2_chr${chr}.log" 2>/dev/null; then
      skipped=$((skipped+1)); continue
    fi
    gwas_throttle
    node="$(gwas_node $i)"
    echo "[$(date +%H:%M:%S)] launch $region chr$chr ${node:+-> $node}"
    launch "$region" "$chr" "$node" &
    i=$((i+1)); launched=$((launched+1))
  done
done
wait
echo "=== step2 done: launched=$launched skipped=$skipped ==="
