#!/usr/bin/env bash
# step1 gene-property re-run using FUMA's own magma.genes.raw (protein-coding, 19,011 genes)
# Replaces deprecated pass1 (outputs/gene_property/) which used full-Ensembl annot (49,329 genes).
# Validation target: reproduce FUMA magma_celltype_step1.txt exactly (NGENES + BETA/P to printed digits).
set -u
ROOT="${MAGMA_ROOT:?set MAGMA_ROOT -- see gwas/config/paths.env.example}"
MAGMA=$ROOT/magma
REGION=${1:-global}
GENESRAW=$ROOT/data/fuma_magma/$REGION/magma.genes.raw
COVDIR=$ROOT/data/FUMA_scRNA_data_v2/celltype
DSLIST=$ROOT/src/regional_pipeline/ds679.txt          # 679 datasets, spinal excluded (FUMA authoritative)
OUTDIR=$ROOT/outputs/gene_property_v2/$REGION
LOGDIR=$ROOT/outputs/logs
PROG=$LOGDIR/step1_v2_${REGION}_progress.log
PAR=${2:-32}

mkdir -p "$OUTDIR" "$LOGDIR"
: > "$PROG"
echo "[step1_v2] START $(date '+%F %T')  region=$REGION  genes.raw=$GENESRAW  PAR=$PAR" >> "$PROG"

run_one() {
  ds="$1"
  covf="$COVDIR/${ds}.txt"
  out="$OUTDIR/${ds}"
  if [ ! -f "$covf" ]; then echo "MISSING_COVAR $ds" >> "$PROG"; return; fi
  "$MAGMA" --gene-results "$GENESRAW" \
           --gene-covar "$covf" \
           --model condition-hide=Average direction=greater \
           --out "$out" > "$out.runlog" 2>&1
  if [ -f "$out.gsa.out" ] && grep -q '^# TOTAL_GENES' "$out.gsa.out"; then
    echo "OK   $ds" >> "$PROG"
  else
    echo "FAIL $ds" >> "$PROG"
  fi
}
export -f run_one
export MAGMA GENESRAW COVDIR OUTDIR PROG

# fan out; xargs keeps PAR jobs in flight
cat "$DSLIST" | grep -v '^[[:space:]]*$' | \
  xargs -P "$PAR" -I{} bash -c 'run_one "$@"' _ {}

nok=$(grep -c '^OK ' "$PROG")
nfail=$(grep -c '^FAIL' "$PROG")
echo "[step1_v2] COMPLETE $(date '+%F %T')  region=$REGION  OK=$nok  FAIL=$nfail  gsa.out=$(ls "$OUTDIR"/*.gsa.out 2>/dev/null | wc -l)" >> "$PROG"
