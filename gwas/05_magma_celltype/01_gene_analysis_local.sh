#!/usr/bin/env bash
# NOT PART OF THE PUBLISHED PIPELINE -- kept for the record.
#
# MAGMA gene analysis run locally against a full-Ensembl annotation. Its output
# was discarded: FUMA's annotation is protein-coding only, which changes NGENES
# (22,933 here vs 16,777 at FUMA) and flips significance for 21 cell types. The
# published cell-type analysis uses the magma.genes.raw downloaded from a FUMA
# SNP2GENE job (FUMA_GENES_RAW_TMPL). See ../README.md.
# ---------------------------------------------------------------------------
# step2: MAGMA gene analysis for all 10 regional BAG GWAS.
# Produces outputs/regions/{region}.genes.raw (+ .genes.out, .log).
#
# Identical command for every region (only sumstats path + out prefix vary):
#   magma --bfile g1000_eur
#         --pval {sumstats} use=MarkerID,p.value ncol=N
#         --gene-annot annot_g1000eur_ENSG.genes.annot
#         --out outputs/regions/{region}
#
# Run on an ALLOWED compute node (00,01,02,03,04,07-14; NOT 05/06/login).
# Bounded parallelism (PAR) to cap RAM/IO. Each region loads the 2.8G bed.
# ---------------------------------------------------------------------------
set -u
ROOT="${MAGMA_ROOT:?set MAGMA_ROOT -- see gwas/config/paths.env.example}"
MAGMA=$ROOT/magma
BFILE=$ROOT/data/g1000_eur
ANNOT=$ROOT/data/annot_g1000eur_ENSG.genes.annot
GWAS="${GWAS_UKB_DIR:?set GWAS_UKB_DIR -- see gwas/config/paths.env.example}"
OUTDIR=$ROOT/outputs/regions
LOGDIR=$ROOT/outputs/logs
mkdir -p "$OUTDIR" "$LOGDIR"

REGIONS=(global caudate cerebellum frontal_lobe insula occipital_lobe \
         parietal_lobe putamen temporal_lobe thalamus)
PAR=${PAR:-5}          # max concurrent regions

run_one() {
  local r=$1
  local ss=$GWAS/$r/results/${r}_imputed_sumstats.txt
  local out=$OUTDIR/$r
  local log=$LOGDIR/step2_${r}.log
  if [[ ! -f "$ss" ]]; then echo "MISS sumstats $r" >&2; return 3; fi
  echo "[start] $r  $(date '+%F %T')" > "$log"
  "$MAGMA" --bfile "$BFILE" \
           --pval "$ss" use=MarkerID,p.value ncol=N \
           --gene-annot "$ANNOT" \
           --out "$out" >> "$log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 && -f "$out.genes.raw" ]]; then
    echo "[done]  $r  rc=$rc  $(date '+%F %T')" >> "$log"
    echo "OK   $r"
  else
    echo "[FAIL] $r  rc=$rc  $(date '+%F %T')" >> "$log"
    echo "FAIL $r  rc=$rc"
  fi
}
export -f run_one
export MAGMA BFILE ANNOT GWAS OUTDIR LOGDIR

echo "=== step2 gene analysis: ${#REGIONS[@]} regions, PAR=$PAR, host=$(hostname) ==="
printf '%s\n' "${REGIONS[@]}" | xargs -P "$PAR" -I{} bash -c 'run_one "$@"' _ {}
echo "=== step2 complete: $(date '+%F %T') ==="
echo "--- .genes.raw produced ---"
ls -la "$OUTDIR"/*.genes.raw 2>/dev/null | awk '{print $NF, $5}'
