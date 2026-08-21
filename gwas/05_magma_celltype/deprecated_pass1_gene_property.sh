#!/usr/bin/env bash
# NOT PART OF THE PUBLISHED PIPELINE -- kept for the record.
#
# The first-pass gene-property run, 681 datasets x 10 regions, driven by the
# locally produced genes.raw from 01_gene_analysis_local.sh. Superseded for the
# same reason: wrong gene universe. The published run is 02_celltype_step1.sh,
# which uses FUMA's genes.raw. See ../README.md.
# ---------------------------------------------------------------------------
# step3: MAGMA gene-property (cell-type enrichment) for 681 covars x 10 regions
#        = 6,810 jobs. Identical model for every job:
#   magma --gene-results outputs/regions/{region}.genes.raw
#         --gene-covar {covar}
#         --model condition-hide=Average direction=greater
#         --out outputs/gene_property/{region}__{tag}
#
# Waits until all 10 {region}.genes.raw exist (step2), then runs. gene-property
# reads only .genes.raw (no genotype) -> CPU-bound, low mem -> high parallelism.
# Run on an allowed compute node (00-14 except 05/06/login).
# ---------------------------------------------------------------------------
set -u
ROOT="${MAGMA_ROOT:?set MAGMA_ROOT -- see gwas/config/paths.env.example}"
MAGMA=$ROOT/magma
REGDIR=$ROOT/outputs/regions
GPDIR=$ROOT/outputs/gene_property
LOGDIR=$ROOT/outputs/logs
COVLIST=$ROOT/outputs/covar_list_681.txt
PROG=$LOGDIR/step3_progress.log
mkdir -p "$GPDIR" "$LOGDIR"

REGIONS=(global caudate cerebellum frontal_lobe insula occipital_lobe \
         parietal_lobe putamen temporal_lobe thalamus)
PAR=${PAR:-32}

# --- wait for all 10 genes.raw (step2) ---
echo "[step3] waiting for 10 .genes.raw ... $(date '+%F %T')" > "$PROG"
while :; do
  n=$(ls "$REGDIR"/*.genes.raw 2>/dev/null | wc -l)
  ok=1
  for r in "${REGIONS[@]}"; do [[ -s "$REGDIR/$r.genes.raw" ]] || ok=0; done
  if [[ $ok -eq 1 ]]; then break; fi
  sleep 30
done
echo "[step3] all 10 .genes.raw present; starting 6,810 jobs $(date '+%F %T')" >> "$PROG"

run_one() {
  local region=$1 covar=$2
  local tag; tag=$(basename "$covar" .txt)
  local out=$GPDIR/${region}__${tag}
  if [[ -s "$out.gsa.out" ]]; then echo "SKIP $region $tag"; return 0; fi
  "$MAGMA" --gene-results "$REGDIR/$region.genes.raw" \
           --gene-covar "$covar" \
           --model condition-hide=Average direction=greater \
           --out "$out" > "$out.log" 2>&1
  if [[ -s "$out.gsa.out" ]]; then echo "OK   $region $tag"; else echo "FAIL $region $tag"; fi
}
export -f run_one
export MAGMA REGDIR GPDIR

# build the full job list (region <tab> covar) then feed to xargs
JOBS=$LOGDIR/step3_jobs.txt
: > "$JOBS"
for r in "${REGIONS[@]}"; do
  while IFS= read -r cov; do [[ -n "$cov" ]] && printf '%s\t%s\n' "$r" "$cov" >> "$JOBS"; done < "$COVLIST"
done
total=$(wc -l < "$JOBS")
echo "[step3] launching $total jobs, PAR=$PAR, host=$(hostname) $(date '+%F %T')" >> "$PROG"

# run; append per-job status; periodic progress handled by counting .gsa.out
xargs -P "$PAR" -a "$JOBS" -d '\n' -I{} bash -c '
  IFS=$'"'"'\t'"'"' read -r reg cov <<< "$1"; run_one "$reg" "$cov"
' _ {} >> "$PROG" 2>&1

done_ct=$(ls "$GPDIR"/*.gsa.out 2>/dev/null | wc -l)
fail_ct=$(grep -c '^FAIL' "$PROG" 2>/dev/null || echo 0)
echo "[step3] COMPLETE $(date '+%F %T')  gsa.out=$done_ct  FAIL=$fail_ct" >> "$PROG"
