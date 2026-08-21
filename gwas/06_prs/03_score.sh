#!/bin/bash
# plink2 --score with the SBayesRC weights. Deterministic: unlike step 02 this
# gives the same numbers on any machine, given the same weights and genotypes.
#
#   bash 03_score.sh ukb  <region>     per-chromosome scores, summed by the analysis scripts
#   bash 03_score.sh adni <region>     one score file over the merged ADNI pgen
#
# snpRes columns used: 2=Name(rsID) 5=A1(effect allele) 8=A1Effect(posterior mean).
# no-mean-imputation: a missing genotype contributes nothing rather than the
# cohort mean, so scores are comparable to the published ones.
set -uo pipefail
. "$(dirname "$0")/../config/common.sh"
gwas_require PRS_WORK_DIR PLINK2

COHORT="${1:?usage: bash 03_score.sh <ukb|adni> <region>}"
REGION="${2:?usage: bash 03_score.sh <ukb|adni> <region>}"
SNPRES="$PRS_WORK_DIR/weights/$REGION/SBayesR.snpRes"
[ -s "$SNPRES" ] || { echo "ABORT: missing $SNPRES"; exit 1; }

case "$COHORT" in
ukb)
    gwas_require UKB_PGEN_TMPL PRS_UKB_KEEP
    OUT="$PRS_WORK_DIR/scores_ukb/$REGION"; mkdir -p "$OUT"
    for chr in $(seq 1 22); do
        "$PLINK2" --pfile "${UKB_PGEN_TMPL//\{chr\}/$chr}" \
            --keep "$PRS_UKB_KEEP" \
            --threads "${PRS_THREADS:-12}" --memory "${PRS_MEMORY_MB:-24000}" \
            --score "$SNPRES" 2 5 8 ignore-dup-ids no-mean-imputation cols=+scoresums \
            --out "$OUT/chr${chr}" > "$OUT/chr${chr}.out" 2>&1 || echo "FAIL $REGION chr$chr"
    done
    echo "$REGION: $(ls "$OUT"/chr*.sscore 2>/dev/null | wc -l)/22 sscore"
    ;;
adni)
    # ADNI variant IDs are CHR:POS:REF:ALT, not rsIDs, so the weights are
    # rewritten against the target .pvar first (04a_build_adni_weights.py).
    gwas_require ADNI_PFILE
    W="$PRS_WORK_DIR/weights_adni/${REGION}.score"
    [ -s "$W" ] || { echo "ABORT: missing $W -- run 04a_build_adni_weights.py"; exit 1; }
    OUT="$PRS_WORK_DIR/scores_adni"; mkdir -p "$OUT"
    "$PLINK2" --pfile "$ADNI_PFILE" \
        --threads "${PRS_THREADS:-8}" --memory "${PRS_MEMORY_MB:-16000}" \
        --score "$W" 1 2 3 header no-mean-imputation cols=+scoresums \
        --out "$OUT/$REGION" > "$OUT/${REGION}.out" 2>&1 \
        && echo "scored $REGION" || echo "FAIL $REGION"
    ;;
*)
    echo "ERROR: cohort must be 'ukb' or 'adni', got '$COHORT'" >&2; exit 1 ;;
esac
