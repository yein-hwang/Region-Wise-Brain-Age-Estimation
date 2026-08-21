#!/bin/bash
# MAGMA cell-type steps 1-3 by running FUMA's own magma_celltype.R, unmodified.
#
# Why the original R rather than a re-implementation: step 2's forward-selection
# state machine has nine outcome scenarios and only one region in this project
# produced a step-2 answer key, so a port could not be validated. Running FUMA's
# script means step 1, step 2 and step 3 are FUMA's code by construction.
#
# The script is NOT vendored here (it is FUMA's, under FUMA's terms). Fetch it
# from the FUMA-webapp repository and point FUMA_CELLTYPE_R at it:
#     storage/scripts/magma_celltype.R
# It sources ConfigParser.R from its own directory and reads app.config there;
# see reference/app.config.example for the two keys it needs.
#
# Per region the job directory must contain:
#   magma.genes.raw   a real copy (not a symlink) of the FUMA SNP2GENE gene
#                     analysis output for that region
#   params.config     the FUMA job's parameter file; only datasets=, adjPmeth=,
#                     step2=, step3= and ensg_id= are consulted
# The R script writes step1.sh into the job directory, runs MAGMA serially over
# every dataset, then does steps 2 and 3 itself.
#
# The argument must be an ABSOLUTE path with a trailing slash -- the R script
# concatenates it with filenames.
#
# Usage: bash 03_celltype_step23.sh <region> [region ...]
set -uo pipefail
. "$(dirname "$0")/../config/common.sh"
gwas_require MAGMA_ROOT FUMA_CELLTYPE_R RSCRIPT FUMA_GENES_RAW_TMPL

OUTBASE="$MAGMA_ROOT/outputs/step23"
LOGDIR="$MAGMA_ROOT/outputs/logs"
mkdir -p "$OUTBASE" "$LOGDIR"

if [ "$#" -ge 1 ]; then regions=("$@"); else mapfile -t regions < <(gwas_regions ukb); fi

for region in "${regions[@]}"; do
    jobdir="$OUTBASE/$region"
    mkdir -p "$jobdir"

    genes_raw="${FUMA_GENES_RAW_TMPL//\{region\}/$region}"
    [ -s "$genes_raw" ] || { echo "ERROR: missing $genes_raw" >&2; exit 1; }
    cp -f "$genes_raw" "$jobdir/magma.genes.raw"      # real copy; symlinks are not read

    if [ ! -s "$jobdir/params.config" ]; then
        echo "ERROR: $jobdir/params.config is missing (copy it from the FUMA job)" >&2
        exit 1
    fi

    echo "===== cell-type steps 1-3: $region ====="
    "$RSCRIPT" "$FUMA_CELLTYPE_R" "$jobdir/" > "$LOGDIR/celltype_${region}.log" 2>&1 &
done
wait
echo "Outputs per region: magma_celltype_step1.txt, magma_celltype_step2.txt,"
echo "step1_2_summary.txt, magma_celltype_step3.txt  (under $OUTBASE/<region>/)"
