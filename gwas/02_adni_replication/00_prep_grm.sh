#!/bin/bash
# ADNI step 0a: build the PLINK1 bed/bim/fam set used as the SAIGE step-1 GRM.
#
# Source = the LD-pruned in-sample PCA set. SAIGE step 1 intersects it with the
# phenotype file's samples automatically, so no subsetting is done here.
set -euo pipefail
. "$(dirname "$0")/../config/common.sh"
gwas_require PLINK2 ADNI_PCA_PFILE GWAS_ADNI_DIR

OUT="$GWAS_ADNI_DIR/grm"
mkdir -p "$OUT"

# hardcall bed/bim/fam (the GRM uses hardcalls); the pruned set is already
# MAF/LD filtered.
"$PLINK2" --pfile "$ADNI_PCA_PFILE" \
          --make-bed \
          --out "$OUT/adni_grm"

echo "GRM plink set:"
wc -l "$OUT/adni_grm.bim" "$OUT/adni_grm.fam"
