"""Rewrite the SBayesRC weights against the ADNI variant IDs.

ADNI's .pvar identifies variants as CHR:POS:REF:ALT, the UK Biobank HapMap3 set
by rsID, so the weights cannot be handed to plink2 as they are. Each weight is
matched on position and allele pair in either order; the effect allele (A1) is
carried through unchanged, so the sign is preserved regardless of which
orientation the target file happens to use.

Aborts if any region matches under 95% of its weights -- that would mean the
build or the reference panel does not line up.

    set -a; . gwas/config/paths.env; set +a
    python gwas/06_prs/04a_build_adni_weights.py
"""
import os
import sys
import time

from _regions import REGIONS, require

PRS_WORK_DIR, ADNI_PFILE = require("PRS_WORK_DIR", "ADNI_PFILE")
PVAR = ADNI_PFILE + ".pvar"
OUT_DIR = os.path.join(PRS_WORK_DIR, "weights_adni")
os.makedirs(OUT_DIR, exist_ok=True)

t = time.time()
idset = set()
with open(PVAR) as f:
    for line in f:
        if line.startswith("#"):
            continue
        idset.add(line.split("\t", 4)[2])
print(f"pvar IDs loaded: {len(idset):,} ({time.time() - t:.1f}s)", flush=True)

fail = False
for region in REGIONS:
    snp = f"{PRS_WORK_DIR}/weights/{region}/SBayesR.snpRes"
    out = f"{OUT_DIR}/{region}.score"
    n_in = n_match = n_ambig = 0
    with open(snp) as f, open(out, "w") as o:
        o.write("ID\tA1\tBETA\n")
        f.readline()
        for line in f:
            c = line.split()
            if len(c) < 8:
                continue
            n_in += 1
            chrom, pos, a1, a2, beta = c[2], c[3], c[4], c[5], c[7]
            id1, id2 = f"{chrom}:{pos}:{a1}:{a2}", f"{chrom}:{pos}:{a2}:{a1}"
            m1, m2 = id1 in idset, id2 in idset
            if m1 and m2:
                n_ambig += 1
                o.write(f"{id2}\t{a1}\t{beta}\n"); n_match += 1
            elif m1:
                o.write(f"{id1}\t{a1}\t{beta}\n"); n_match += 1
            elif m2:
                o.write(f"{id2}\t{a1}\t{beta}\n"); n_match += 1
    pct = 100 * n_match / max(n_in, 1)
    print(f"{region:16s} weights={n_in:>9,} matched={n_match:>9,} ({pct:5.1f}%) ambiguous={n_ambig}",
          flush=True)
    if pct < 95:
        print(f"  FAIL: {region} match rate {pct:.1f}% < 95%")
        fail = True

if fail:
    sys.exit(1)
print(f"\nwrote {OUT_DIR}/<region>.score")
