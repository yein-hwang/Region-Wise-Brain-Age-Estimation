"""Package the SBayesRC weight sets for deposit.

This is what makes the published numbers reproducible. SBayesRC is an MCMC
sampler and a re-run does not return the same weights (README.md, "What is and
is not reproducible"). Everything downstream of the weights -- scoring, ST17,
ST18, ST21-23 -- is deterministic, so depositing the weights moves the
reproducible boundary to where it can actually hold.

Writes one tab-separated file per region:

    SNP  CHR  BP  A1  A2  BETA

A1 is the effect allele; BETA is the SBayesRC posterior mean, per allele.
Coordinates are GRCh37/hg19, inherited from the LD reference panel -- match on
rsID or lift over before scoring a GRCh38 cohort.

    plink2 --pfile <target> --score <region>.weights.tsv 1 4 6 header \
           cols=+scoresums no-mean-imputation

Create-only: refuses to overwrite an existing export, so a deposited package
cannot be silently replaced.

    set -a; . gwas/config/paths.env; set +a
    python gwas/06_prs/07_export_weights.py
"""
import hashlib
import os
import sys

from _regions import REGIONS, require

PRS_WORK_DIR = require("PRS_WORK_DIR")
OUT = os.environ.get("PRS_EXPORT_DIR", os.path.join(PRS_WORK_DIR, "weights_export"))
EXPECTED_N = int(os.environ.get("PRS_EXPORT_EXPECTED_N", "0"))  # 0 = do not check

os.makedirs(OUT, exist_ok=True)
manifest, fail = [], False

for region in REGIONS:
    src = f"{PRS_WORK_DIR}/weights/{region}/SBayesR.snpRes"
    dst = f"{OUT}/{region}.weights.tsv"
    if os.path.exists(dst):
        print(f"ABORT: {dst} already exists (create-only)")
        sys.exit(1)
    n = 0
    with open(src) as f, open(dst, "w") as o:
        o.write("SNP\tCHR\tBP\tA1\tA2\tBETA\n")
        f.readline()
        for line in f:
            c = line.split()
            if len(c) < 8:
                continue
            # snpRes columns: 2=Name 3=Chrom 4=Position 5=A1 6=A2 8=A1Effect
            o.write(f"{c[1]}\t{c[2]}\t{c[3]}\t{c[4]}\t{c[5]}\t{c[7]}\n")
            n += 1
    h = hashlib.md5(open(dst, "rb").read()).hexdigest()
    manifest.append((region, n, h))
    print(f"  {region:16s} {n:>9,} SNP  md5={h}", flush=True)
    if EXPECTED_N and n != EXPECTED_N:
        print(f"  FAIL: {region} wrote {n:,} rows, expected {EXPECTED_N:,}")
        fail = True

with open(f"{OUT}/MANIFEST.txt", "w") as m:
    m.write("Regional brain-age gap PRS weights (SBayesRC posterior means)\n")
    m.write("effect allele: A1   effect size: BETA (per allele)\n")
    m.write("build: GRCh37/hg19, from the LD reference panel\n")
    m.write("scoring: plink2 --score <file> 1 4 6 header cols=+scoresums no-mean-imputation\n\n")
    m.write(f"{'region':16s} {'n_snp':>10s}  md5\n")
    for region, n, h in manifest:
        m.write(f"{region:16s} {n:>10,}  {h}\n")

if fail:
    print("\nEXPORT FAILED")
    sys.exit(1)
print(f"\nwrote {OUT} ({len(manifest)} regions) and MANIFEST.txt")
