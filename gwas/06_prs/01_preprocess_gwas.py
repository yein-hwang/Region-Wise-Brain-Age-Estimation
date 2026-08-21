"""SAIGE summary statistics -> GCTB/COJO .ma, one chromosome at a time.

The allele mapping is the whole point of this file, and it was wrong once.

GCTB's .ma format requires A1 to be the *effect* allele and `freq` to be the
frequency of A1. SAIGE reports BETA and AF_Allele2 with respect to Allele2, so
Allele2 is the effect allele:

    A1   <- Allele2      (effect)
    A2   <- Allele1      (non-effect)
    freq <- AF_Allele2   (frequency of A1)
    b    <- BETA         (effect of A1)

An earlier version mapped A1 <- Allele1 while leaving `b` and `freq` on
Allele2. GCTB does not check this, so the run completes and every downstream
number silently carries the wrong sign. `--check-orientation` re-reads the
source and asserts the mapping held; run it at least once per region.

Columns are rebuilt by name rather than renamed, because a bare rename leaves
the column order as SNP A2 A1 ... which .ma readers take positionally.

    python 01_preprocess_gwas.py --file_path chr22.txt --save_path chr22.ma
    python 01_preprocess_gwas.py --file_path chr22.txt --save_path chr22.ma \
                                 --check-orientation
"""
import argparse
import sys

import pandas as pd

COLS = ['MarkerID', 'Allele1', 'Allele2', 'AF_Allele2', 'BETA', 'SE', 'p.value', 'N']
MA_COLUMNS = ['SNP', 'A1', 'A2', 'freq', 'b', 'se', 'p', 'N']


def to_ma(df):
    return pd.DataFrame({
        'SNP':  df['MarkerID'],
        'A1':   df['Allele2'],
        'A2':   df['Allele1'],
        'freq': df['AF_Allele2'],
        'b':    df['BETA'],
        'se':   df['SE'],
        'p':    df['p.value'],
        'N':    df['N'],
    })


def check_orientation(src, out):
    """Assert the .ma is oriented on the SAIGE effect allele. Exits non-zero if not."""
    problems = []
    if list(out.columns) != MA_COLUMNS:
        problems.append(f"column order is {list(out.columns)}, expected {MA_COLUMNS}")
    if not out['A1'].equals(src['Allele2']):
        problems.append("A1 is not Allele2 -- the effect allele is not in the A1 column")
    if not out['A2'].equals(src['Allele1']):
        problems.append("A2 is not Allele1")
    if not out['freq'].equals(src['AF_Allele2']):
        problems.append("freq is not AF_Allele2 -- the frequency does not refer to A1")
    if not out['b'].equals(src['BETA']):
        problems.append("b is not BETA")
    if problems:
        for p in problems:
            print(f"ORIENTATION FAIL: {p}", file=sys.stderr)
        sys.exit(1)
    print(f"orientation PASS  ({len(out):,} variants: A1=Allele2, freq=AF_Allele2, b=BETA)")


def main(file_path, save_path, verify):
    src = pd.read_csv(file_path, sep='\t', low_memory=False)[COLS].copy()
    out = to_ma(src)
    if verify:
        check_orientation(src, out)
    out.to_csv(save_path, sep='\t', index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert SAIGE summary statistics to the GCTB/COJO .ma format")
    ap.add_argument("--file_path", required=True,
                    help="SAIGE per-chromosome summary statistics (tab-separated)")
    ap.add_argument("--save_path", required=True, help="output .ma path")
    ap.add_argument("--check-orientation", dest="verify", action="store_true",
                    help="assert A1/freq/b refer to the SAIGE effect allele, then exit "
                         "non-zero if they do not")
    a = ap.parse_args()
    main(a.file_path, a.save_path, a.verify)
