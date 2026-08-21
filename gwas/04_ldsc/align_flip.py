#!/usr/bin/env python3
"""Allele-aware alignment of one munged sumstats file onto another.

Aligns a source munged sumstats (SNP A1 A2 Z N) to a reference munged sumstats
of the same form. Unlike a strict-match aligner, which keeps only exact allele
matches, this keeps A1/A2-swapped SNPs by flipping the sign of Z -- necessary
when the two files carry opposite allele order for the same variants. Both sides are restricted to LDSC-usable rows
(valid alleles + finite Z/N, N>0), so dry-run counts match the actual rg input. For
each SNP shared between the two:
  - exact  (src A1==ref A1 and src A2==ref A2): keep Z as-is, write ref-order A1/A2.
  - swapped(src A1==ref A2 and src A2==ref A1): FLIP Z sign, write ref-order A1/A2.
  - other: drop.
Strand-ambiguous SNPs (A/T, C/G) were already removed by LDSC munge, so exact vs
swapped is unambiguous. Output columns: SNP A1 A2 Z N (A1/A2 in reference orientation),
gzip-compressed, written atomically (temp in same dir + os.replace).
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Allele-aware (exact + swap/flip) alignment")
    ap.add_argument("--source_file", required=True, help="munged sumstats to be realigned (.sumstats.gz)")
    ap.add_argument("--reference_file", required=True, help="munged sumstats defining the target A1/A2; the output is written in this file's allele orientation (.sumstats.gz)")
    ap.add_argument("--output_file", required=True, help="aligned output (.sumstats.gz)")
    args = ap.parse_args()

    if os.path.exists(args.output_file):
        sys.exit(f"ERROR: output already exists, refusing to overwrite: {args.output_file}")

    # --- reference: restrict to LDSC-usable rows (valid alleles + finite Z/N, N>0) ---
    ref = pd.read_csv(args.reference_file, sep="\t", usecols=["SNP", "A1", "A2", "Z", "N"], low_memory=False)
    ref["Z"] = pd.to_numeric(ref["Z"], errors="coerce")
    ref["N"] = pd.to_numeric(ref["N"], errors="coerce")
    ref_valid = (
        ref["SNP"].notna()
        & ref["A1"].notna()
        & ref["A2"].notna()
        & ref["SNP"].astype(str).str.strip().ne("")
        & ref["A1"].astype(str).str.strip().ne("")
        & ref["A2"].astype(str).str.strip().ne("")
        & ref["A1"].ne(ref["A2"])
        & np.isfinite(ref["Z"])
        & np.isfinite(ref["N"])
        & ref["N"].gt(0)
    )
    n_invalid_reference = int((~ref_valid).sum())
    ref = ref.loc[ref_valid, ["SNP", "A1", "A2"]].copy()
    ref = ref.rename(columns={"A1": "A1_ref", "A2": "A2_ref"})

    # --- source: valid alleles + numeric/finite Z,N (N>0) ---
    src = pd.read_csv(args.source_file, sep="\t", low_memory=False)
    for c in ("SNP", "A1", "A2", "Z", "N"):
        if c not in src.columns:
            sys.exit(f"ERROR: source missing column {c}: {args.source_file}")
    src["Z"] = pd.to_numeric(src["Z"], errors="coerce")
    src["N"] = pd.to_numeric(src["N"], errors="coerce")
    src_valid = (
        src["SNP"].notna()
        & src["A1"].notna()
        & src["A2"].notna()
        & src["SNP"].astype(str).str.strip().ne("")
        & src["A1"].astype(str).str.strip().ne("")
        & src["A2"].astype(str).str.strip().ne("")
        & src["A1"].ne(src["A2"])
        & np.isfinite(src["Z"])
        & np.isfinite(src["N"])
        & src["N"].gt(0)
    )
    n_invalid_source = int((~src_valid).sum())
    src = src.loc[src_valid].copy()

    # --- duplicate SNP guard (prevents many-to-many merge blow-up) ---
    for label, df in (("reference", ref), ("source", src)):
        dup = df["SNP"].duplicated(keep=False)
        if dup.any():
            examples = ",".join(df.loc[dup, "SNP"].astype(str).head(5))
            sys.exit(
                f"ERROR: duplicate valid SNPs in {label}: rows={int(dup.sum())}; examples={examples}"
            )

    m = src.merge(ref, on="SNP", how="inner", validate="one_to_one")
    shared = len(m)

    exact = m["A1"].eq(m["A1_ref"]) & m["A2"].eq(m["A2_ref"])
    swap = m["A1"].eq(m["A2_ref"]) & m["A2"].eq(m["A1_ref"]) & ~exact
    if (exact & swap).any():
        sys.exit("ERROR: exact and swapped masks overlap")
    n_exact = int(exact.sum())
    n_swap = int(swap.sum())
    n_other = shared - n_exact - n_swap

    kept_mask = exact | swap
    kept = m[kept_mask].copy()
    kept_exact = exact[kept_mask].values
    kept["Zout"] = np.where(kept_exact, kept["Z"].values, -kept["Z"].values)

    out = pd.DataFrame({
        "SNP": kept["SNP"].values,
        "A1": kept["A1_ref"].values,   # reference orientation
        "A2": kept["A2_ref"].values,
        "Z": kept["Zout"].values,
        "N": kept["N"].values,
    })

    # --- invariants ---
    if len(out) == 0:
        sys.exit(
            f"ERROR: alignment produced zero rows; shared={shared} exact={n_exact} "
            f"swapped={n_swap} other={n_other}"
        )
    if len(out) != n_exact + n_swap:
        sys.exit("ERROR: aligned row-count invariant failed")
    if not np.isfinite(out["Z"]).all():
        sys.exit("ERROR: non-finite Z remained after alignment")
    if not np.isfinite(pd.to_numeric(out["N"], errors="coerce")).all():
        sys.exit("ERROR: non-finite N remained after alignment")
    if not (pd.to_numeric(out["N"]) > 0).all():
        sys.exit("ERROR: non-positive N remained after alignment")

    tmp = args.output_file + ".tmp.%d" % os.getpid()
    try:
        out.to_csv(tmp, sep="\t", index=False, compression="gzip")
        os.replace(tmp, args.output_file)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    print("STATS shared=%d exact=%d swapped=%d other_dropped=%d "
          "source_invalid_dropped=%d reference_invalid_dropped=%d output_rows=%d"
          % (shared, n_exact, n_swap, n_other, n_invalid_source, n_invalid_reference, len(out)))
    print("OUTPUT %s" % args.output_file)


if __name__ == "__main__":
    main()
