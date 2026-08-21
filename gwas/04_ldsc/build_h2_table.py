"""Assemble the SNP-heritability table from standalone LDSC --h2 logs.

One row per trait, in the order given by the manifest. Every value is parsed
from the log; nothing is recomputed except

    h2_Z = h2 / h2_SE                       (from the printed 4-decimal values)
    h2_P = one-sided normal tail, norm.sf(h2_Z)

which is how the published table was derived.

The h2 printed inside an --rg log is a *different* estimate (LDSC re-fits it on
the two-trait SNP intersection) and is deliberately not used here.

    set -a; . gwas/config/paths.env; set +a
    python gwas/04_ldsc/build_h2_table.py \
        --manifest gwas/04_ldsc/h2_table_traits.tsv \
        --out h2_table.tsv
"""
import argparse
import os
import re
import sys

from scipy import stats

FLT = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
RE_NSNP = re.compile(r"^After merging with regression SNP LD, (\d+) SNPs remain\.", re.M)
RE_H2 = re.compile(r"^Total Observed scale h2:\s*(" + FLT + r")\s*\((" + FLT + r")\)", re.M)
RE_LAMBDA = re.compile(r"^Lambda GC:\s*(" + FLT + r")", re.M)
RE_CHI2 = re.compile(r"^Mean Chi\^2:\s*(" + FLT + r")", re.M)
RE_INT = re.compile(r"^Intercept:\s*(" + FLT + r")\s*\((" + FLT + r")\)", re.M)
RE_RATIO = re.compile(r"^Ratio:\s*(" + FLT + r")\s*\((" + FLT + r")\)", re.M)

STATS = ["N_SNPs", "h2", "h2_SE", "h2_Z", "h2_P", "Lambda_GC",
         "Mean_Chi2", "Intercept", "Intercept_SE", "Ratio", "Ratio_SE"]


def one(rx, txt, path, what, groups=1):
    m = rx.search(txt)
    if not m:
        sys.exit(f"ERROR: no {what} in {path}")
    return m.groups() if groups > 1 else m.group(1)


def parse_log(path, verbatim=False):
    """Parse one --h2 log.

    LDSC prints a variable number of decimals. The published table fixes them,
    except for rows that were transcribed straight from the log -- pass
    verbatim=True for those (see the manifest).
    """
    with open(path) as f:
        txt = f.read()
    if "--h2 " not in txt:
        sys.exit(f"ERROR: {path} is not a standalone --h2 run")
    h2, h2_se = one(RE_H2, txt, path, "h2", 2)
    icpt, icpt_se = one(RE_INT, txt, path, "intercept", 2)
    ratio, ratio_se = one(RE_RATIO, txt, path, "ratio", 2)
    z = float(h2) / float(h2_se)
    if verbatim:
        return {
            "N_SNPs": one(RE_NSNP, txt, path, "SNP count"),
            "h2": h2,
            "h2_SE": h2_se,
            "h2_Z": "%.2f" % z,
            "h2_P": "%.3g" % stats.norm.sf(z),
            "Lambda_GC": one(RE_LAMBDA, txt, path, "lambda GC"),
            "Mean_Chi2": one(RE_CHI2, txt, path, "mean chi2"),
            "Intercept": icpt,
            "Intercept_SE": icpt_se,
            "Ratio": ratio,
            "Ratio_SE": ratio_se,
        }
    return {
        "N_SNPs": one(RE_NSNP, txt, path, "SNP count"),
        "h2": "%.4f" % float(h2),
        "h2_SE": "%.4f" % float(h2_se),
        "h2_Z": "%.2f" % z,
        "h2_P": "%.3e" % stats.norm.sf(z),
        "Lambda_GC": "%.4f" % float(one(RE_LAMBDA, txt, path, "lambda GC")),
        "Mean_Chi2": "%.4f" % float(one(RE_CHI2, txt, path, "mean chi2")),
        "Intercept": "%.4f" % float(icpt),
        "Intercept_SE": "%.4f" % float(icpt_se),
        "Ratio": "%.3f" % float(ratio),
        "Ratio_SE": "%.4f" % float(ratio_se),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True,
                    help="TSV, one row per trait in table order: "
                         "display_name<TAB>path_to_h2_log[<TAB>verbatim]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-column", default="Trait",
                    help="header of the first column (default: Trait)")
    args = ap.parse_args()

    rows = []
    with open(args.manifest) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            display, path = parts[0], os.path.expandvars(parts[1])
            verbatim = len(parts) > 2 and parts[2].strip() == "verbatim"
            if not os.path.exists(path):
                sys.exit(f"ERROR: log not found for '{display}': {path}")
            rec = parse_log(path, verbatim=verbatim)
            rec[args.label_column] = display
            rows.append(rec)

    columns = [args.label_column] + STATS
    with open(args.out, "w") as f:
        f.write("\t".join(columns) + "\n")
        for r in rows:
            f.write("\t".join(r[c] for c in columns) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
