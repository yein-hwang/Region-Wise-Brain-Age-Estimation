"""Turn the retained-signals table into the published supplementary table.

Presentation only -- no row is added, dropped or recomputed here:

  * region codes become display names (config/regions.tsv)
  * the curated broad cell class is joined in and placed after the cell-type
    column; `canonical_group`, which it replaces, is dropped
  * internal provenance columns are dropped (--drop)
  * pair counts are written as integers rather than floats
  * the literal "NA" becomes an empty cell

The cell class is a hand-assigned label per cell-type name, not something
derived by rule from the data, so it lives in cell_class_map.tsv and is joined
rather than computed. Any cell type missing from that map is an error, not a
blank.

    python gwas/05_magma_celltype/05_build_supplementary_table.py \
        --table retained_signals.tsv \
        --out   supplementary_retained_cell_type_signals.tsv
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REGIONS_TSV = os.path.join(HERE, "..", "config", "regions.tsv")


def region_labels():
    labels = {}
    with open(REGIONS_TSV) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                labels[parts[0]] = parts[1]
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="retained-signals table (TSV)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default=os.path.join(HERE, "cell_class_map.tsv"))
    ap.add_argument("--cell-col", default="cell_type_original")
    ap.add_argument("--region-col", default="region")
    ap.add_argument("--drop", default="canonical_group,source_file",
                    help="comma-separated columns to drop if present")
    ap.add_argument("--int-cols", default="n_step3_pairs",
                    help="comma-separated columns to render as integers")
    ap.add_argument("--keep-region-codes", action="store_true",
                    help="leave region codes as-is instead of using display names")
    args = ap.parse_args()

    tab = pd.read_csv(args.table, sep="\t", dtype=str, keep_default_na=False)
    if args.cell_col not in tab.columns:
        sys.exit(f"ERROR: --table has no column '{args.cell_col}'")

    cmap = pd.read_csv(args.map, sep="\t", dtype=str, keep_default_na=False, comment="#")
    lookup = dict(zip(cmap["cell_type_original"], cmap["cell_class"]))
    missing = sorted(set(tab[args.cell_col]) - set(lookup))
    if missing:
        sys.exit("ERROR: no cell_class for: " + ", ".join(missing))

    drop = [c for c in args.drop.split(",") if c and c in tab.columns]
    tab = tab.drop(columns=drop)
    tab.insert(tab.columns.get_loc(args.cell_col) + 1,
               "cell_class", tab[args.cell_col].map(lookup))

    if not args.keep_region_codes and args.region_col in tab.columns:
        labels = region_labels()
        unknown = sorted(set(tab[args.region_col]) - set(labels))
        if unknown:
            sys.exit(f"ERROR: no display name for region(s): {unknown}")
        tab[args.region_col] = tab[args.region_col].map(labels)

    for col in args.int_cols.split(","):
        if col and col in tab.columns:
            tab[col] = tab[col].map(
                lambda v: "" if v in ("", "NA") else str(int(float(v))))

    tab = tab.replace("NA", "")

    tab.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {len(tab)} rows, {len(tab.columns)} columns -> {args.out}")
    print(tab["cell_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
