"""Check that the scripts in gwas/ reproduce the published supplementary tables.

Generalizing a pipeline can silently change it. Each check below re-runs the
released code against the original inputs and compares the result to the
published table, cell by cell. The comparison understands spreadsheet display
precision; see compare_tables.py.

  h2        rebuild the SNP-heritability table from the standalone LDSC --h2
            logs named in the manifest
  rg        rebuild the genetic/phenotypic correlation table by running
            build_rg_table.py then update_rg_table_external.py
  celltype  apply the conditional-retention rule to the significant set, check
            the retained count and the per-region split, then rebuild the
            published table from the retained-signals table

Any check whose reference is not supplied is skipped. The script exits non-zero
if any check that did run failed.

    set -a; . gwas/config/paths.env; set +a
    python gwas/validation/verify_release.py \
        --h2-reference        published/Table_S14.tsv \
        --rg-reference        published/Table_S15.tsv \
        --celltype-reference  published/Table_S16.tsv \
        --retained-signals    retained_signals.tsv \
        --significant-set     significant_set.tsv
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_tables import compare  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GWAS = os.path.dirname(HERE)

PASS, FAIL, SKIP = [], [], []


def read_tsv(path):
    import pandas as pd
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       comment="#").reset_index(drop=True)


def ok(name, detail):
    PASS.append(name)
    print(f"PASS  {name}  ({detail})")


def bad(name, detail):
    FAIL.append(name)
    print(f"FAIL  {name}\n      {detail}")


def run(script, args, env=None):
    r = subprocess.run([sys.executable, script] + args,
                       capture_output=True, text=True, env=env)
    return r.returncode, (r.stderr.strip() or r.stdout.strip())


def check_h2(ref_path, manifest, workdir):
    out = os.path.join(workdir, "h2_table.tsv")
    rc, msg = run(os.path.join(GWAS, "04_ldsc", "build_h2_table.py"),
                  ["--manifest", manifest, "--out", out])
    if rc:
        return bad("h2 / build", msg)
    ref, got = read_tsv(ref_path), read_tsv(out)
    problems = compare(ref, got, label_col=ref.columns[0])
    if problems:
        return bad("h2", "\n      ".join(problems))
    ok("h2", f"{len(ref)} rows x {len(ref.columns)} columns, every cell")


def check_rg(ref_path, workdir):
    out = os.path.join(workdir, "rg_table.tsv")
    env = dict(os.environ, RG_TABLE_OUT=out)
    for script in ("build_rg_table.py", "update_rg_table_external.py"):
        rc, msg = run(os.path.join(GWAS, "04_ldsc", script), [], env=env)
        if rc:
            return bad(f"rg / {script}", msg)
    ref, got = read_tsv(ref_path), read_tsv(out)
    problems = compare(ref, got, label_col="Trait2" if "Trait2" in ref.columns else None)
    if problems:
        return bad("rg", "\n      ".join(problems))
    ok("rg", f"{len(ref)} rows x {len(ref.columns)} columns, every cell")


def check_retention(significant_set):
    import pandas as pd
    spec = json.load(open(os.path.join(HERE, "expected", "celltype_invariants.json")))
    states = set(spec["retained_states"])

    B = pd.read_csv(significant_set, sep="\t", dtype=str, keep_default_na=False)
    if len(B) != spec["n_significant"]:
        return bad("celltype / significant set",
                   f"{len(B)} rows, expected {spec['n_significant']}")
    unexpected = sorted(set(B["survived_at"]) - (states | {"dropped_at_step2"}))
    if unexpected:
        return bad("celltype / states", f"unexpected survived_at values: {unexpected}")
    ret = B[B["survived_at"].isin(states)]
    if len(ret) != spec["n_retained"]:
        return bad("celltype / retention rule",
                   f"{len(ret)} retained, expected {spec['n_retained']}")
    got = ret["region"].value_counts().to_dict()
    if got != spec["region_counts"]:
        return bad("celltype / region split",
                   f"got {got}\n      expected {spec['region_counts']}")
    ok("celltype / retention rule",
       f"{len(ret)} of {len(B)} retained; per-region split matches")


def check_celltype_table(ref_path, retained_signals, workdir):
    out = os.path.join(workdir, "celltype_table.tsv")
    rc, msg = run(os.path.join(GWAS, "05_magma_celltype", "05_build_supplementary_table.py"),
                  ["--table", retained_signals, "--out", out])
    if rc:
        return bad("celltype / table build", msg)
    ref, got = read_tsv(ref_path), read_tsv(out)
    problems = compare(ref, got, label_col="cell_type_original"
                       if "cell_type_original" in ref.columns else None)
    if problems:
        return bad("celltype / table", "\n      ".join(problems))
    ok("celltype / table", f"{len(ref)} rows x {len(ref.columns)} columns, every cell")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h2-reference")
    ap.add_argument("--h2-manifest", default=os.path.join(GWAS, "04_ldsc", "h2_table_traits.tsv"))
    ap.add_argument("--rg-reference")
    ap.add_argument("--celltype-reference")
    ap.add_argument("--retained-signals", help="retained-signals table (209 rows)")
    ap.add_argument("--significant-set", help="significant-set table (264 rows)")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as workdir:
        if args.h2_reference:
            check_h2(args.h2_reference, args.h2_manifest, workdir)
        else:
            SKIP.append("h2")

        if args.rg_reference:
            check_rg(args.rg_reference, workdir)
        else:
            SKIP.append("rg")

        if args.significant_set:
            check_retention(args.significant_set)
        else:
            SKIP.append("celltype / retention rule")

        if args.celltype_reference and args.retained_signals:
            check_celltype_table(args.celltype_reference, args.retained_signals, workdir)
        else:
            SKIP.append("celltype / table")

    for name in SKIP:
        print(f"SKIP  {name}  (no reference given)")
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
