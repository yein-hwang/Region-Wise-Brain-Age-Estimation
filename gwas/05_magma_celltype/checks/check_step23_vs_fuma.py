#!/usr/bin/env python3
"""Compare our R-driven step2/step3 outputs (outputs/step23_v2/<region>/) against
FUMA's downloaded answer key (data/fuma_celltype_results/<region>/).

For each of step1 / step2 / step1_2_summary / step3 that exists on BOTH sides,
match rows on stable key columns and report:
  - row counts, common/only-ours/only-fuma
  - per numeric column: max|diff|
  - per string column: mismatch count
Success = all present files: keys match, numeric max|diff| ~ 0, string mismatches 0.

Usage: python compare_step23_vs_fuma.py <region>
"""
import csv, os, sys, math

def _env(name):
    import os as _os, sys as _sys
    v = _os.environ.get(name)
    if not v:
        _sys.exit(f"ERROR: {name} is not set (see gwas/config/paths.env.example)")
    return v


ROOT = _env("MAGMA_ROOT")
region = sys.argv[1] if len(sys.argv) > 1 else "caudate"
OUR = f"{ROOT}/outputs/step23_v2/{region}"
FUMA = f"{ROOT}/data/fuma_celltype_results/{region}"

def num(x):
    try: return float(x)
    except: return None

def load(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))

def diff_report(name, our_rows, fuma_rows, keycols, numcols, strcols):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    if our_rows is None and fuma_rows is None:
        print("  (absent on both sides — skip)"); return True
    if our_rows is None:
        print("  ❌ MISSING on our side (FUMA has it)"); return False
    if fuma_rows is None:
        print("  (FUMA has none — nothing to validate against)"); return True
    def keyof(r): return tuple(r.get(k, "") for k in keycols)
    ours = {keyof(r): r for r in our_rows}
    fuma = {keyof(r): r for r in fuma_rows}
    ok, fk = set(ours), set(fuma)
    common = ok & fk
    print(f"  our rows={len(our_rows)} fuma rows={len(fuma_rows)} "
          f"common={len(common)} only_ours={len(ok-fk)} only_fuma={len(fk-ok)}")
    for k in list(ok - fk)[:4]: print("    only_ours:", k)
    for k in list(fk - ok)[:4]: print("    only_fuma:", k)
    passed = (len(ok - fk) == 0 and len(fk - ok) == 0 and len(common) > 0)
    for c in numcols:
        maxd, nbad = 0.0, 0
        for k in common:
            a, b = num(ours[k].get(c)), num(fuma[k].get(c))
            if a is None and b is None: continue
            if a is None or b is None: nbad += 1; continue
            d = abs(a - b)
            if d > maxd: maxd = d
        flag = "" if maxd < 1e-4 and nbad == 0 else "  ⚠️"
        print(f"    [num] {c:14s} max|diff|={maxd:.3e}  NA-mismatch={nbad}{flag}")
        if maxd >= 1e-4 or nbad > 0: passed = False
    for c in strcols:
        nbad = sum(1 for k in common if (ours[k].get(c) or "") != (fuma[k].get(c) or ""))
        flag = "" if nbad == 0 else "  ⚠️"
        print(f"    [str] {c:14s} mismatches={nbad}{flag}")
        if nbad > 0:
            passed = False
            for k in list(common)[:200]:
                if (ours[k].get(c) or "") != (fuma[k].get(c) or ""):
                    print(f"        {k}: ours='{ours[k].get(c)}' fuma='{fuma[k].get(c)}'")
    return passed

results = []

# --- step1 ---
results.append(diff_report(
    "step1  (magma_celltype_step1.txt)",
    load(f"{OUR}/magma_celltype_step1.txt"), load(f"{FUMA}/magma_celltype_step1.txt"),
    keycols=["Dataset", "Cell_type"],
    numcols=["NGENES", "BETA", "BETA_STD", "SE", "P", "P.adj.pds", "P.adj"],
    strcols=[]))

# --- step2 ---
results.append(diff_report(
    "step2  (magma_celltype_step2.txt)",
    load(f"{OUR}/magma_celltype_step2.txt"), load(f"{FUMA}/magma_celltype_step2.txt"),
    keycols=["Dataset", "Cell_type", "MODEL"],
    numcols=["NGENES", "BETA", "BETA_STD", "SE", "P", "Marginal.P", "PS"],
    strcols=[]))

# --- step1_2_summary ---
results.append(diff_report(
    "step1_2_summary.txt",
    load(f"{OUR}/step1_2_summary.txt"), load(f"{FUMA}/step1_2_summary.txt"),
    keycols=["Dataset", "Cell_type"],
    numcols=["NGENES", "BETA", "SE", "P", "P.adj.pds", "P.adj", "step3"],
    strcols=["cond_state", "cond_cell_type"]))

# --- step3 --- (key includes CDM.ds = conditioning dataset, stable vs MODEL reindex)
results.append(diff_report(
    "step3  (magma_celltype_step3.txt)",
    load(f"{OUR}/magma_celltype_step3.txt"), load(f"{FUMA}/magma_celltype_step3.txt"),
    keycols=["Dataset", "Cell_type", "CDM.ds"],
    numcols=["NGENES", "BETA", "SE", "P", "CDM.BETA", "CDM.SE", "CDM.P",
             "Marginal.P", "PS", "PS.avg"],
    strcols=[]))

print(f"\n{'#'*60}")
print(f"REGION {region}: {'ALL PASS ✅' if all(results) else 'FAIL ❌ (see ⚠️ above)'}")
print(f"{'#'*60}")
