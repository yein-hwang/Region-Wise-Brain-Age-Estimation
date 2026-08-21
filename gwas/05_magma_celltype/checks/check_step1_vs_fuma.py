#!/usr/bin/env python3
"""Compare our step1_v2 (.gsa.out under outputs/gene_property_v2/<region>/) against
FUMA's magma_celltype_step1.txt for the same region.

Usage: python compare_v2_vs_fuma.py <region>
Success criteria: rows==12531, common==12531, NGENES all equal, max|dBETA|~0, flips==0.
"""
import csv, glob, os, sys, math

def _env(name):
    import os as _os, sys as _sys
    v = _os.environ.get(name)
    if not v:
        _sys.exit(f"ERROR: {name} is not set (see gwas/config/paths.env.example)")
    return v


ROOT = _env("MAGMA_ROOT")
region = sys.argv[1] if len(sys.argv) > 1 else "global"

def num(x):
    try: return float(x)
    except: return None

# --- FUMA step1 reference, key=(Dataset, Cell_type) ---
fuma = {}
with open(f"{ROOT}/data/fuma_celltype_results/{region}/magma_celltype_step1.txt") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        fuma[(row["Dataset"], row["Cell_type"])] = row

# --- our v2 .gsa.out, FULL_NAME = ' '.join(p[7:]) ---
ours = {}
for path in glob.glob(f"{ROOT}/outputs/gene_property_v2/{region}/*.gsa.out"):
    ds = os.path.basename(path)[:-len(".gsa.out")]
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("VARIABLE"):
                continue
            p = line.split()
            if len(p) < 7 or p[1] != "COVAR":
                continue
            fullname = " ".join(p[7:]) if len(p) > 7 else p[0]
            ours[(ds, fullname)] = dict(NGENES=num(p[2]), BETA=num(p[3]),
                                        SE=num(p[5]), P=num(p[6]))

fk, ok = set(fuma), set(ours)
common = fk & ok
print(f"=== region: {region} ===")
print(f"FUMA rows : {len(fuma)}   (target 12531)")
print(f"OUR rows  : {len(ours)}   (target 12531)")
print(f"common    : {len(common)}")
print(f"only_ours : {len(ok - fk)}")
print(f"only_fuma : {len(fk - ok)}")
if ok - fk:
    for k in list(ok - fk)[:5]: print("   only_ours e.g.:", k)
if fk - ok:
    for k in list(fk - ok)[:5]: print("   only_fuma e.g.:", k)

# --- numeric comparison over common (skip rows where either side NA) ---
ng_mismatch = []
dbeta = []; dp = []
for k in common:
    a = ours[k]; b = fuma[k]
    ng_o, ng_f = a["NGENES"], num(b["NGENES"])
    if ng_o is not None and ng_f is not None and ng_o != ng_f:
        ng_mismatch.append((k, ng_o, ng_f))
    bo, bf = a["BETA"], num(b["BETA"])
    po, pf = a["P"], num(b["P"])
    if None not in (bo, bf): dbeta.append(abs(bo - bf))
    if None not in (po, pf): dp.append(abs(po - pf))

print(f"\nNGENES mismatches : {len(ng_mismatch)}")
for k, o, f in ng_mismatch[:10]:
    print(f"   {k}: ours={o} fuma={f}")
print(f"BETA  : max|d|={max(dbeta):.3e}  mean|d|={sum(dbeta)/len(dbeta):.3e}  (n={len(dbeta)})")
print(f"P     : max|d|={max(dp):.3e}  mean|d|={sum(dp)/len(dp):.3e}  (n={len(dp)})")

# --- significance flips: Bonferroni across all, m = len(fuma) = 12531 ---
m = len(fuma)
both = oo = ff = 0
flips = []
for k in common:
    po = ours[k]["P"]
    if po is None: continue
    so = min(po * m, 1.0) < 0.05
    padj = num(fuma[k]["P.adj"])
    sf = (padj is not None and padj < 0.05)
    if so and sf: both += 1
    elif so and not sf: oo += 1; flips.append(("ours_only", k))
    elif sf and not so: ff += 1; flips.append(("fuma_only", k))
print(f"\nSignificance (Bonferroni m={m}, P.adj<0.05):")
print(f"   both={both}  ours_only={oo}  fuma_only={ff}  FLIPS={oo+ff}")
for tag, k in flips[:15]:
    print(f"   {tag}: {k[1]} @ {k[0]}")

ok_rows = len(ours) == 12531 and len(common) == 12531 and not (ok - fk) and not (fk - ok)
verdict = (ok_rows and len(ng_mismatch) == 0 and (oo + ff) == 0 and max(dbeta) < 1e-4)
print(f"\n{'='*40}")
print(f"VERDICT: {'PASS ✅ (reproduction exact)' if verdict else 'FAIL ❌ (see above)'}")
print(f"{'='*40}")
