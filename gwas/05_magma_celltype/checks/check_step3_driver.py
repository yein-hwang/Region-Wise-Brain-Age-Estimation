#!/usr/bin/env python3
"""Compare driver-produced magma_celltype_step3.txt vs R-produced reference (.R.txt).

Key = (Dataset, Cell_type, CDM.ds)  -- MODEL is NOT in the key (it is a reindexed value,
using it as key would be circular). Duplicate keys (e.g. all-NA failure-fallback rows whose
CDM.ds='NA') are handled by comparing the two sides' rows for that key as an order-independent
multiset, sorted by their value tuple. MODEL is compared separately (per-key multiset).

Usage: compare_step3_driver.py <jobdir>
"""
import csv, sys
from collections import defaultdict

JOB = sys.argv[1].rstrip("/")
DRV = f"{JOB}/magma_celltype_step3.txt"
REF = f"{JOB}/magma_celltype_step3.R.txt"
NUMCOLS = ["NGENES","BETA","BETA_STD","SE","P","CDM.BETA","CDM.BETA_STD","CDM.SE","CDM.P",
           "Marginal.P","PS","PS.avg"]

def load(p):
    with open(p) as f: return list(csv.DictReader(f, delimiter="\t"))
def num(x):
    try: return float(x)
    except: return None
def isna(x): return x in ("NA", "", None)
def key(r): return (r["Dataset"], r["Cell_type"], r["CDM.ds"])
def valtuple(r): return tuple(r.get(c, "NA") for c in NUMCOLS)   # canonical sort/compare key

drv, ref = load(DRV), load(REF)
dg, rg = defaultdict(list), defaultdict(list)
for r in drv: dg[key(r)].append(r)
for r in ref: rg[key(r)].append(r)
dks, rks = set(dg), set(rg)

print(f"=== {JOB.split('/')[-1]} ===")
print(f"driver rows={len(drv)}  R rows={len(ref)}")
print(f"keys: driver={len(dks)} R={len(rks)} common={len(dks&rks)} only_drv={len(dks-rks)} only_R={len(rks-dks)}")
for k in list(dks-rks)[:6]: print("   only_drv:", k)
for k in list(rks-dks)[:6]: print("   only_R  :", k)

passed = (len(drv)==len(ref) and not (dks-rks) and not (rks-dks))
maxd = {c: 0.0 for c in NUMCOLS}; namis = {c: 0 for c in NUMCOLS}; worst = {}
count_mis = 0; model_mis = 0
for k in dks & rks:
    dl = sorted(dg[k], key=valtuple); rl = sorted(rg[k], key=valtuple)
    if len(dl) != len(rl):
        count_mis += 1; passed = False; continue
    for a, b in zip(dl, rl):                       # element-wise within the sorted multiset
        for c in NUMCOLS:
            av, bv = a.get(c), b.get(c)
            if isna(av) != isna(bv): namis[c] += 1; continue
            x, y = num(av), num(bv)
            if x is None or y is None: continue
            d = abs(x - y)
            if d > maxd[c]: maxd[c] = d; worst[c] = k
    # MODEL: per-key multiset (order-independent), compared separately
    if sorted(x["MODEL"] for x in dg[k]) != sorted(x["MODEL"] for x in rg[k]):
        model_mis += 1

for c in NUMCOLS:
    flag = "" if (maxd[c] < 1e-6 and namis[c] == 0) else "  <== MISMATCH"
    extra = f" worst@{worst[c]}" if maxd[c] >= 1e-6 else ""
    print(f"  [{c:13s}] max|diff|={maxd[c]:.2e}  NA-mismatch={namis[c]}{flag}{extra}")
    if maxd[c] >= 1e-6 or namis[c] > 0: passed = False
print(f"  [row-count-per-key mismatches] {count_mis}" + ("" if count_mis==0 else "  <== MISMATCH"))
print(f"  [MODEL per-key multiset mismatches] {model_mis}" + ("" if model_mis==0 else "  <== MODEL differs"))
if count_mis: passed = False

verdict = "ALL MATCH (incl MODEL) ✅" if (passed and model_mis==0) else \
          ("VALUES MATCH, MODEL differs ⚠️" if passed else "FAIL ❌")
print(f"VERDICT: {verdict}")
