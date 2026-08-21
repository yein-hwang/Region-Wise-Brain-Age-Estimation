#!/usr/bin/env python3
"""
Parallel port of FUMA magma_celltype.R step3 pairwise loop (L269-368) ONLY.

DOES NOT touch step1 or forward-selection (step2). Consumes R's already-computed:
  <jobdir>/step1_2_summary.txt      survivors = rows with step3==1 (cond_state has no "drop")
  <jobdir>/magma_celltype_step1.txt Marginal.P (= step1 P)         (R L365)
  <jobdir>/magma_celltype_step2.txt within-dataset pairs (if any)  (R L345-354)
  <jobdir>/magma.genes.raw
  covar files data/FUMA_scRNA_data_v2/celltype/<ds>.txt

Faithful to R L279-368, with two mechanical (value-identical) changes:
  - C(n,2) dataset pairs run in PARALLEL, each magma writing UNIQUE temp files.
  - Assembly is DETERMINISTIC: results keyed by (i,j) and reassembled in the exact R
    serial order (i<j outer/inner over step3_ds; cross-models in magma output order;
    within-dataset pairs appended after in step3_ds order) BEFORE MODEL is reindexed
    rep(1:(n/2),each=2) (R L355). CDM attached by explicit (ds,cell,cond_ds) join,
    which is value-identical to R's positional label match (L357-364).

Independent failure handling (verified against R L294/L312):
  - CD-marginal fails  (res>0, L295-297): pair contributes NO step3_avg -> CDM.* = NA.
  - CD-conditional fails(res>0, L313-322): NA rows for every cross-dataset covariate pair,
    order = for a in cells1: for b in cells2 (row1=ds1 cell, row2=ds2 cell).
  - the two magma calls are checked independently (all four success/fail combos possible).

Output: <jobdir>/magma_celltype_step3.txt (same columns as FUMA).
Usage:  run_step3_parallel.py <jobdir> [n_workers=32]
"""
import csv, os, sys, math, shutil, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

def _env(name):
    import os as _os, sys as _sys
    v = _os.environ.get(name)
    if not v:
        _sys.exit(f"ERROR: {name} is not set (see gwas/config/paths.env.example)")
    return v


ROOT   = _env("MAGMA_ROOT")
MAGMA  = f"{ROOT}/magma"
JOBDIR = sys.argv[1].rstrip("/")
NWORK  = int(sys.argv[2]) if len(sys.argv) > 2 else 32
GENESRAW = f"{JOBDIR}/magma.genes.raw"
TMP  = f"{JOBDIR}/step3_tmp"     # per-pair temp (each pair cleans its own files)
SLIM = f"{JOBDIR}/step3_slim"    # per-dataset slim covar: GENE + survivors + Average

def num(x):
    try: return float(x)
    except: return None

def split_ds_cell(full):        # R: ds = sub("(.+):.+","\\1"); cell = sub(".+:(.+)","\\1")  (greedy -> last ':')
    ds, _, cell = full.rpartition(":")
    return ds, cell

# ---- worker-level covar cache (module global; one per worker process) ----
_CACHE = {}
def read_slim(ds):
    v = _CACHE.get(ds)
    if v is None:
        with open(f"{SLIM}/{ds}.txt") as f:
            hdr = f.readline().split()
            rows = {}
            for line in f:
                p = line.split()
                if p: rows[p[0]] = p
        v = (hdr, rows); _CACHE[ds] = v
    return v

def read_covar_full(path):
    with open(path) as f:
        hdr = f.readline().split()
        rows = {}
        for line in f:
            p = line.split()
            if p: rows[p[0]] = p
    return hdr, rows

def parse_gsa(path):
    """yield dicts with full name (':'-joined), MODEL, NGENES, BETA, BETA_STD, SE, P, in file order."""
    with open(path) as f:
        cols = None; ci = None
        for line in f:
            if line.startswith("#"): continue
            p = line.split()
            if cols is None:
                cols = p; ci = {c: k for k, c in enumerate(cols)}; continue
            if len(p) < len(cols): continue
            if "FULL_NAME" in ci and len(p) > ci["FULL_NAME"]:
                full = " ".join(p[ci["FULL_NAME"]:])
            else:
                full = p[ci["VARIABLE"]]            # untruncated when no FULL_NAME column
            yield dict(full=full,
                       MODEL=p[ci["MODEL"]] if "MODEL" in ci else "1",
                       NGENES=p[ci["NGENES"]], BETA=p[ci["BETA"]],
                       BETA_STD=p[ci["BETA_STD"]], SE=p[ci["SE"]], P=p[ci["P"]])

# ------------------------------ per-pair worker ------------------------------
def process_pair(args):
    i, j, ds1, ds2, cells1, cells2 = args
    tag = f"{i}_{j}"
    exp = f"{TMP}/exp_{tag}.txt"; oavg = f"{TMP}/avg_{tag}"; ocond = f"{TMP}/cond_{tag}"
    h1, r1 = read_slim(ds1); h2, r2 = read_slim(ds2)
    n1, n2 = len(cells1), len(cells2)
    hdr = (["GENE"] + [f"{ds1}:{c}" for c in cells1] + ["Average1"]
                    + [f"{ds2}:{c}" for c in cells2] + ["Average2"])
    with open(exp, "w") as o:                       # R L279-290 (merge on exp1 GENE order, drop Average2 NA)
        o.write("\t".join(hdr) + "\n")
        for g, p1 in r1.items():
            p2 = r2.get(g)
            if p2 is None: continue
            o.write("\t".join([g] + p1[1:1+n1] + [p1[1+n1]] + p2[1:1+n2] + [p2[1+n2]]) + "\n")
    base = ["--gene-results", GENESRAW, "--gene-covar", exp, "max-miss=0.1",
            "--model", "condition-hide=Average1,Average2", "direction=greater"]

    # (a) CD-marginal  (R L291-307) -- independent failure check
    ra = subprocess.run([MAGMA] + base + ["--out", oavg],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    avg = []   # (ds, cell, cond_ds, CDM.BETA, CDM.BETA_STD, CDM.SE, CDM.P)
    if ra == 0 and os.path.exists(oavg + ".gsa.out"):
        for rec in parse_gsa(oavg + ".gsa.out"):
            ds, cell = split_ds_cell(rec["full"])
            cond = ds2 if ds == ds1 else ds1
            avg.append((ds, cell, cond, rec["BETA"], rec["BETA_STD"], rec["SE"], rec["P"]))

    # (b) CD-conditional joint-pairs (R L309-333) -- independent failure check
    rb = subprocess.run([MAGMA] + base + ["joint-pairs", "--out", ocond],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    cond = []  # ordered pairs; each: (ds, cell, cond_ds, NGENES, BETA, BETA_STD, SE, P)
    if rb == 0 and os.path.exists(ocond + ".gsa.out"):
        by_model = {}                                # insertion order = magma file/model order
        for rec in parse_gsa(ocond + ".gsa.out"):
            by_model.setdefault(rec["MODEL"], []).append(rec)
        for m, pr in by_model.items():               # keep cross-dataset models only (R L332-333)
            if len(pr) != 2: continue
            d0, _ = split_ds_cell(pr[0]["full"]); d1_, _ = split_ds_cell(pr[1]["full"])
            if d0 == d1_: continue
            for rec in pr:                           # magma row order within model preserved
                ds, cell = split_ds_cell(rec["full"])
                partner = d1_ if ds == d0 else d0
                cond.append((ds, cell, partner, rec["NGENES"], rec["BETA"],
                             rec["BETA_STD"], rec["SE"], rec["P"]))
    else:
        # FAILURE fallback (R L313-322 -> L330-333): NA rows for cross-dataset covariate pairs,
        # order = for a in cells1: for b in cells2 (row1=ds1 cell, row2=ds2 cell).
        for a in range(n1):
            for b in range(n2):
                cond.append((ds1, cells1[a], ds2, None, None, None, None, None))
                cond.append((ds2, cells2[b], ds1, None, None, None, None, None))

    for base_out in (oavg, ocond):
        for suf in (".gsa.out", ".log"):
            try: os.remove(base_out + suf)
            except OSError: pass
    try: os.remove(exp)
    except OSError: pass
    return i, j, avg, cond

# ------------------------------- inputs / assembly -------------------------------
def load_inputs():
    survivors = []
    with open(f"{JOBDIR}/step1_2_summary.txt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if str(r.get("step3", "0")).strip() == "1":
                survivors.append((r["Dataset"], r["Cell_type"]))
    ds_order, by_ds = [], {}
    for d, c in survivors:
        if d not in by_ds: by_ds[d] = []; ds_order.append(d)
        by_ds[d].append(c)
    marg = {}
    with open(f"{JOBDIR}/magma_celltype_step1.txt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            marg[(r["Dataset"], r["Cell_type"])] = num(r["P"])
    return ds_order, by_ds, marg

def fmt15(s):                   # R write.table writes doubles via as.character() = up to 15 sig figs.
    try: return "%.15g" % float(s)   # verified byte-identical to R's step3_exp.txt for the covar values.
    except (TypeError, ValueError): return s

def write_slim(ds_order, by_ds):
    os.makedirs(SLIM, exist_ok=True)
    COVDIR = f"{ROOT}/data/FUMA_scRNA_data_v2/celltype"
    for ds in ds_order:
        hdr, rows = read_covar_full(f"{COVDIR}/{ds}.txt")
        idx = {n: k for k, n in enumerate(hdr)}
        cols = ["GENE"] + by_ds[ds] + ["Average"]; ci = [idx[c] for c in cols]
        with open(f"{SLIM}/{ds}.txt", "w") as o:
            o.write("\t".join(cols) + "\n")
            for g, p in rows.items():           # GENE verbatim; expression values -> 15 sig figs (== R exp)
                o.write("\t".join([p[ci[0]]] + [fmt15(p[k]) for k in ci[1:]]) + "\n")

def within_dataset_pairs(ds_order, by_ds):
    """R L345-354: datasets with >=2 survivors contribute their step2 within-dataset pairs."""
    path = f"{JOBDIR}/magma_celltype_step2.txt"
    if not os.path.exists(path): return []
    surv = {d: set(cs) for d, cs in by_ds.items()}
    step2 = list(csv.DictReader(open(path), delimiter="\t"))
    out = []
    for ds in ds_order:                              # step3_ds order (R L345)
        if len(by_ds[ds]) < 2: continue
        rows = [r for r in step2 if r["Dataset"] == ds and r["Cell_type"] in surv[ds]]
        by_m = {}
        for r in rows: by_m.setdefault(r["MODEL"], []).append(r)
        for m, pr in by_m.items():
            if len(pr) == 2:
                for r in pr:                         # cond_ds=None -> CDM stays NA (within-dataset)
                    out.append((ds, r["Cell_type"], None, r["NGENES"], r["BETA"],
                                r["BETA_STD"], r["SE"], r["P"]))
    return out

def main():
    os.makedirs(TMP, exist_ok=True)
    ds_order, by_ds, marg = load_inputs()
    if len(ds_order) < 2:
        print("step3 skipped: survivors span <2 datasets"); return
    n = len(ds_order); npairs = n * (n - 1) // 2
    print(f"survivors: {sum(len(v) for v in by_ds.values())} cells / {n} datasets | "
          f"pairs={npairs} magma={npairs*2} workers={NWORK}", flush=True)
    write_slim(ds_order, by_ds)
    tasks = [(i, j, ds_order[i], ds_order[j], by_ds[ds_order[i]], by_ds[ds_order[j]])
             for i in range(n - 1) for j in range(i + 1, n)]
    results = {}; avg_all = []; done = 0
    with ProcessPoolExecutor(max_workers=NWORK) as ex:
        for fut in as_completed([ex.submit(process_pair, t) for t in tasks]):
            i, j, avg, cond = fut.result()
            results[(i, j)] = cond; avg_all.extend(avg); done += 1
            if done % 500 == 0: print(f"  {done}/{npairs} pairs", flush=True)
    # deterministic assembly in R serial order (i<j), then within-dataset pairs (R L341-354)
    cond_all = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            cond_all.extend(results[(i, j)])
    cond_all.extend(within_dataset_pairs(ds_order, by_ds))
    assert len(cond_all) % 2 == 0, f"odd cond_all: {len(cond_all)}"
    cdm = {(ds, cell, cond): (b, bs, se, p) for (ds, cell, cond, b, bs, se, p) in avg_all}

    cols = ["Dataset", "Cell_type", "MODEL", "NGENES", "BETA", "BETA_STD", "SE", "P",
            "CDM.BETA", "CDM.BETA_STD", "CDM.SE", "CDM.P", "CDM.ds",
            "Marginal.P", "PS", "PS.avg"]
    out_path = f"{JOBDIR}/magma_celltype_step3.txt"
    with open(out_path, "w") as o:
        o.write("\t".join(cols) + "\n")
        for k in range(0, len(cond_all), 2):         # MODEL reindex rep(1:(n/2),each=2)  (R L355)
            model_idx = k // 2 + 1
            for (ds, cell, cond, ng, b, bs, se, p) in cond_all[k:k+2]:
                cv = cdm.get((ds, cell, cond))
                cdmb, cdmbs, cdmse, cdmp = cv if cv else ("NA", "NA", "NA", "NA")
                mp = marg.get((ds, cell))
                P, CDMP = num(p), num(cdmp)
                denom = CDMP if CDMP is not None else mp                     # R L366 ifelse(is.na(CDM.P),Marginal.P,CDM.P)
                PS    = (math.log10(P) / math.log10(denom)) if (P and denom and P > 0 and denom > 0) else "NA"
                PSavg = (math.log10(CDMP) / math.log10(mp)) if (CDMP and mp and CDMP > 0 and mp > 0) else "NA"
                row = [ds, cell, model_idx, ng or "NA", b or "NA", bs or "NA", se or "NA", p or "NA",
                       cdmb, cdmbs, cdmse, cdmp, cond if cv else "NA",
                       mp if mp is not None else "NA", PS, PSavg]
                o.write("\t".join(str(x) for x in row) + "\n")
    shutil.rmtree(TMP, ignore_errors=True); shutil.rmtree(SLIM, ignore_errors=True)
    print(f"wrote {out_path}: {len(cond_all)} rows", flush=True)

if __name__ == "__main__":
    main()
