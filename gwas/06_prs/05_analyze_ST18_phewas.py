"""ST18: PheWAS of the regional BAG polygenic scores over UK Biobank non-imaging
phenotypes (n = 446,342).

    Logit(phecode ~ PRS_region + Age + Sex + PC1..PC10)

Phecodes are aggregated to one decimal place (a case for any child code is a
case for the parent) and kept when they have at least 30 cases. A NaN in the
phecode table is the phecode exclusion range -- neither case nor control -- so
those subjects are dropped from that phecode's test rather than recoded as
controls. Benjamini-Hochberg FDR is applied within each region, not across
regions.

The PRS is raw SCORE1_SUM summed over chromosomes 1-22, with no sign flip.

    set -a; . gwas/config/paths.env; set +a
    python gwas/06_prs/05_analyze_ST18_phewas.py
"""
import os
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from _regions import LABEL, REGIONS, require

warnings.filterwarnings("ignore")
PRS_WORK_DIR, PHEWAS_TABLE, PHECODE_CATALOG = require(
    "PRS_WORK_DIR", "PRS_PHEWAS_TABLE", "PRS_PHECODE_CATALOG")
OUT_DIR = os.environ.get("ST18_OUT_DIR", os.path.join(PRS_WORK_DIR, "tables"))
NPROC = int(os.environ.get("PRS_NPROC", "24"))
COVS = ["Age", "Sex"] + [f"PC{i}" for i in range(1, 11)]


def prs(region):
    total = None
    for i in range(1, 23):
        s = pd.read_csv(f"{PRS_WORK_DIR}/scores_ukb/{region}/chr{i}.sscore",
                        sep="\t")[["IID", "SCORE1_SUM"]].rename(columns={"SCORE1_SUM": f"c{i}"})
        total = s if total is None else total.merge(s, on="IID", how="inner")
    return pd.DataFrame({"eid": total["IID"],
                         f"PRS_{region}": total[[f"c{i}" for i in range(1, 23)]].sum(axis=1)})


print("loading the phenotype table ...", flush=True)
df = pd.read_csv(PHEWAS_TABLE, low_memory=False)
print(f"  shape {df.shape}", flush=True)
raw = [c for c in df.columns if c.startswith("XX")]
for c in raw:
    # NaN = phecode exclusion range; kept as NaN and dropped per test below
    df[c] = df[c].astype(np.float32)

for region in REGIONS:
    p = prs(region)
    if f"PRS_{region}" in df.columns:
        df = df.drop(columns=[f"PRS_{region}"])
    df = df.merge(p, on="eid", how="left")
    print(f"  PRS_{region}: matched {df[f'PRS_{region}'].notna().sum():,}/{len(df):,}", flush=True)

# --- phecodes to one decimal, minimum 30 cases ----------------------------
mapping = {}
for col in raw:
    s = col.replace("XX", "")
    mapping[col] = f"XX{s.split('.')[0]}.{s.split('.')[1][:1]}" if "." in s else col
rev = {}
for old, new in mapping.items():
    rev.setdefault(new, []).append(old)
sub = df[raw]
agg = pd.DataFrame({k: (sub[v[0]] if len(v) == 1 else sub[v].max(axis=1))
                    for k, v in rev.items()}, index=df.index)
keep = [c for c in agg.columns if agg[c].sum() >= 30]
print(f"phecodes: {len(raw)} raw -> {len(rev)} aggregated -> {len(keep)} with >=30 cases", flush=True)

X_cov = df[COVS].to_numpy(np.float64)
PRS = {r: df[f"PRS_{r}"].to_numpy(np.float64) for r in REGIONS}
Y = {c: agg[c].to_numpy(np.float32) for c in keep}
del df, sub, agg


def bh_fdr(s):
    """Benjamini-Hochberg within one region, NaN-safe.

    statsmodels' multipletests propagates a single NaN across its whole input,
    so one unusable p-value blanks out every FDR in the group. Non-finite
    p-values are excluded from the BH input -- and therefore from the m used in
    the correction -- and come back as NaN. one() already drops such rows, so
    this is a backstop that keeps the remaining tests correct if one slips
    through.
    """
    s = pd.Series(s)
    ok = np.isfinite(s.to_numpy(dtype=float))
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if ok.any():
        out.loc[s.index[ok]] = multipletests(s[ok], method="fdr_bh")[1]
    return out


def one(task):
    region, pc = task
    y, x = Y[pc], PRS[region]
    ok = np.isfinite(x) & np.isfinite(X_cov).all(axis=1) & (~np.isnan(y))
    y2 = y[ok].astype(np.int8)
    if y2.sum() < 30 or len(np.unique(y2)) < 2:
        return None
    X = np.column_stack([np.ones(ok.sum()), x[ok], X_cov[ok]])
    try:
        m = sm.Logit(y2, X).fit(disp=0, maxiter=50)
        if not m.mle_retvals["converged"]:
            return None
        # A rank-deficient fit can still report converged=True: a sex-specific
        # phecode is quasi-completely separated by the Sex covariate, the
        # observed information matrix loses rank, and its inverse gives
        # bse=NaN -> p_value=NaN. Such a row carries no inference and, left in,
        # blanks out every FDR in the region's BH group. Drop it at the source.
        if not np.isfinite(m.params[1]) or not np.isfinite(m.pvalues[1]):
            return None
        return {"Region": LABEL[region], "phecode": pc, "coef": m.params[1],
                "odds_ratio": float(np.exp(m.params[1])), "p_value": m.pvalues[1],
                "n_case": int(y2.sum()), "n_total": int(len(y2))}
    except Exception:
        return None


tasks = [(r, c) for r in REGIONS for c in keep]
print(f"running {len(tasks):,} logistic regressions ...", flush=True)
with Pool(NPROC) as pool:
    res = [r for r in pool.imap_unordered(one, tasks, chunksize=8) if r is not None]
out = pd.DataFrame(res)
print(f"  converged: {len(out):,}/{len(tasks):,}", flush=True)

cat = pd.read_csv(PHECODE_CATALOG)
cat["Phecode_str"] = cat["Phecode"].apply(
    lambda x: f"{x:.2f}".rstrip('0').rstrip('.') if '.' in f"{x:.2f}" else str(int(x)))
cat["phecode"] = "XX" + cat["Phecode_str"]
out = out.merge(cat[["phecode", "Phenotype", "Category"]].drop_duplicates("phecode"),
                on="phecode", how="left")
# Pool.imap_unordered returns results in completion order, so fix the row order
# before anything depends on it. BH is computed per region and is order-
# independent; the final sort is stable, so rows tied on (fdr, p_value) come out
# in (Region, phecode) order rather than in whatever order the workers finished.
out = out.sort_values(["Region", "phecode"]).reset_index(drop=True)
out["fdr"] = out.groupby("Region")["p_value"].transform(bh_fdr)
out["phecode"] = out["phecode"].str.replace("^XX", "", regex=True)
out = out.sort_values(["fdr", "p_value"], kind="mergesort")

os.makedirs(OUT_DIR, exist_ok=True)
out.to_csv(f"{OUT_DIR}/ST18_full.tsv", sep="\t", index=False)
sig = out[out.fdr < 0.05]
sig.to_csv(f"{OUT_DIR}/ST18.tsv", sep="\t", index=False)
print(f"\n=== ST18 ===")
print(f"  tests            : {len(out):,}")
print(f"  FDR<0.05 rows    : {len(sig)}")
print(f"  distinct phecodes: {sig.phecode.nunique()}")
print(f"  positive coef    : {(sig.coef > 0).sum()}/{len(sig)}")
print(f"wrote {OUT_DIR}/ST18.tsv and ST18_full.tsv")
