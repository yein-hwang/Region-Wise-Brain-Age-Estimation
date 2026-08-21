"""Per-region genomic inflation (lambda_GC) and QQ plots from SAIGE step-2 output.

lambda_GC = median(chi2_obs) / qchisq(0.5,1)   with chi2_obs = chi2.isf(p, df=1)
0.4549364 = scipy.stats.chi2.ppf(0.5, 1)  (median of 1-df chi-square).
Also reports lambda based on median(p) as a cross-check.

Reads {GWAS_DIR}/{region}/results/chr{1..22}.txt and writes lambda_gc_summary.csv
plus a QQ panel under {GWAS_DIR}/postgwas/.

    set -a; . gwas/config/paths.env; set +a
    GWAS_DIR=$GWAS_ADNI_DIR PLOT_SAMPLE_N=N=1693 \
    PLOT_TITLE="ADNI regional brain-age GWAS (N=1,693)" \
        python gwas/03_postgwas/lambda_qq.py
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("GWAS_DIR") or os.environ.get("GWAS_ADNI_DIR")
if not BASE:
    raise SystemExit("ERROR: set GWAS_DIR (or GWAS_ADNI_DIR) -- see gwas/config/paths.env.example")
OUT  = os.environ.get("POSTGWAS_OUT_DIR", f"{BASE}/postgwas")
REGIONS = os.environ.get("POSTGWAS_REGIONS",
    "global,caudate,cerebellum,frontal_lobe,insula,"
    "occipital_lobe,parietal_lobe,putamen,temporal_lobe,thalamus").split(",")
SAMPLE_N = os.environ.get("PLOT_SAMPLE_N", "")          # e.g. "N=1693"; blank to omit
PLOT_TITLE = os.environ.get("PLOT_TITLE", "Regional brain-age GWAS")
MEDIAN_CHISQ_1DF = stats.chi2.ppf(0.5, 1)   # 0.45493642

def load_pvals(region):
    """Concatenate p.value across chr1-22 for a region. Returns float array (NaN dropped)."""
    parts = []
    for c in range(1, 23):
        f = f"{BASE}/{region}/results/chr{c}.txt"
        df = pd.read_csv(f, sep="\t", usecols=["p.value"], dtype={"p.value": float},
                         na_values=["NA","nan",""], engine="c")
        parts.append(df["p.value"].to_numpy())
    p = np.concatenate(parts)
    p = p[np.isfinite(p)]
    p = p[(p > 0) & (p <= 1)]
    return p

def lambda_gc(p):
    chisq = stats.chi2.isf(p, df=1)          # observed 1-df chi-square from p
    lam_chi = np.median(chisq) / MEDIAN_CHISQ_1DF
    lam_medp = stats.chi2.isf(np.median(p), df=1) / MEDIAN_CHISQ_1DF
    return lam_chi, lam_medp

def qq_plot(p, region, lam, ax):
    n = len(p)
    obs = -np.log10(np.sort(p))                      # ascending -logp
    exp = -np.log10((np.arange(1, n + 1) - 0.5) / n) # expected uniform
    # thin dense bulk for file size: keep all obs>2 + 20k sample of the rest
    keep = obs > 2
    rng = np.random.default_rng(7)
    bulk = np.where(~keep)[0]
    samp = rng.choice(bulk, size=min(len(bulk), 20000), replace=False)
    idx = np.sort(np.concatenate([np.where(keep)[0], samp]))
    ax.scatter(exp[idx], obs[idx], s=4, color="#2c3e50", alpha=0.6, rasterized=True)
    lim = max(exp.max(), obs.max()) * 1.02
    ax.plot([0, lim], [0, lim], color="red", lw=1)
    ci = 0.95
    mm = np.arange(1, n + 1)
    c_lo = -np.log10(stats.beta.ppf((1-ci)/2, mm, n - mm + 1))
    c_hi = -np.log10(stats.beta.ppf(1-(1-ci)/2, mm, n - mm + 1))
    ax.fill_between(exp, c_lo, c_hi, color="grey", alpha=0.18, lw=0)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    ax.set_title(f"{region}\n$\\lambda_{{GC}}$={lam:.3f}  "
                 f"({SAMPLE_N + ', ' if SAMPLE_N else ''}{n:,} SNP)", fontsize=10)

rows = []
ncol = int(os.environ.get("PLOT_NCOL", 5))
nrow = -(-len(REGIONS) // ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.5 * nrow))
axes = axes.ravel()
for i, r in enumerate(REGIONS):
    p = load_pvals(r)
    lam_chi, lam_medp = lambda_gc(p)
    nsig5e8 = int((p < 5e-8).sum())
    nsig1e5 = int((p < 1e-5).sum())
    rows.append(dict(region=r, n_snp=len(p), lambda_GC=round(lam_chi,4),
                     lambda_medp=round(lam_medp,4),
                     n_5e_8=nsig5e8, n_1e_5=nsig1e5, min_p=float(p.min())))
    qq_plot(p, r, lam_chi, axes[i])
    print(f"{r:15s} lambda={lam_chi:.4f} (medp {lam_medp:.4f})  "
          f"5e-8:{nsig5e8}  1e-5:{nsig1e5}  minp:{p.min():.2e}  n={len(p):,}", flush=True)

fig.suptitle(f"{PLOT_TITLE} — QQ plots", fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT}/qq/qq_all_regions.png", dpi=130, bbox_inches="tight")
fig.savefig(f"{OUT}/qq/qq_all_regions.pdf", bbox_inches="tight")
plt.close(fig)

summ = pd.DataFrame(rows)
summ.to_csv(f"{OUT}/lambda_gc_summary.csv", index=False)
print("\n=== lambda_GC summary ===")
print(summ.to_string(index=False))
print(f"\nsaved: {OUT}/lambda_gc_summary.csv ; {OUT}/qq/qq_all_regions.png")
