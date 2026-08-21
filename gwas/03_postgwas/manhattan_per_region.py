"""Per-region Manhattan plots from SAIGE per-chromosome output (chr1..22.txt).

Output: {MANHATTAN_OUT_DIR}/manhattan_all_regions.png   (one row per region)
        {MANHATTAN_OUT_DIR}/manhattan_<region>.png       (per-region, full-res)

    set -a; . gwas/config/paths.env; set +a
    GWAS_DIR=$GWAS_ADNI_DIR PLOT_TITLE="ADNI regional brain-age GWAS (N=1,693)" \
        python gwas/03_postgwas/manhattan_per_region.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("GWAS_DIR") or os.environ.get("GWAS_ADNI_DIR")
if not BASE:
    raise SystemExit("ERROR: set GWAS_DIR (or GWAS_ADNI_DIR) -- see gwas/config/paths.env.example")
OUT  = os.environ.get("MANHATTAN_OUT_DIR", f"{BASE}/postgwas/manhattan")
REGIONS = os.environ.get("POSTGWAS_REGIONS",
    "global,caudate,cerebellum,frontal_lobe,insula,"
    "occipital_lobe,parietal_lobe,putamen,temporal_lobe,thalamus").split(",")
PLOT_TITLE = os.environ.get("PLOT_TITLE", "Regional brain-age GWAS")

# Display names come from the shared region table.
_REG_TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "regions.tsv")
LABEL = {}
with open(_REG_TSV) as _f:
    for _line in _f:
        if _line.startswith("#"):
            continue
        _p = _line.rstrip("\n").split("\t")
        if len(_p) >= 2:
            LABEL[_p[0]] = _p[1]

GW, SUG = 5e-8, 1e-5
CHR_COLORS = ["#3b5b78", "#9ab3c9"]   # alternating dark/light

def load_region(region):
    """Return DataFrame CHR,POS,P sorted by genomic position; thinned for plotting."""
    parts = []
    for c in range(1, 23):
        f = f"{BASE}/{region}/results/chr{c}.txt"
        df = pd.read_csv(f, sep="\t", usecols=["CHR","POS","p.value"],
                         dtype={"CHR":int,"POS":int,"p.value":float}, engine="c")
        parts.append(df)
    d = pd.concat(parts, ignore_index=True)
    d = d[(d["p.value"] > 0) & (d["p.value"] <= 1)].copy()
    d["logp"] = -np.log10(d["p.value"])
    # thin: keep all logp>2 (p<1e-2) + 3% random sample of the rest (file size)
    rng = np.random.default_rng(7)
    big = d["logp"] > 2
    small_idx = d.index[~big].to_numpy()
    samp = rng.choice(small_idx, size=min(len(small_idx), 120000), replace=False)
    d = pd.concat([d[big], d.loc[samp]]).sort_values(["CHR","POS"])
    return d

def cumulative_pos(d):
    """Add cumulative x position; return (d, ticks, labels)."""
    d = d.sort_values(["CHR","POS"]).reset_index(drop=True)
    offset = 0; ticks=[]; labs=[]; xpos = np.empty(len(d))
    for c in range(1, 23):
        m = (d["CHR"] == c).to_numpy()
        if not m.any():
            continue
        pos = d.loc[m, "POS"].to_numpy()
        xpos[m] = pos + offset
        ticks.append(offset + (pos.min() + pos.max())/2)
        labs.append(str(c))
        offset += pos.max() + 2e7   # gap between chromosomes
    d["x"] = xpos
    return d, ticks, labs

def draw(ax, d, ticks, labs, title):
    for i, c in enumerate(range(1, 23)):
        sub = d[d["CHR"] == c]
        ax.scatter(sub["x"], sub["logp"], s=3, color=CHR_COLORS[i % 2],
                   alpha=0.6, rasterized=True, linewidths=0)
    ax.axhline(-np.log10(GW), color="red", ls="--", lw=0.8)
    ax.axhline(-np.log10(SUG), color="orange", ls="--", lw=0.8)
    ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=6)
    ymax = max(d["logp"].max()*1.05, -np.log10(GW)+0.5)
    ax.set_ylim(0, ymax)
    ax.set_xlim(d["x"].min()-1e7, d["x"].max()+1e7)
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=8)
    ax.set_title(title, fontsize=10)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)

fig, axes = plt.subplots(len(REGIONS), 1, figsize=(14, 2.6 * len(REGIONS)))
for ax, r in zip(axes, REGIONS):
    d = load_region(r)
    d, ticks, labs = cumulative_pos(d)
    n_gw = int((d["p.value"] < GW).sum()); n_sug = int((d["p.value"] < SUG).sum())
    draw(ax, d, ticks, labs, f"{LABEL[r]}  (min p={d['p.value'].min():.1e}; "
                             f"5e-8:{n_gw}, 1e-5:{n_sug})")
    print(f"{r:15s} minp={d['p.value'].min():.2e}  5e-8:{n_gw}  1e-5:{n_sug}", flush=True)
axes[-1].set_xlabel("Chromosome", fontsize=10)
fig.suptitle(f"{PLOT_TITLE} — Manhattan", fontsize=14, y=1.001)
fig.tight_layout()
fig.savefig(f"{OUT}/manhattan_all_regions.png", dpi=130, bbox_inches="tight")
fig.savefig(f"{OUT}/manhattan_all_regions.pdf", bbox_inches="tight")
plt.close(fig)
print(f"\nsaved: {OUT}/manhattan_all_regions.png")
