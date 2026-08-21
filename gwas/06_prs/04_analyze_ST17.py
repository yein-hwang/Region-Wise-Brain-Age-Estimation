"""ST17: held-out UK Biobank validation of the regional BAG polygenic scores.

    OLS   measured_BAG_region ~ PRS_region        n = 10,267

The PRS is the sum of SCORE1_SUM over chromosomes 1-22, used raw: no sign flip,
no residualisation, no z-scaling. `test_R2` in the published table is the
ADJUSTED R2; `raw_R2` is also written out so the two are never confused.

    set -a; . gwas/config/paths.env; set +a
    python gwas/06_prs/04_analyze_ST17.py
"""
import os
import warnings

import pandas as pd
import statsmodels.formula.api as smf

from _regions import LABEL, REGIONS, require

warnings.filterwarnings("ignore")
PRS_WORK_DIR, PRS_UKB_PHENO = require("PRS_WORK_DIR", "PRS_UKB_PHENO")
BAG_COL = os.environ.get("PRS_UKB_BAG_COL", "{region}_corrected_delta_age_int")
OUT = os.environ.get("ST17_OUT", os.path.join(PRS_WORK_DIR, "tables", "ST17.tsv"))


def prs(region):
    """Sum the per-chromosome plink2 scores into one PRS column."""
    total = None
    for i in range(1, 23):
        s = pd.read_csv(f"{PRS_WORK_DIR}/scores_ukb/{region}/chr{i}.sscore",
                        sep="\t")[["IID", "SCORE1_SUM"]].rename(columns={"SCORE1_SUM": f"c{i}"})
        total = s if total is None else total.merge(s, on="IID", how="inner")
    return pd.DataFrame({"IID": total["IID"],
                         f"PRS_{region}": total[[f"c{i}" for i in range(1, 23)]].sum(axis=1)})


ph = pd.read_csv(PRS_UKB_PHENO, low_memory=False)
rows = []
for region in REGIONS:
    bag = BAG_COL.format(region=region)
    d = ph[["IID", bag]].merge(prs(region), on="IID", how="inner").dropna()
    fit = smf.ols(f"{bag} ~ PRS_{region}", data=d).fit()
    k = f"PRS_{region}"
    rows.append({"BAG phenotype": LABEL[region], "coef": fit.params[k], "se": fit.bse[k],
                 "pvalue": fit.pvalues[k], "test_R2": fit.rsquared_adj,
                 "raw_R2": fit.rsquared, "F_pvalue": fit.f_pvalue,
                 "AIC": fit.aic, "BIC": fit.bic, "N": int(fit.nobs)})
    print(f"  {LABEL[region]:16s} coef={fit.params[k]:+.6f}  adjR2={fit.rsquared_adj:.6f}  "
          f"p={fit.pvalues[k]:.3e}  N={int(fit.nobs):,}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
print(f"\nwrote {OUT}")
