"""Build the ADNI regional brain-age GWAS phenotype table.

Replicates the UK Biobank phenotype construction exactly:
  reg: pred ~ Age   (fit on cognitively normal baseline)
  corrected_pred = (pred - b0) / b1                  # de Lange & Cole, division form
  corrected_delta = corrected_pred - true            # +ve = accelerated aging
  INT = norm.ppf((rank - 0.5)/n)                     # on the FINAL GWAS sample

Procedure:
  1. baseline only (is_baseline == True), 1 row/subject (dedup to earliest Acq Date)
  2. CN bias correction: fit slope/intercept on Group == 'CN' baseline, apply to ALL baseline
  3. restrict to the final GWAS sample (baseline subjects present in ADNI_KEEP_IIDS)
  4. INT per region on that final sample -> {region}_corrected_delta_age_int
  5. merge covariates: Sex (genomic), Age, PC1-10, genotyping-batch dummies

The bias fit and the INT are per region, so adding a region leaves the other
columns bit-identical.

Configuration comes from gwas/config/paths.env:
    set -a; . gwas/config/paths.env; set +a
    python gwas/02_adni_replication/build_pheno.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.linear_model import LinearRegression


def env(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"ERROR: {name} is not set (see gwas/config/paths.env.example)")
    return v


# Region key as it appears in the prediction CSV filename -> phenotype column
# prefix. The whole-brain model's predictions are filed under a different key
# than its phenotype column; ADNI_GLOBAL_PRED_KEY names it.
GLOBAL_PRED_KEY = os.environ.get('ADNI_GLOBAL_PRED_KEY', 'imgs')
REGIONS = os.environ.get(
    'ADNI_REGIONS',
    f'{GLOBAL_PRED_KEY},caudate,cerebellum,frontal_lobe,insula,'
    'occipital_lobe,parietal_lobe,putamen,temporal_lobe,thalamus,hippocampus'
).split(',')
REGION_TO_COL = {r: ('global' if r == GLOBAL_PRED_KEY else r) for r in REGIONS}

CSV_TMPL = env('ADNI_PRED_TMPL')          # must contain {region}
KEEP = env('ADNI_KEEP_IIDS')              # one IID per line, the final QC sample
SEXF = env('ADNI_SAMPLE_TSV')             # IID + SEX (genomic; 1=M, 2=F)
EIGV = env('ADNI_EIGENVEC')               # plink2 .eigenvec, #IID + PC1..
COHO = env('ADNI_COHORT_TSV')             # IID + genotyping-batch dummies
OUT = env('ADNI_PHENO_FILE')
COHORT_DUMMIES = os.environ.get(
    'ADNI_EXTRA_COVARS', 'is_GO2_set1,is_GO2_set2,is_ADNI3_set1,is_ADNI3_set2'
).split(',')


def ranked_int(values):
    values = np.asarray(values, dtype=float)
    ranks = rankdata(values, method='average')
    return norm.ppf((ranks - 0.5) / len(values))


def cn_bias_fit(true_cn, pred_cn):
    reg = LinearRegression().fit(true_cn.reshape(-1, 1), pred_cn)
    return float(reg.intercept_), float(reg.coef_[0])


def load_region_baseline(region):
    """Return baseline-only df (1 row/subject) with Subject, Age, Sex, Group, pred."""
    df = pd.read_csv(CSV_TMPL.format(region=region))
    bl = df[df['is_baseline'] == True].copy()
    # dedup: subject with >1 baseline scan -> earliest Acq Date
    n_before = len(bl)
    if bl['Subject'].duplicated().any():
        bl['_acq'] = pd.to_datetime(bl['Acq Date'], errors='coerce')
        bl = bl.sort_values(['Subject', '_acq']).drop_duplicates('Subject', keep='first')
        bl = bl.drop(columns='_acq')
    n_after = len(bl)
    if n_before != n_after:
        print(f'  [{region}] baseline dedup: {n_before} -> {n_after} rows')
    return bl[['Subject', 'Age', 'Sex', 'Group', 'predicted_brain_age']].rename(
        columns={'predicted_brain_age': 'pred'})


def main():
    print('=== Step 1: load region baselines + CN bias correction ===')
    # anchor (Subject, Age, Sex, Group) from imgs; verify alignment across regions
    base = None
    corrected = {}   # region_col -> pd.Series indexed by Subject (corrected_delta over ALL baseline)
    cn_params = {}

    for region in REGIONS:
        col = REGION_TO_COL[region]
        bl = load_region_baseline(region).set_index('Subject')
        true = bl['Age'].to_numpy(dtype=float)
        pred = bl['pred'].to_numpy(dtype=float)

        cn_mask = (bl['Group'] == 'CN').to_numpy()
        b0, b1 = cn_bias_fit(true[cn_mask], pred[cn_mask])
        corrected_pred = (pred - b0) / b1
        corrected_delta = corrected_pred - true          # +ve = accelerated aging
        corrected[col] = pd.Series(corrected_delta, index=bl.index)
        cn_params[col] = (b0, b1, int(cn_mask.sum()), len(bl))
        print(f'  [{col:14s}] n_baseline={len(bl):4d} n_CN={cn_mask.sum():4d} '
              f'b0={b0:8.4f} b1={b1:7.4f} delta(mean={corrected_delta.mean():.3f},sd={corrected_delta.std():.3f})')

        if base is None:
            base = bl[['Age', 'Sex', 'Group']].copy()
        else:
            # alignment check: same Age for shared subjects
            shared = base.index.intersection(bl.index)
            if not np.allclose(base.loc[shared, 'Age'], bl.loc[shared, 'Age']):
                raise SystemExit(f'AGE MISMATCH across region files at {region}')

    # assemble all-baseline corrected_delta frame
    delta_df = pd.DataFrame(corrected)
    print(f'\nbaseline subjects (union): {len(delta_df)}; '
          f'all regions same subject set: {all(corrected[c].index.equals(delta_df.index) for c in corrected)}')

    print('\n=== Step 2: restrict to final GWAS sample (baseline ∩ 1856) ===')
    keep = pd.read_csv(KEEP, header=None)[0].astype(str).tolist()
    keep_set = set(keep)
    final_ids = [s for s in delta_df.index if s in keep_set]
    n_final = len(final_ids)
    print(f'  final_qc_keep.iids: {len(keep)}; baseline: {len(delta_df)}; '
          f'intersection (GWAS N): {n_final}')
    missing_geno = sorted(keep_set - set(delta_df.index))
    print(f'  genotyped-but-no-baseline-phenotype: {len(missing_geno)}'
          + (f' e.g. {missing_geno[:5]}' if missing_geno else ''))

    fin = delta_df.loc[final_ids].copy()
    base_f = base.loc[final_ids]

    print('\n=== Step 3: INT per region on final sample ===')
    out = pd.DataFrame(index=final_ids)
    for col in REGION_TO_COL.values():
        out[f'{col}_corrected_delta_age_int'] = ranked_int(fin[col].to_numpy())

    print('\n=== Step 4: merge covariates ===')
    # genomic sex (authoritative): 1=M, 2=F
    sex = pd.read_csv(SEXF, sep='\t').set_index('IID')['SEX']
    # PCs
    pcs = pd.read_csv(EIGV, sep='\t').rename(columns={'#IID': 'IID'}).set_index('IID')
    pc_cols = [f'PC{i}' for i in range(1, 11)]   # PC1-10 per spec
    # cohort dummies (ADNI1 = ref)
    coh = pd.read_csv(COHO, sep='\t').set_index('IID')
    dummy_cols = COHORT_DUMMIES

    res = pd.DataFrame(index=final_ids)
    res.index.name = 'IID'
    res['Age'] = base_f['Age'].astype(int)
    res['Sex'] = sex.reindex(final_ids).astype('Int64')
    for c in pc_cols:
        res[c] = pcs[c].reindex(final_ids)
    for c in dummy_cols:
        res[c] = coh[c].reindex(final_ids).astype('Int64')
    res = res.join(out)

    # sanity: no missing covariates
    miss = res[['Age', 'Sex'] + pc_cols + dummy_cols].isna().sum()
    if miss.any():
        print('  WARNING missing covariates:\n', miss[miss > 0])
    else:
        print('  all covariates complete (Age, Sex, PC1-10, 4 cohort dummies)')

    res = res.reset_index()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f'\n=== SAVED {OUT} ===')
    print(f'  N = {len(res)}')
    print(f'  Sex (1=M,2=F): {res.Sex.value_counts().to_dict()}')
    print(f'  cols: {list(res.columns)}')
    print('\n  CN-INT sanity (should be ~N(0,1)):')
    for col in REGION_TO_COL.values():
        v = res[f'{col}_corrected_delta_age_int']
        print(f'    {col:14s} mean={v.mean():+.3f} sd={v.std():.3f} '
              f'min={v.min():.2f} max={v.max():.2f}')


if __name__ == '__main__':
    main()
