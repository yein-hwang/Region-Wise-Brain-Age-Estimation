"""10-region UKB->ADNI sign-concordance, corrected pipeline.

Same method as concordance_audit_global.py, applied to all 10 regions:
  - load UKB (N=41,067) + ADNI (N=1,693) regional sumstats
  - harmonize to UKB effect allele (Allele2 both): direct/swap/strand-flip/+swap,
    palindromes resolved by AF else dropped
  - COMMON-set distance-pruning (lead = lowest-UKB-p SNP that is present in ADNI)
    -> this is what recovers loci whose UKB top tag-SNP is missing from ADNI's panel
  - sign concordance + binomial(0.5), for gw5e8 and sug1e5 hit sets
  - per-region lead/locus recovery diagnostic: is the UKB top lead in ADNI? if not,
    what common-set SNP recovers the locus and is it concordant?

Outputs: allregions_concordance_table.csv  (the unified slide-replacement table)
         allregions_lead_recovery.csv       (per-region top-lead diagnostic)
Ladder + beta-beta are NOT repeated here (global is the representative; see global script).
"""
import os
import numpy as np, pandas as pd, bisect
from scipy import stats


def env(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"ERROR: {name} is not set (see gwas/config/paths.env.example)")
    return v


UKB=env('GWAS_UKB_DIR'); ADNI=env('GWAS_ADNI_DIR')
OUT=os.environ.get('CONCORDANCE_OUT_DIR', os.path.join(ADNI,'postgwas','concordance'))
os.makedirs(OUT, exist_ok=True)
REGIONS=os.environ.get('CONCORDANCE_REGIONS',
    "global,caudate,cerebellum,frontal_lobe,insula,"
    "occipital_lobe,parietal_lobe,putamen,temporal_lobe,thalamus").split(',')
COMP={'A':'T','T':'A','C':'G','G':'C'}; AMB={frozenset(['A','T']),frozenset(['C','G'])}
GW,SUG=5e-8,1e-5; WIN=500_000
cols=['CHR','POS','MarkerID','Allele1','Allele2','AF_Allele2','BETA','SE','p.value']

def load_adni(region):
    fr=[pd.read_csv(f"{ADNI}/{region}/results/chr{c}.txt",sep='\t',usecols=cols,
        dtype={'CHR':int,'POS':int},engine='c') for c in range(1,23)]
    a=pd.concat(fr,ignore_index=True)
    return a.rename(columns={'Allele1':'a1','Allele2':'a2','AF_Allele2':'af_a2',
        'BETA':'b_ad','SE':'se_ad','p.value':'p_ad','MarkerID':'id_ad'})

def load_ukb(region):
    u=pd.read_csv(f"{UKB}/{region}/results/{region}_imputed_sumstats.txt",sep='\t',
        usecols=cols,dtype={'POS':int},engine='c')
    u['CHR']=pd.to_numeric(u['CHR'],errors='coerce'); u=u.dropna(subset=['CHR'])
    u['CHR']=u['CHR'].astype(int); u=u[(u['CHR']>=1)&(u['CHR']<=22)]
    return u.rename(columns={'Allele1':'u1','Allele2':'u2','AF_Allele2':'af_u2',
        'BETA':'b_uk','SE':'se_uk','p.value':'p_uk','MarkerID':'id_uk'})

def snp_mask(df,c1,c2): return df[c1].isin(list('ACGT'))&df[c2].isin(list('ACGT'))
def drop_multi(df):
    k=df['CHR'].astype(np.int64)*10**9+df['POS'].astype(np.int64)
    return df[~k.duplicated(keep=False)].copy()

def harmonize(ukb,adni):
    ukb=drop_multi(ukb[snp_mask(ukb,'u1','u2')].copy())
    adni=drop_multi(adni[snp_mask(adni,'a1','a2')].copy())
    m=ukb.merge(adni,on=['CHR','POS'],how='inner')
    u1,u2,a1,a2=m['u1'].values,m['u2'].values,m['a1'].values,m['a2'].values
    cmap=np.vectorize(COMP.get); cu1,cu2=cmap(u1),cmap(u2)
    direct=(a1==u1)&(a2==u2); swap=(a1==u2)&(a2==u1)
    flip=(a1==cu1)&(a2==cu2); flipswap=(a1==cu2)&(a2==cu1)
    amb=(cu2==u1)
    orient=np.select([direct|flip,swap|flipswap],[1.0,-1.0],default=0.0)
    af_u2,af_a2=m['af_u2'].values,m['af_a2'].values; maf=np.minimum(af_u2,1-af_u2)
    amb_same=amb&(maf<=0.40)&(np.abs(af_u2-af_a2)<0.10)
    amb_swap=amb&(maf<=0.40)&(np.abs(af_u2-(1-af_a2))<0.10)&~amb_same
    final=np.where(amb,np.where(amb_same,1.0,np.where(amb_swap,-1.0,0.0)),orient)
    m['final_sign']=final
    m['b_ad_aligned']=np.where(final!=0,final*m['b_ad'].values,np.nan)
    return m[(final!=0)].copy()

def prune(df,pcol='p_uk',window=WIN):
    d=df.sort_values(pcol); chrs=d['CHR'].values; poss=d['POS'].values; idx=d.index.values
    kept={}; keep=[]
    for ch,pos,ix in zip(chrs,poss,idx):
        lst=kept.setdefault(ch,[]); j=bisect.bisect_left(lst,pos); ok=True
        if j<len(lst) and lst[j]-pos<window: ok=False
        if ok and j>0 and pos-lst[j-1]<window: ok=False
        if ok: keep.append(ix); bisect.insort(lst,pos)
    return df.loc[keep]

def conc(sub):
    s=sub[sub['b_ad_aligned'].notna()]; s=s[np.sign(s['b_uk'])!=0]
    n=len(s); k=int((np.sign(s['b_uk'])==np.sign(s['b_ad_aligned'])).sum())
    p=stats.binomtest(k,n,0.5).pvalue if n else np.nan
    return n,k,(k/n if n else np.nan),p

rows=[]; recov=[]
for r in REGIONS:
    print(f"[{r}] loading ...",flush=True)
    adni=load_adni(r); ukb=load_ukb(r)
    harm=harmonize(ukb,adni)
    for thr,nm in [(GW,'gw5e8'),(SUG,'sug1e5')]:
        lead=prune(harm[harm['p_uk']<thr]); n,k,f,p=conc(lead)
        rows.append(dict(region=r,hit_set=nm,n_loci=n,n_concordant=k,
                         pct_concordant=round(100*f,1) if f==f else np.nan,binom_p=p))
        print(f"   {nm}: loci={n:4d} concordant={k:4d} {100*f if f==f else float('nan'):5.1f}% p={p:.3g}",flush=True)
    # --- lead/locus recovery diagnostic (gw5e8) ---
    ukb_gw=ukb[(ukb['p_uk']<GW)&snp_mask(ukb,'u1','u2')]
    if len(ukb_gw):
        ukb_leads=prune(ukb_gw)                      # UKB-only clumped leads (old-style)
        top=ukb_leads.sort_values('p_uk').iloc[0]
        common_keys=set(harm['CHR'].astype(np.int64)*10**9+harm['POS'].astype(np.int64))
        def in_adni(row): return (int(row['CHR'])*10**9+int(row['POS'])) in common_keys
        top_in=in_adni(top)
        n_uk_leads=len(ukb_leads)
        n_uk_leads_in_adni=int(sum(in_adni(rw) for _,rw in ukb_leads.iterrows()))
        # recovery in the top lead's window from the common set
        win=harm[(harm['CHR']==top['CHR'])&(harm['POS'].between(top['POS']-WIN,top['POS']+WIN))]
        if len(win):
            rec=win.sort_values('p_uk').iloc[0]
            rec_conc=bool(np.sign(rec['b_uk'])==np.sign(rec['b_ad_aligned']))
            rec_id,rec_puk,rec_pad=rec['id_uk'],rec['p_uk'],rec['p_ad']
        else:
            rec_conc=None; rec_id=rec_puk=rec_pad=np.nan
        recov.append(dict(region=r,n_ukb_gw_leads=n_uk_leads,n_ukb_leads_in_adni=n_uk_leads_in_adni,
            top_lead_id=top['id_uk'],top_lead_chrpos=f"{int(top['CHR'])}:{int(top['POS'])}",
            top_lead_p_ukb=top['p_uk'],top_lead_in_adni=top_in,
            recovered_lead_id=rec_id,recovered_p_ukb=rec_puk,recovered_p_adni=rec_pad,
            recovered_concordant=rec_conc))
        print(f"   top UKB lead {top['id_uk']} ({int(top['CHR'])}:{int(top['POS'])}) in ADNI={top_in}; "
              f"UKB gw leads in ADNI={n_uk_leads_in_adni}/{n_uk_leads}; recovered={rec_id} conc={rec_conc}",flush=True)
    else:
        recov.append(dict(region=r,n_ukb_gw_leads=0,n_ukb_leads_in_adni=0,top_lead_id=None,
            top_lead_chrpos=None,top_lead_p_ukb=np.nan,top_lead_in_adni=None,
            recovered_lead_id=None,recovered_p_ukb=np.nan,recovered_p_adni=np.nan,recovered_concordant=None))
        print(f"   no UKB genome-wide hits",flush=True)
    del adni,ukb,harm

tab=pd.DataFrame(rows); tab.to_csv(f"{OUT}/allregions_concordance_table.csv",index=False)
rec=pd.DataFrame(recov); rec.to_csv(f"{OUT}/allregions_lead_recovery.csv",index=False)
print("\n=== UNIFIED CONCORDANCE TABLE ===")
print(tab.to_string(index=False))
print("\n=== LEAD / LOCUS RECOVERY ===")
print(rec.to_string(index=False))
# pooled across regions (gw5e8)
gw=tab[tab['hit_set']=='gw5e8']; N=int(gw['n_loci'].sum()); K=int(gw['n_concordant'].sum())
print(f"\nPOOLED gw5e8: {K}/{N} = {100*K/N:.1f}%  binom p={stats.binomtest(K,N,0.5).pvalue:.3g}")
sg=tab[tab['hit_set']=='sug1e5']; N2=int(sg['n_loci'].sum()); K2=int(sg['n_concordant'].sum())
print(f"POOLED sug1e5: {K2}/{N2} = {100*K2/N2:.1f}%  binom p={stats.binomtest(K2,N2,0.5).pvalue:.3g}")
print("done")
