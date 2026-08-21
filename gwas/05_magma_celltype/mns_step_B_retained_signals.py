#!/usr/bin/env python3
"""
Manuscript restructure -- STEP B: ST11 mapping-INDEPENDENT tables only.
Builds ST11a (conditionally retained dataset-cell-type signals) from confirmed tableB, and
ST11c (Step3 cross-dataset pairwise annotations) from confirmed supp_step3.
canonical_group is NOT resolved (no approved 11-group map) -> NA_PENDING_APPROVED_MAPPING.
NO added multiple-testing correction, NO final pass/drop flag on Step3, NO independence
verdict on ST11a, NO ST11b/d/e, NO group-level rollup, NO Fig.4b composition. Inputs read-only.
Analysis key = (region, Dataset, Cell_type). Retained = survived_at in retained-positive set
(step1_single | step2_retained | entered_step3).
"""
import os, sys, json, hashlib
from datetime import datetime, timezone
import pandas as pd

def _env(name):
    import os as _os, sys as _sys
    v = _os.environ.get(name)
    if not v:
        _sys.exit(f"ERROR: {name} is not set (see gwas/config/paths.env.example)")
    return v


REPO = _env("MAGMA_ROOT")
OUT  = os.path.join(REPO, "outputs/manuscript_tables")
RUNS = os.path.join(OUT, "runs")
B_PATH  = os.path.join(OUT, "tableB_independent_significant_set.tsv")
S3_PATH = os.path.join(OUT, "supp_step3_cross_dataset_all.tsv")
APPROVED_MAP_PATH = os.path.join(OUT, "harmonization_map_v2_APPROVED.tsv")

EXPECTED_REGIONS = ["global","caudate","frontal_lobe","insula","occipital_lobe",
                    "parietal_lobe","putamen","temporal_lobe","thalamus","cerebellum"]
RETAINED_STATES = {"entered_step3","step2_retained","step1_single"}
CANON = "NA_PENDING_APPROVED_MAPPING"
EXPECTED_ST11A_ROWS = 209
EXPECTED_ST11C_ROWS = 25390
FORBIDDEN_ST11C_NEW_COLS = {"adjusted","significant","passed","dropped","final"}
REQ_B  = ["region","Dataset","Cell_type","survived_at"]
REQ_S3 = ["region","MODEL","Dataset","Cell_type"]

def sha256(path):
    if not path or not os.path.exists(path): return None
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1<<20), b""): h.update(c)
    return h.hexdigest()

def read_str(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

def now_utc():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def main():
    os.makedirs(RUNS, exist_ok=True)
    base = f"st11_build_{now_utc()}"
    run = os.path.join(RUNS, base); i = 1
    while os.path.exists(run):
        run = os.path.join(RUNS, f"{base}_{i}"); i += 1
    os.makedirs(run)

    # -------------------- preflight (read-only) --------------------
    problems = []
    for nm,pth,req in [("B",B_PATH,REQ_B),("S3",S3_PATH,REQ_S3)]:
        if not os.path.exists(pth): problems.append(f"{nm}:MISSING"); continue
        cols = read_str(pth).columns.tolist()
        miss = [c for c in req if c not in cols]
        if miss: problems.append(f"{nm}:missing_cols={miss}")
    if problems:
        print("PREFLIGHT FAILED:", problems); sys.exit(1)

    B  = read_str(B_PATH)
    S3 = read_str(S3_PATH)
    s3_cols_before = list(S3.columns)

    # -------------------- ST11a: conditionally retained dataset-cell-type signals --------------------
    bad_state = sorted(set(B["survived_at"]) - (RETAINED_STATES | {"dropped_at_step2"}))
    if bad_state:
        print("UNEXPECTED survived_at states:", bad_state); sys.exit(1)
    st11a = B[B["survived_at"].isin(RETAINED_STATES)].copy()
    st11a.insert(4, "canonical_group", CANON)
    st11a["retained_status"] = st11a["survived_at"]
    st11a["_ro"] = st11a["region"].map({r:i for i,r in enumerate(EXPECTED_REGIONS)})
    st11a = st11a.sort_values(["_ro","Dataset","Cell_type"]).drop(columns="_ro")

    # -------------------- ST11c: Step3 cross-dataset pairwise (annotation only) --------------------
    st11c = S3.copy()
    st11c.insert(4, "canonical_group", CANON)   # pending; no group resolution, no verdict columns
    st11c["_ro"] = st11c["region"].map({r:i for i,r in enumerate(EXPECTED_REGIONS)})
    st11c = st11c.sort_values(["_ro","MODEL","Dataset","Cell_type"], kind="stable").drop(columns="_ro")

    # -------------------- blocking checks --------------------
    if len(st11a) != EXPECTED_ST11A_ROWS:
        sys.exit(f"ST11a row-count mismatch: observed={len(st11a)}, expected={EXPECTED_ST11A_ROWS}")
    if len(st11c) != EXPECTED_ST11C_ROWS:
        sys.exit(f"ST11c row-count mismatch: observed={len(st11c)}, expected={EXPECTED_ST11C_ROWS}")
    if st11a.duplicated(["region","Dataset","Cell_type"]).any():
        sys.exit("ST11a contains duplicate signal keys")
    if set(st11a["canonical_group"]) != {CANON}:
        sys.exit("Unexpected canonical_group assignment in ST11a")
    if set(st11c["canonical_group"]) != {CANON}:
        sys.exit("Unexpected canonical_group assignment in ST11c")
    # ST11c must not introduce any new verdict/adjustment column beyond source + canonical_group
    new_cols = set(st11c.columns) - set(s3_cols_before) - {"canonical_group"}
    if new_cols:
        sys.exit(f"ST11c introduced unexpected new columns: {sorted(new_cols)}")
    leaked = {c for c in st11c.columns if c.lower() in FORBIDDEN_ST11C_NEW_COLS}
    if leaked:
        sys.exit(f"ST11c contains forbidden verdict/adjustment columns: {sorted(leaked)}")

    st11a.to_csv(os.path.join(run,"ST11a_retained_signals_DRAFT.tsv"), sep="\t", index=False)
    st11c.to_csv(os.path.join(run,"ST11c_step3_pairwise_DRAFT.tsv"), sep="\t", index=False)

    per_region   = st11a.groupby("region").size().reindex(EXPECTED_REGIONS).fillna(0).astype(int).to_dict()
    state_counts = B["survived_at"].value_counts(dropna=False).to_dict()

    man = dict(utc_timestamp=now_utc(), status="ST11_MAPPING_INDEPENDENT_COMPLETE", run_dir=run,
        script_path=os.path.abspath(__file__), script_sha256=sha256(os.path.abspath(__file__)),
        python=sys.version.split()[0], pandas=pd.__version__,
        inputs={"B":{"path":B_PATH,"sha256":sha256(B_PATH)},"S3":{"path":S3_PATH,"sha256":sha256(S3_PATH)}},
        outputs={fn:sha256(os.path.join(run,fn)) for fn in sorted(os.listdir(run))},
        retained_states=sorted(RETAINED_STATES), canonical_group_policy=CANON,
        ST11a_rows=len(st11a), ST11a_per_region=per_region, survived_at_counts=state_counts,
        ST11c_rows=len(st11c), ST11c_source_columns=s3_cols_before,
        approved_map_present=os.path.exists(APPROVED_MAP_PATH),
        not_generated=["ST11b","ST11d_group_level","ST11e","Fig4b_group_composition",
                       "added_multiple_correction","step3_final_pass_drop_flag","ST11a_independence_verdict"],
        note="ST11a = conditionally retained signals (step1_single | step2_retained | entered_step3); "
             "canonical_group pending approved 11-group map; ST11c annotation only (no extra correction / no pass-drop flag)")
    with open(os.path.join(run,"st11_run_manifest.json"),"w") as fh:
        json.dump(man, fh, indent=2, default=str)

    print("ST11a retained rows :", len(st11a), " per_region:", per_region)
    print("ST11c pairwise rows :", len(st11c))
    print("survived_at counts  :", state_counts)
    print("canonical_group     :", CANON, "(no approved map:", not os.path.exists(APPROVED_MAP_PATH), ")")
    print("RUN DIR             :", run)
    for fn in sorted(os.listdir(run)): print(f"  {fn}  sha256={sha256(os.path.join(run,fn))[:16]}...")


if __name__ == "__main__":
    main()
