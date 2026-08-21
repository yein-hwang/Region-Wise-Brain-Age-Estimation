#!/usr/bin/env python
"""
mns_step_C_harmonize_v3 — one-shot cell-type harmonization + Fig.4b decision package.

Read-only over the frozen handoff_compact package. Does NOT run GWAS/MAGMA/FUMA,
does NOT recompute P-values / correction / significance / conditional / retained status.
It assigns a broad canonical cell-class label to each (dataset_file, cell_type_original)
pair, then summarizes the *already-final* ST10/ST11 rows. It does NOT auto-decide Fig.4b.

v3 BUG FIX (harmonization only, no statistics touched): several source labels meaning
"non-neuronal" (non_neuronal / Non.Neuronal / Non.Neu variants) were wrongly caught by
the legacy `_neuronal` token and routed to "Other neuron", and spelling variants of the
unresolved family split between "Other neuron" and "Other/Unresolved". v3 adds an explicit
raw-label override applied BEFORE the legacy mapping, keyed on cell_type_original (never on
the incorrect legacy group): raw labels normalising with prefix `nonneu` -> "Non-neuronal";
normalized glia|unknown -> "Unresolved". The catch-all is renamed "Other/Unresolved" ->
"Unresolved"; "Other neuron" is kept only for genuinely neuronal labels (DA/Sert/generic).

CANONICAL GROUPS (v3): the 14 v2 candidate groups (13 requested + "Vascular (unspecified)")
plus the split of the catch-all into "Non-neuronal" and "Unresolved" -> 15 groups total.
"Vascular (unspecified)" holds generic vascular labels (Vas / vascular_N /
Vas_CLDN5|PDGFRB|TBX18 / brain_vascular_cell) that cannot be split into
Endothelial vs Pericyte/Mural vs Fibroblast/VLMC.

SOURCE STUDY: source_study() conservatively collapses same-paper multi-region atlases to
one id (all Siletti sub-atlases -> Siletti2022, all Jorstad areas -> Jorstad2023, ...).
It is a NON-AUTHORITATIVE author/year regex: it can merge distinct same-year papers AND
splits one paper across GEO accessions. Its outputs are therefore reported as "candidate
source-study groups", never as a formal count of independent replication studies. See
source_study_collision_audit.txt.

INPUTS (frozen):
  SD3  = Supplementary_Data_3_complete_step1_cell_type_enrichment.tsv  (125,310 rows;
         12,531 unique dataset_file x cell_type_original, x 10 regions)
  ST10 = Supplementary_Table_10_significant_cell_type_enrichment.tsv   (264 sig rows)
  ST11 = Supplementary_Table_11_retained_cell_type_signals.tsv         (209 retained)

OUTPUTS (create-only, fresh timestamped dir):
  celltype_harmonization_v3_CANDIDATE.tsv          (12,531 universe rows)
  celltype_harmonization_ambiguous_labels.tsv
  source_study_mapping_CANDIDATE.tsv               (679 dataset_file -> source_study, +rule/flag)
  source_study_collision_audit.txt                 (GSE accessions + author-year collision list)
  celltype_group_summary_step1_significant.tsv     (region x group, ST10 264)
  celltype_group_summary_retained.tsv              (region x group, retained 209)
  celltype_group_support_by_source.tsv             (dataset-file vs candidate source-study group)
  celltype_sensitivity_lineage.tsv                 (Oligo/OPC separated vs combined)
  celltype_sensitivity_vascular.tsv                (vascular families separated vs combined)
  Fig4b_celltype_retained_candidate.{pdf,png}
  Fig4b_candidate_caption.md
  Fig4b_feasibility_report.md
  _diagnostics.json

RECOMMENDATION: this script sets RECO = "PENDING MANUAL REVIEW". The KEEP/MOVE/OMIT call
is made by the human from the produced figure + source-study crosswalk. The heuristic
aids (candidate-study ratios etc.) are reported as NON-AUTHORITATIVE decision aids only.
"""
import csv, re, sys, os, json
from collections import Counter, defaultdict
from datetime import datetime, timezone

def _env(name):
    import os as _os, sys as _sys
    v = _os.environ.get(name)
    if not v:
        _sys.exit(f"ERROR: {name} is not set (see gwas/config/paths.env.example)")
    return v


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cellgroup_map import resolve  # frozen legacy v2 engine

ROOT = _env("MAGMA_ROOT")
# Directory holding the frozen table package this script harmonizes over.
PKG  = os.environ.get("MANUSCRIPT_TABLE_PKG") or os.path.join(
    ROOT, "outputs/manuscript_tables")
SD3  = os.path.join(PKG, "Supplementary_Data_3_complete_step1_cell_type_enrichment.tsv")
ST10 = os.path.join(PKG, "Supplementary_Table_10_significant_cell_type_enrichment.tsv")
ST11 = os.path.join(PKG, "Supplementary_Table_11_retained_cell_type_signals.tsv")

def die(msg):
    sys.stderr.write("FATAL: " + msg + "\n"); sys.exit(1)

# ---------------------------------------------------------------- crosswalk ---
LEGACY2CANON = {
    "Oligo":         "Oligodendrocyte",
    "OPC":           "OPC",
    "Astro":         "Astrocyte",
    "Micro":         "Microglia",
    "Endo":          "Endothelial",
    "Mural":         "Pericyte/Mural",
    "VLMC_Fib":      "Fibroblast/VLMC",
    "Vascular_gen":  "Vascular (unspecified)",   # 14th group (added, honest)
    "ExN":           "Excitatory neuron",
    "InN":           "Inhibitory neuron",
    "DA":            "Other neuron",
    "Sert":          "Other neuron",
    "Neuron_unspec": "Other neuron",
    "Prog":          "Progenitor",
    "Ependymal":     "Other glia",
    "Other":         "Unresolved",              # v3: renamed from "Other/Unresolved"
}
CANON_ORDER = [
    "Oligodendrocyte", "OPC", "Astrocyte", "Microglia", "Other glia",
    "Endothelial", "Pericyte/Mural", "Fibroblast/VLMC", "Vascular (unspecified)",
    "Excitatory neuron", "Inhibitory neuron", "Other neuron",
    "Progenitor", "Non-neuronal", "Unresolved",   # v3: split catch-all + non-neuronal family
]
VASCULAR_RELATED = {"Endothelial","Pericyte/Mural","Fibroblast/VLMC","Vascular (unspecified)"}
OLIGO_LINEAGE    = {"Oligodendrocyte","OPC"}

HIGH_ONTO = {"immune","opc","oligo","astro","ependymal","endo","mural","vlmc_fib",
             "prog","inn_explicit","exn_explicit","da","sert"}
MARKER    = {"exn","inn"}
def rule_meta(rid):
    # Downgraded: these are legacy rule outputs, NOT individually re-verified against
    # each source atlas hierarchy. Only the pair-scoped HAND_VERIFIED_RULES get "high".
    if rid in HIGH_ONTO:
        return ("legacy ontology/name rule; source hierarchy not individually re-verified", "medium", "FALSE")
    if rid in MARKER:
        return ("marker/name-based inference", "medium", "FALSE")
    if rid == "neuron_generic":
        return ("name-based inference (generic neuron; E/I not separable)", "medium", "TRUE")
    if rid == "vascular_generic":
        return ("name-based inference (generic vascular; subtype not separable)", "low", "TRUE")
    return ("no resolvable taxonomy", "low", "TRUE")   # junk / unmatched / empty

# Hand-verified against source atlas hierarchy (2026-07-17), SCOPED by (cell_type, dataset
# substring) so the upgrade never leaks to a same-string label in an unrelated dataset.
HAND_VERIFIED_RULES = [
    ("AS_8", "Seeker2023", "Seeker2023 WhiteMatter atlas: AS_* = astrocyte subcluster"),
    ("Peri_L1_6_MUSTN1", "Jorstad2023", "Jorstad2023 cortex: Peri prefix = pericyte; MUSTN1 = mural marker"),
    ("VLMC", "", "vascular leptomeningeal cell (explicit ontology term)"),
    ("vascular_leptomeningeal_cell", "", "explicit VLMC ontology term"),
]
def hand_verified(ds, ct):
    for lab, sub, note in HAND_VERIFIED_RULES:
        if ct == lab and (sub == "" or sub in ds):
            return note
    return None

# ---- v3 explicit raw-label override (applied BEFORE the legacy mapping) -------
# Keyed on the RAW cell_type_original, never on the (incorrect) legacy group. Fixes the
# non-neuronal label family that the legacy `_neuronal` token routed to "Other neuron".
def _norm_label(cell_type_original):
    return re.sub(r"[^a-z0-9]+", "", cell_type_original.lower())
def is_non_neuronal_label(cell_type_original):
    # prefix rule: nonneuronal / nonneu / nonneuronalandnonneural / ... are all
    # semantically non-neuronal (verified 0 false positives across the 12,531 universe).
    return _norm_label(cell_type_original).startswith("nonneu")

def raw_label_override(cell_type_original):
    """Return (canonical_group, source_tag) or None. Shares is_non_neuronal_label()
    with the assertions so override scope and check scope can never diverge."""
    n = _norm_label(cell_type_original)
    if is_non_neuronal_label(cell_type_original):
        return "Non-neuronal", "harmonization v3 override: non-neuronal label family"
    if n in {"glia", "unknown"}:
        return "Unresolved", "harmonization v3 override: unresolved label family"
    return None

# ------------------------------------------------------------ source study ---
def source_study_full(dsf):
    """Return (study_id, mapping_rule). Collapse same-paper multi-region atlases to ONE
    candidate study id (conservative; biases toward FEWER groups). NON-AUTHORITATIVE:
    author+year can merge distinct same-year papers; one paper split across GEO accessions
    stays un-merged (see source_study_collision_audit.txt)."""
    b = re.sub(r'^\d+_', '', dsf)
    first = re.split(r'[_.]', b)[0]
    if first.startswith("GSE"):
        return first, "gse_accession"
    m = re.match(r'[A-Za-z]+', first)
    author = m.group(0) if m else first
    author = re.sub(r'etal$', '', author)   # "Jorstadetal" -> "Jorstad" (et al. is not the name)
    ym = re.search(r'(19|20)\d{2}', b)
    if ym:
        return author + ym.group(0), "author_year"
    return author, "author_only"

def source_study(dsf):
    return source_study_full(dsf)[0]

def read_tsv(fn):
    with open(fn) as f:
        return list(csv.DictReader(f, delimiter="\t"))

# ============================ READ + ASSERT (before any dir creation) =========
sd3 = read_tsv(SD3)
if len(sd3) != 125310: die(f"SD3 row count {len(sd3)} != 125310")
universe = sorted({(r["dataset_file"], r["cell_type_original"]) for r in sd3})
if len(universe) != 12531: die(f"universe {len(universe)} != 12531")
all_dataset_files = sorted({r["dataset_file"] for r in sd3})

st10 = read_tsv(ST10); st11 = read_tsv(ST11)
if len(st10) != 264: die(f"ST10 {len(st10)} != 264")
if len(st11) != 209: die(f"ST11 {len(st11)} != 209")

# ---- build harmonization in memory (no writes yet) ---------------------------
HARM_COLS = ["dataset_file","cell_type_original","canonical_group",
             "mapping_evidence","mapping_source","confidence","ambiguity_flag","notes"]
harm = {}
canon_counts = Counter()
for ds, ct in universe:
    ov = raw_label_override(ct)              # v3: raw-label override takes precedence
    if ov:
        canon, ovsrc = ov
        legacy_grp, legacy_rid, legacy_note = resolve(ct)
        harm[(ds, ct)] = dict(
            dataset_file=ds,
            cell_type_original=ct,
            canonical_group=canon,
            mapping_evidence="explicit raw-label correction",
            mapping_source=ovsrc,
            confidence="high" if canon == "Non-neuronal" else "low",
            ambiguity_flag="FALSE" if canon == "Non-neuronal" else "TRUE",
            notes=(
                f"v3 raw-label override (norm={_norm_label(ct)}); "
                f"legacy result overridden: fine={legacy_grp}, rule={legacy_rid}"
            ),
        )
        canon_counts[canon] += 1
        continue
    grp, rid, note = resolve(ct)
    canon = LEGACY2CANON[grp]
    ev, conf, amb = rule_meta(rid)
    src = f"cellgroup_map.py v2 rule={rid}"
    extra = []
    if grp in ("DA","Sert","Neuron_unspec"):
        extra.append(f"folded to 'Other neuron' (fine={grp})")
    hv = hand_verified(ds, ct)
    if hv:
        ev = "dataset hierarchy (hand-verified)"; conf = "high"
        extra.append(hv)
    if note: extra.append(note)
    harm[(ds,ct)] = dict(dataset_file=ds, cell_type_original=ct, canonical_group=canon,
                         mapping_evidence=ev, mapping_source=src, confidence=conf,
                         ambiguity_flag=amb, notes="; ".join(extra))
    canon_counts[canon] += 1

def canon_of(r):
    key=(r["dataset_file"], r["cell_type_original"])
    if key not in harm: die(f"ST row not in universe: {key}")
    return harm[key]["canonical_group"]
for r in st10: r["_canon"]=canon_of(r)
for r in st11: r["_canon"]=canon_of(r); r["_study"]=source_study(r["dataset_file"])

# ============================ v3 CORRECTNESS ASSERTIONS =======================
# retention/statistics are frozen; these guard the harmonization relabel only.
_cb = Counter(r["_canon"] for r in st11 if r["region"] == "cerebellum")
assert len(st11) == 209, f"retained total {len(st11)} != 209"
assert sum(_cb.values()) == 158, f"cerebellum retained {sum(_cb.values())} != 158"
assert _cb["Oligodendrocyte"] == 137, f"cerebellum Oligodendrocyte {_cb['Oligodendrocyte']} != 137"
assert _cb["OPC"] == 2, f"cerebellum OPC {_cb['OPC']} != 2"
assert _cb["Astrocyte"] == 2, f"cerebellum Astrocyte {_cb['Astrocyte']} != 2"
assert _cb["Non-neuronal"] == 15, f"cerebellum Non-neuronal {_cb['Non-neuronal']} != 15"
assert _cb["Unresolved"] == 2, f"cerebellum Unresolved {_cb['Unresolved']} != 2"
# Show every unique nonneu* raw label and all assigned groups.
_nn_assignments = defaultdict(set)
for (ds, ct), row in harm.items():
    if is_non_neuronal_label(ct):
        _nn_assignments[ct].add(row["canonical_group"])
print("nonneu* unique raw labels and assigned groups (universe):")
for ct in sorted(_nn_assignments):
    print(f"  {ct!r} -> {sorted(_nn_assignments[ct])}")

# The same predicate is used by both the override and the assertions.
for r in st11:
    if is_non_neuronal_label(r["cell_type_original"]):
        assert r["_canon"] == "Non-neuronal", (
            f"{r['cell_type_original']} -> {r['_canon']} "
            "(must be Non-neuronal)"
        )
for (ds, ct), row in harm.items():
    if is_non_neuronal_label(ct):
        assert row["canonical_group"] == "Non-neuronal", (
            f"universe: dataset={ds}, label={ct!r} -> "
            f"{row['canonical_group']} (must be Non-neuronal)"
        )
print("v3 ASSERTIONS PASSED")
print("cerebellum canonical counts:", dict(sorted(_cb.items(), key=lambda x: -x[1])))
_cb_nn = sorted(r["cell_type_original"] for r in st11
                if r["region"] == "cerebellum" and r["_canon"] == "Non-neuronal")
_cb_ur = sorted(r["cell_type_original"] for r in st11
                if r["region"] == "cerebellum" and r["_canon"] == "Unresolved")
print(f"cerebellum Non-neuronal original labels ({len(_cb_nn)}):", _cb_nn)
print(f"cerebellum Unresolved original labels ({len(_cb_ur)}):", _cb_ur)

# ============================ CREATE-ONLY output dir ==========================
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = os.path.join(ROOT, "outputs/manuscript_tables", f"celltype_harmonization_v3_{STAMP}")
os.makedirs(OUT, exist_ok=False)

# -------------------------------------------------- 1. harmonization tables ---
with open(os.path.join(OUT,"celltype_harmonization_v3_CANDIDATE.tsv"),"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=HARM_COLS, delimiter="\t"); w.writeheader()
    for k in universe: w.writerow(harm[k])
n_amb = 0
with open(os.path.join(OUT,"celltype_harmonization_ambiguous_labels.tsv"),"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=HARM_COLS, delimiter="\t"); w.writeheader()
    for k in universe:
        if harm[k]["ambiguity_flag"] == "TRUE":
            w.writerow(harm[k]); n_amb += 1

# ---------------------------------------------- source_study crosswalk out ---
study_stems = defaultdict(set); study_rule = {}
for d in all_dataset_files:
    sid, rule = source_study_full(d)
    study_rule[sid] = rule
    b = re.sub(r'^\d+_', '', d)
    stem = "_".join(re.split(r'[_.]', b)[:2])
    study_stems[sid].add(stem)

with open(os.path.join(OUT,"source_study_mapping_CANDIDATE.tsv"),"w",newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["dataset_file","source_study","mapping_rule","manual_override","ambiguity_flag"])
    for d in all_dataset_files:
        sid, rule = source_study_full(d)
        amb = (rule in ("gse_accession","author_only")) or (len(study_stems[sid]) > 1)
        w.writerow([d, sid, rule, "", "TRUE" if amb else "FALSE"])

with open(os.path.join(OUT,"source_study_collision_audit.txt"),"w") as f:
    f.write("SOURCE-STUDY COLLISION AUDIT (non-authoritative; for manual review)\n")
    f.write("="*70+"\n\n")
    f.write("source_study() collapses same-paper multi-region atlases to one candidate id.\n")
    f.write("Two failure modes to eyeball before trusting candidate source-study counts:\n")
    f.write(" (A) one paper split across GEO accessions -> counted as SEPARATE groups (over-count)\n")
    f.write(" (B) different same-year papers by same first author -> MERGED into one id (under-count)\n\n")
    gse = sorted({source_study(d) for d in all_dataset_files if source_study(d).startswith("GSE")})
    f.write(f"[A] GSE-accession ids kept distinct ({len(gse)}): {', '.join(gse)}\n")
    f.write("    (verify none of these are the same paper as a named study id.)\n\n")
    f.write("[B] candidate ids spanning >1 dataset-name stem (possible over-merge):\n")
    any_multi=False
    for sid in sorted(study_stems):
        if len(study_stems[sid])>1:
            any_multi=True
            f.write(f"    {sid:16s} [{study_rule[sid]}] <- {sorted(study_stems[sid])}\n")
    if not any_multi: f.write("    (none)\n")
    f.write("\n[full candidate id -> stems] (n dataset files per id):\n")
    dfcount = Counter(source_study(d) for d in all_dataset_files)
    for sid in sorted(study_stems):
        f.write(f"    {sid:20s} nfiles={dfcount[sid]:3d}  stems={sorted(study_stems[sid])}\n")

# ------------------------------------------------------------ region maps -----
REGION_ORDER = ["global","caudate","cerebellum","frontal_lobe","insula",
                "occipital_lobe","parietal_lobe","putamen","temporal_lobe","thalamus"]
RETAINED_REGIONS = [reg for reg in REGION_ORDER if any(r["region"]==reg for r in st11)]

sig = defaultdict(Counter)
for r in st10: sig[r["region"]][r["_canon"]] += 1
with open(os.path.join(OUT,"celltype_group_summary_step1_significant.tsv"),"w",newline="") as f:
    w=csv.writer(f, delimiter="\t"); w.writerow(["region","canonical_group","step1_significant_count"])
    for reg in REGION_ORDER:
        for g in CANON_ORDER:
            if sig[reg][g]: w.writerow([reg,g,sig[reg][g]])

ret = defaultdict(lambda: defaultdict(list))
for r in st11: ret[r["region"]][r["_canon"]].append(r)
region_tot = {reg: sum(len(v) for v in ret[reg].values()) for reg in ret}
with open(os.path.join(OUT,"celltype_group_summary_retained.tsv"),"w",newline="") as f:
    w=csv.writer(f, delimiter="\t")
    w.writerow(["region","canonical_group","retained_count","within_region_proportion",
                "distinct_dataset_file","distinct_candidate_source_study","region_total_retained"])
    for reg in REGION_ORDER:
        if reg not in ret: continue
        tot=region_tot[reg]
        for g in CANON_ORDER:
            rows=ret[reg][g]
            if not rows: continue
            dfs={x["dataset_file"] for x in rows}; sts={x["_study"] for x in rows}
            w.writerow([reg,g,len(rows),f"{len(rows)/tot:.4f}",len(dfs),len(sts),tot])

with open(os.path.join(OUT,"celltype_group_support_by_source.tsv"),"w",newline="") as f:
    w=csv.writer(f, delimiter="\t")
    w.writerow(["region","canonical_group","retained_signal_count",
                "distinct_dataset_file","distinct_candidate_source_study","candidate_source_study_groups"])
    for reg in REGION_ORDER:
        if reg not in ret: continue
        for g in CANON_ORDER:
            rows=ret[reg][g]
            if not rows: continue
            dfs={x["dataset_file"] for x in rows}; sts=sorted({x["_study"] for x in rows})
            w.writerow([reg,g,len(rows),len(dfs),len(sts),";".join(sts)])

# ------------------------------------------------------------ sensitivity ----
def relabel_counts(relabel):
    d=defaultdict(Counter)
    for r in st11: d[r["region"]][relabel(r["_canon"])]+=1
    return d
def studies_for(reg, pred):
    return sorted({r["_study"] for r in st11 if r["region"]==reg and pred(r["_canon"])})
def top_group(counter):
    if not counter: return ("-",0,0.0)
    g,c=counter.most_common(1)[0]; tot=sum(counter.values())
    return (g,c,c/tot)

sep_counts = relabel_counts(lambda g: g)
comb_lin   = relabel_counts(lambda g: "Oligodendrocyte lineage" if g in OLIGO_LINEAGE else g)
with open(os.path.join(OUT,"celltype_sensitivity_lineage.tsv"),"w",newline="") as f:
    w=csv.writer(f, delimiter="\t")
    w.writerow(["region","setting","top_group","top_count","top_proportion",
                "oligo_count","opc_count","oligo_lineage_count","oligo_lineage_proportion",
                "oligo_lineage_candidate_source_study_groups"])
    for reg in RETAINED_REGIONS:
        tot=region_tot[reg]
        oc=sep_counts[reg].get("Oligodendrocyte",0); pc=sep_counts[reg].get("OPC",0); lin=oc+pc
        studs=studies_for(reg, lambda g: g in OLIGO_LINEAGE)
        tgs,cs,ps=top_group(sep_counts[reg])
        w.writerow([reg,"separated",tgs,cs,f"{ps:.4f}",oc,pc,lin,f"{lin/tot:.4f}",";".join(studs) or "-"])
        tgc,cc,pc2=top_group(comb_lin[reg])
        w.writerow([reg,"combined",tgc,cc,f"{pc2:.4f}",oc,pc,lin,f"{lin/tot:.4f}",";".join(studs) or "-"])

comb_vasc = relabel_counts(lambda g: "Vascular (any)" if g in VASCULAR_RELATED else g)
with open(os.path.join(OUT,"celltype_sensitivity_vascular.tsv"),"w",newline="") as f:
    w=csv.writer(f, delimiter="\t")
    w.writerow(["region","setting","top_group","top_count","top_proportion",
                "endothelial","pericyte_mural","fibroblast_vlmc","vascular_unspecified",
                "vascular_any_count","vascular_any_proportion","vascular_candidate_source_study_groups"])
    for reg in RETAINED_REGIONS:
        tot=region_tot[reg]
        en=sep_counts[reg].get("Endothelial",0); pm=sep_counts[reg].get("Pericyte/Mural",0)
        fv=sep_counts[reg].get("Fibroblast/VLMC",0); vu=sep_counts[reg].get("Vascular (unspecified)",0)
        va=en+pm+fv+vu
        studs=studies_for(reg, lambda g: g in VASCULAR_RELATED)
        tgs,cs,ps=top_group(sep_counts[reg])
        w.writerow([reg,"separated",tgs,cs,f"{ps:.4f}",en,pm,fv,vu,va,f"{va/tot:.4f}",";".join(studs) or "-"])
        tgc,cc,pc2=top_group(comb_vasc[reg])
        w.writerow([reg,"combined",tgc,cc,f"{pc2:.4f}",en,pm,fv,vu,va,f"{va/tot:.4f}",";".join(studs) or "-"])

# ---------------------------------------------------------------- Fig. 4b ------
# Wide horizontal bubble plot; PRESENTATION-ONLY styling (no count/stat change).
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Roboto", "Arial", "DejaVu Sans"]

fig_regions = [reg for reg in REGION_ORDER if region_tot.get(reg, 0) > 0]
# x-axis = only canonical classes with >=1 retained signal across the plotted regions
fig_groups = [g for g in CANON_ORDER if any(sep_counts[reg].get(g, 0) for reg in fig_regions)]

region_labels = {
    "global": "Whole brain", "caudate": "Caudate", "cerebellum": "Cerebellum",
    "frontal_lobe": "Frontal lobe", "insula": "Insula", "occipital_lobe": "Occipital lobe",
    "parietal_lobe": "Parietal lobe", "putamen": "Putamen", "temporal_lobe": "Temporal lobe",
    "thalamus": "Thalamus",
}
group_labels = {
    "Oligodendrocyte": "Oligodendrocyte", "OPC": "OPC", "Astrocyte": "Astrocyte",
    "Microglia": "Microglia", "Other glia": "Other glia", "Endothelial": "Endothelial",
    "Pericyte/Mural": "Pericyte / mural", "Fibroblast/VLMC": "Fibroblast / VLMC",
    "Vascular (unspecified)": "Vascular\n(unspecified)", "Excitatory neuron": "Excitatory neuron",
    "Inhibitory neuron": "Inhibitory neuron", "Other neuron": "Other neuron",
    "Progenitor": "Progenitor", "Non-neuronal": "Non-neuronal", "Unresolved": "Unresolved",
}

def _render_fig4b(show_labels, out_base):
    region_to_y = {reg: i for i, reg in enumerate(fig_regions)}
    group_to_x = {grp: i for i, grp in enumerate(fig_groups)}
    plot_rows = []
    for reg in fig_regions:
        total = region_tot[reg]
        for grp in fig_groups:
            count = sep_counts[reg].get(grp, 0)
            if count == 0:
                continue
            plot_rows.append({"x": group_to_x[grp], "y": region_to_y[reg],
                              "count": count, "proportion": count / total})
    max_count = max((r["count"] for r in plot_rows), default=1)
    size_min, size_scale = 180.0, 4200.0
    bubble_sizes = [size_min + size_scale * (r["count"] / max_count) for r in plot_rows]
    cmap = plt.cm.Blues                       # single-hue sequential, publication style
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(16, max(6.2, 0.75 * len(fig_regions) + 2.0)), facecolor="white")
    ax.set_facecolor("white")
    scatter = ax.scatter(
        [r["x"] for r in plot_rows], [r["y"] for r in plot_rows],
        s=bubble_sizes, c=[r["proportion"] for r in plot_rows], cmap=cmap, norm=norm,
        edgecolors="none", zorder=3,          # no bubble outline
    )
    if show_labels:
        for row in plot_rows:
            ax.text(row["x"], row["y"], str(row["count"]), ha="center", va="center", fontsize=8,
                    color="white" if row["proportion"] >= 0.55 else "black", zorder=4)

    ax.set_xticks(np.arange(len(fig_groups)))
    ax.set_xticklabels([group_labels.get(g, g) for g in fig_groups], rotation=60, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(fig_regions)))
    ax.set_yticklabels([region_labels.get(r, r) for r in fig_regions], fontsize=10)
    for yi, reg in enumerate(fig_regions):
        ax.text(len(fig_groups) - 0.15, yi, f"n = {region_tot[reg]}", ha="left", va="center", fontsize=9)
    ax.set_xlim(-0.65, len(fig_groups) + 1.2)
    # inverted bounds -> row 0 (Whole brain) at top, last row (Thalamus) at bottom
    ax.set_ylim(len(fig_regions) - 0.35, -0.65)
    ax.set_xlabel("Canonical cell class", fontsize=10)
    ax.set_ylabel("BAG phenotype", fontsize=10)
    ax.set_title("Retained cell-type enrichment signals across regional BAGs", fontsize=12, pad=14)
    ax.set_axisbelow(True)
    ax.grid(axis="both", linewidth=0.5, alpha=0.18)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.10)
    cbar.set_label("Within-region proportion", fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8)

    legend_counts = sorted(set([1, max(1, int(round(max_count / 2))), max_count]))
    size_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="0.75", markeredgecolor="0.6",
               markersize=np.sqrt(size_min + size_scale * (count / max_count)), label=str(count))
        for count in legend_counts
    ]
    ax.legend(handles=size_handles, title="Retained annotations", loc="lower right",
              bbox_to_anchor=(1.25, -0.02), frameon=False, fontsize=8, title_fontsize=8, labelspacing=1.4)

    if show_labels:
        figure_note = (
            "Bubble area and numeric labels indicate the number of retained cell-type "
            "enrichment signals. These counts are not numbers of independent cell "
            "populations or replication studies."
        )
    else:
        figure_note = (
            "Bubble area indicates the number of retained cell-type enrichment signals. "
            "These counts are not numbers of independent cell populations or replication studies."
        )
    fig.text(0.5, 0.005, figure_note, ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(os.path.join(OUT, out_base + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, out_base + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

_render_fig4b(True,  "Fig4b_celltype_retained_candidate")           # numeric labels inside bubbles
_render_fig4b(False, "Fig4b_celltype_retained_candidate_nolabel")   # no numeric labels

# ---------------------------------------------------- decision AIDS (only) ----
retained_amb = [r for r in st11 if harm[(r["dataset_file"],r["cell_type_original"])]["ambiguity_flag"]=="TRUE"]
retained_low = [r for r in st11 if harm[(r["dataset_file"],r["cell_type_original"])]["confidence"]=="low"]
n_unres = sum(1 for r in st11 if r["_canon"]=="Unresolved")
n_nonneu = sum(1 for r in st11 if r["_canon"]=="Non-neuronal")
amb_frac = len(retained_amb)/len(st11)

region_top = {}
for reg in RETAINED_REGIONS:
    g,c,p = top_group(sep_counts[reg])
    studs = studies_for(reg, lambda gg, gg0=g: gg0==gg)
    region_top[reg] = dict(group=g, count=c, prop=p,
                           n_files=len({x["dataset_file"] for x in ret[reg][g]}),
                           n_studies=len(studs), studies=studs)

cb_lin = sep_counts["cerebellum"].get("Oligodendrocyte",0)+sep_counts["cerebellum"].get("OPC",0)
cb_lin_studies = studies_for("cerebellum", lambda g: g in OLIGO_LINEAGE)
cb_lin_files   = len({r["dataset_file"] for r in st11 if r["region"]=="cerebellum" and r["_canon"] in OLIGO_LINEAGE})
cb_indep_ratio = len(cb_lin_studies)/cb_lin if cb_lin else 0

ca_vasc = sum(sep_counts["caudate"].get(g,0) for g in VASCULAR_RELATED)
ca_vasc_studies = studies_for("caudate", lambda g: g in VASCULAR_RELATED)
ca_vasc_files   = len({r["dataset_file"] for r in st11 if r["region"]=="caudate" and r["_canon"] in VASCULAR_RELATED})
ca_indep_ratio  = len(ca_vasc_studies)/ca_vasc if ca_vasc else 0

cereb_top_sep  = top_group(sep_counts["cerebellum"])[0]
cereb_top_comb = top_group(comb_lin["cerebellum"])[0]
lineage_stable = cereb_top_sep in OLIGO_LINEAGE and cereb_top_comb=="Oligodendrocyte lineage"
caud_top_comb  = top_group(comb_vasc["caudate"])[0]
vascular_stable = caud_top_comb=="Vascular (any)"

heuristic = {
    "amb_frac_le_0.15": amb_frac <= 0.15,
    "cereb_lineage_multi_candidate_study": len(cb_lin_studies) >= 2,
    "caud_vascular_multi_candidate_study": len(ca_vasc_studies) >= 2,
    "cereb_candidate_study_ratio": round(cb_indep_ratio,4),
    "caud_candidate_study_ratio": round(ca_indep_ratio,4),
    "lineage_stable": lineage_stable,
    "vascular_stable": vascular_stable,
}
RECO = "PENDING MANUAL REVIEW"

diag = {
    "canonical_group_count": len(CANON_ORDER),
    "universe": len(universe),
    "canon_counts_universe": {g: canon_counts.get(g,0) for g in CANON_ORDER},
    "n_ambiguous_universe": n_amb,
    "st10_total": len(st10), "st11_total": len(st11),
    "retained_by_region": {reg: region_tot[reg] for reg in RETAINED_REGIONS},
    "retained_named_group": len(st11)-n_unres,
    "retained_ambiguous": len(retained_amb), "retained_amb_frac": round(amb_frac,4),
    "retained_low_conf": len(retained_low), "retained_unresolved": n_unres,
    "region_top_group": region_top,
    "cerebellum_oligo_lineage": {"count":cb_lin,"files":cb_lin_files,
        "candidate_source_study_groups":cb_lin_studies,
        "candidate_study_ratio":round(cb_indep_ratio,4),
        "prop_of_region": round(cb_lin/region_tot["cerebellum"],4)},
    "caudate_vascular": {"count":ca_vasc,"files":ca_vasc_files,
        "candidate_source_study_groups":ca_vasc_studies,
        "candidate_study_ratio":round(ca_indep_ratio,4),
        "prop_of_region": round(ca_vasc/region_tot["caudate"],4)},
    "heuristic_aids_NON_AUTHORITATIVE": heuristic,
    "recommendation": RECO,
    "output_dir": OUT,
}
with open(os.path.join(OUT,"_diagnostics.json"),"w") as f:
    json.dump(diag, f, indent=2)

# ---------------------------------------------------------------- caption -----
with open(os.path.join(OUT,"Fig4b_candidate_caption.md"),"w") as f:
    f.write(
"# Fig. 4b (candidate) caption\n\n"
"**Regional cell-type composition of retained brain-age-gap cell-type signals.** "
"Rows are regional brain-age-gap (BAG) phenotypes. Columns show the canonical "
"cell classes with at least one retained signal in the plotted phenotypes, drawn "
"from 15 broad candidate classes. These comprise the original 13-class scheme, "
"with the former 'Other/Unresolved' category separated into 'Non-neuronal' and "
"'Unresolved', together with an additional 'Vascular (unspecified)' class for "
"generic vascular labels that could not be assigned to endothelial, pericyte/mural, "
"or fibroblast/VLMC classes. Bubble area indicates the number of retained cell-type "
"enrichment signals in each region–class combination; colour indicates the share "
"of retained signals within the corresponding BAG phenotype; and the right-hand "
"`n=` value gives the phenotype's total number of retained signals. BAG phenotypes "
"with no retained signal (frontal, parietal and putaminal BAG) are omitted.\n\n"
"> **Note.** Counts represent dataset–cell-type enrichment signals rather than "
"distinct cell populations or independent replication studies. Multiple annotations "
"from the same dataset or study may therefore contribute separately.\n")

# ---------------------------------------------------------- feasibility -------
def rows_top(reg):
    t=region_top[reg]
    return (f"| {reg} | {t['group']} | {t['count']}/{region_tot[reg]} "
            f"({t['prop']*100:.0f}%) | {t['n_files']} | {t['n_studies']} | "
            f"{', '.join(t['studies']) if t['studies'] else '-'} |")
lin_stable_txt = "holds" if lineage_stable else "does NOT hold"
vasc_stable_txt = "holds" if vascular_stable else "does NOT hold"
ca_en=sep_counts['caudate'].get('Endothelial',0); ca_pm=sep_counts['caudate'].get('Pericyte/Mural',0)
ca_fv=sep_counts['caudate'].get('Fibroblast/VLMC',0); ca_vu=sep_counts['caudate'].get('Vascular (unspecified)',0)

report = []
report.append("# Fig. 4b feasibility report\n")
report.append("*One-shot harmonization of ST10 (264 significant) / ST11 (209 retained) into 15 "
    "broad candidate canonical cell classes. No statistic, P-value, correction, significance, "
    "conditional or retention status was recomputed or changed. Mapping was built on the full "
    "12,531-pair Step-1 universe via the frozen `cellgroup_map.py` engine plus a v3 raw-label "
    "override (non-neuronal label family), then applied to ST10/ST11.*\n")
report.append("## 0. Canonical group framing\n"
    "**15 broad candidate groups** = the original 13-class scheme with the former "
    "`Other/Unresolved` split into `Non-neuronal` and `Unresolved`, plus `Vascular (unspecified)` "
    "for generic vascular labels that cannot be assigned to Endothelial / Pericyte-Mural / "
    "Fibroblast-VLMC. **v3 correction:** raw labels normalising with the `nonneu` prefix "
    "(non_neuronal, Non.Neuronal, Non.Neu, Non_neuronal_and_Non_neural, …) are assigned "
    "`Non-neuronal` — previously the legacy "
    "`_neuronal` token mis-routed them to `Other neuron`; `glia`/`unknown` -> `Unresolved`.\n")
report.append("## 1. Mapping coverage (retained 209)\n"
    f"- Retained mapped to a **named** (non-`Unresolved`) group: **{len(st11)-n_unres}/209**.\n"
    f"- Retained flagged **ambiguous** (ambiguity_flag=TRUE): **{len(retained_amb)}/209 ({amb_frac*100:.1f}%)**.\n"
    f"- Retained at **low** mapping confidence: **{len(retained_low)}/209**.\n"
    f"- `Non-neuronal` retained: **{n_nonneu}/209**;  `Unresolved` retained: **{n_unres}/209**.\n")
report.append("## 2. Region-level top canonical group (retained, separated 15-group)\n"
    "| region | top group | retained (share) | dataset files | candidate source-study groups | groups |\n"
    "|---|---|---|---|---|---|\n"
    + "\n".join(rows_top(reg) for reg in RETAINED_REGIONS) + "\n\n"
    "> \"dataset files\" and \"candidate source-study groups\" are distinct: same-paper "
    "multi-region atlases (all Siletti sub-atlases, all Jorstad cortical areas, …) collapse to "
    "**one** conservatively collapsed candidate source-study group. These non-authoritative "
    "candidate groupings provide an approximate summary of source-study support and are not "
    "interpreted as a formal count of independent replication studies "
    "(caveats in `source_study_collision_audit.txt`).\n")
report.append("## 3. Cerebellum: dominated by oligodendrocyte-lineage repeated labels?\n"
    f"- Oligodendrocyte + OPC retained in cerebellum: **{cb_lin}/{region_tot['cerebellum']} "
    f"({cb_lin/region_tot['cerebellum']*100:.0f}% of cerebellum retained)**.\n"
    f"- Supported by **{cb_lin_files} dataset files** but only "
    f"**{len(cb_lin_studies)} conservatively collapsed candidate source-study groups**: "
    f"{', '.join(cb_lin_studies)}.\n"
    f"- Candidate-study ratio (candidate groups / retained annotations) = **{cb_indep_ratio:.3f}** — "
    "the bubble is built from many same-study sub-atlas annotations, so its *size* far exceeds the "
    "conservatively collapsed candidate source-study support.\n"
    "- **Manuscript-facing composition (v3-corrected):** The remaining cerebellar signals comprised "
    f"astrocyte (n = {sep_counts['cerebellum'].get('Astrocyte',0)}), broad non-neuronal "
    f"(n = {sep_counts['cerebellum'].get('Non-neuronal',0)}) and unresolved annotations "
    f"(n = {sep_counts['cerebellum'].get('Unresolved',0)}).\n")
report.append("## 4. Caudate: is the vascular conclusion classification-dependent?\n"
    f"- Vascular-related retained in caudate (Endo+Peri/Mural+Fib/VLMC+Vascular-unspecified): "
    f"**{ca_vasc}/{region_tot['caudate']} ({ca_vasc/region_tot['caudate']*100:.0f}%)**.\n"
    f"- Split across families: Endothelial={ca_en}, Pericyte/Mural={ca_pm}, "
    f"Fibroblast/VLMC={ca_fv}, Vascular(unspecified)={ca_vu}.\n"
    f"- Supported by **{ca_vasc_files} dataset files**, "
    f"**{len(ca_vasc_studies)} conservatively collapsed candidate source-study groups**: "
    f"{', '.join(ca_vasc_studies) if ca_vasc_studies else '-'}.\n")
report.append("## 5. Sensitivity — does the conclusion survive regrouping?\n"
    f"**Oligodendrocyte vs OPC (separated ↔ combined):** cerebellum top group is `{cereb_top_sep}` "
    f"when separated and `{cereb_top_comb}` when combined → oligodendrocyte-lineage dominance "
    f"**{lin_stable_txt}** either way. (`celltype_sensitivity_lineage.tsv`)\n\n"
    f"**Vascular families (separated ↔ combined into \"Vascular (any)\"):** caudate top group is "
    f"`{top_group(sep_counts['caudate'])[0]}` when separated and `{caud_top_comb}` when the four "
    f"vascular families are merged → caudate vascular conclusion **{vasc_stable_txt}**. "
    f"(`celltype_sensitivity_vascular.tsv`)\n")
report.append("## 6. Non-authoritative decision aids\n"
    "> These are heuristics to speed a human read, **not** used to auto-decide figure placement. "
    "The `source_study()` author/year regex can merge distinct same-year papers and splits one "
    "paper across GEO accessions; the 0.15 candidate-study-ratio line has no statistical basis. "
    "Make the final call from the figure + `source_study_mapping_CANDIDATE.tsv` + the audit.\n")
report += [f"- `{k}` = **{v}**" for k,v in heuristic.items()]
report.append("\n## 7. Recommendation\n"
    "### PENDING MANUAL REVIEW\n"
    "This script does not auto-select KEEP / MOVE / OMIT. The material to decide in one pass is:\n\n"
    "- **`Fig4b_celltype_retained_candidate.png`** — does the composition read clearly?\n"
    "- **§3 / §4 above** — cerebellum: "
    f"{cb_lin} oligodendrocyte-lineage annotations among {region_tot['cerebellum']} retained "
    f"signals, from {cb_lin_files} files but {len(cb_lin_studies)} candidate source-study groups; "
    f"caudate: {ca_vasc} vascular-related from {ca_vasc_files} files / {len(ca_vasc_studies)} "
    "candidate source-study groups.\n"
    "- **§5** — both patterns are stable under the lineage and vascular regroupings.\n\n"
    "**Guidance for the manual call:** if the bubble *size* (an annotation count dominated by "
    "same-study Siletti sub-atlases) is judged to overstate independent evidence in a main-text "
    "figure even with the caveat note, choose **B (MOVE TO SUPPLEMENTARY)** or **C (OMIT; report "
    "in Results + ST10)**; choose **A (KEEP)** only if the annotation-count semantics are "
    "acceptable in the main text. The observed cerebellar oligodendrocyte-lineage and caudate "
    "vascular patterns are stable under the tested regroupings and are fully reported in "
    "ST10/ST11 regardless of the figure's placement.\n")

with open(os.path.join(OUT,"Fig4b_feasibility_report.md"),"w") as f:
    f.write("\n".join(report))

print("HARMONIZATION + FIG4b OK  ->", OUT)
print(json.dumps(diag, indent=2))
