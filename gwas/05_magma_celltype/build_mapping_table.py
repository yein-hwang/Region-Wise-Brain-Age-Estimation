#!/usr/bin/env python
"""
Build the cell-group mapping table + full coverage/audit report for user review.

Outputs (outputs/combined/):
  cellgroup_mapping.tsv      label, n_files, n_dataset_bases, atlases, group,
                             rule_id, note   (one row per unique label)
  cellgroup_rule_audit.tsv   group, rule_id, label, n_files  (every match)
  cellgroup_report.txt       everything printed below

Report sections:
  1. coverage (by label / by (file,label) pair)
  2. risky short-token audit (every label each risky token grabbed)
  3. Other — full list + suggested disposition
  4. VASCULAR_UNION stats (per sub-group + union bases)
  5. Linnarsson_Prenatal meninges non-brain fractions
  6. VLMC_Fib dataset distribution (does it depend on the 3 meninges sets?)
  + per-group stats

Parses covar headers only; runs no MAGMA. Mapping NOT finalized until reviewed.
"""

import os
import re
from collections import Counter, defaultdict

from . import inventory as inv
from . import cellgroup_map as cm
from . import config

OUT = config.COMBINED_DIR

# risky short tokens to spotlight: (label, regex, expected group hint)
RISKY = [
    ("_as_",   "Astro"), ("_asc_", "Astro"), ("_ast_", "Astro"),
    ("_en_",   "ExN"),   ("_en\\d", "ExN"),  ("_ex_",  "ExN"),
    ("_pn_",   "ExN"),   ("_it_",  "ExN"),   ("_ct_",  "ExN"),
    ("_np_",   "ExN"),   ("_et_",  "ExN"),
    ("_ol_",   "Oligo"), ("_odc_", "Oligo"),
    ("_ec_",   "Endo"),  ("_peri_", "Mural"),
    ("_rg_",   "Prog"),  ("_vas_", "Vascular_gen"),
    ("_neur_", "Neuron_unspec"), ("_reln_", "Neuron_unspec"),
    ("_in_",   "InN"),   ("_in\\d", "InN"),  ("_pv_",  "InN"),
    # newly-added flagged tokens (user review 2026-07-12)
    ("_matrix_", "InN"), ("_rn_",  "Neuron_unspec"), ("_end_", "Endo"),
    ("_sox6_", "DA"),    ("_calb1_", "DA"), ("_macro_", "Micro"),
    ("_glut\\d", "ExN"), ("exca",  "ExN"), ("l\\dit", "ExN"),
    ("stem_cell", "Prog"), ("_nbm", "Prog"),
    # third batch (user review 2026-07-12)
    ("_per_", "Mural"), ("_peric_", "Mural"), ("_h_bg_", "Astro"),
    ("_h_gn_", "ExN"), ("_h_ecn", "ExN"), ("_h_rl_", "ExN"),
    ("_h_icn_", "InN"), ("_h_pc_", "InN"), ("_h_mli_", "InN"),
    ("_exc_", "ExN"), ("_inn_", "InN"), ("imolgs", "Oligo"),
    ("brainstem", "Neuron_unspec"), ("_rb_", "Other"), ("_hba\\d", "Other"),
]

# non-brain / peripheral label markers for the meninges-composition check
NONBRAIN = ["retinal", "schwann", "pinealocyte", "chondrocyte",
            "mesenchyme_condensation", "mesenchymal", "mechanoreceptor",
            "erythro", "platelet", "megakaryocyte", "promyelocyte",
            "metamyelocyte", "reticulocyte", "neural_crest"]


def atlas_of(base):
    return re.split(r"[._]", base)[0]


def collect():
    files = inv.human_brain_files()
    lab_files = Counter()
    lab_bases = defaultdict(set)
    lab_atlas = defaultdict(set)
    pair_count = 0
    for f in files:
        base = inv.dataset_base(f)
        at = atlas_of(base)
        with open(f) as fh:
            header = fh.readline().split()
        for c in header[1:]:
            if c == "Average":
                continue
            lab_files[c] += 1
            lab_bases[c].add(base)
            lab_atlas[c].add(at)
            pair_count += 1
    return files, lab_files, lab_bases, lab_atlas, pair_count


def is_nonbrain(lab):
    n = cm.normalize(lab)
    return any(k in n for k in NONBRAIN)


def meninges_report(P):
    import glob
    files = sorted(glob.glob(os.path.join(cm.__file__, "..", "..", "..",
                   "data/FUMA_scRNA_data_v2/celltype/*Linnarsson_Prenatal*")))
    files = sorted(glob.glob(os.path.join(config.SC_DATA_DIR,
                   "*Linnarsson_Prenatal*")))
    P("\n" + "=" * 70)
    P("5. Linnarsson_Prenatal MENINGES — cell-type composition (by label)")
    P("   (covar files carry mean expr per type, NOT #cells; fractions are of")
    P("    the cell-type COLUMNS, i.e. what the dataset contributes as covariates)")
    P("=" * 70)
    for f in files:
        labs = [l for l in open(f).readline().split()[1:] if l != "Average"]
        tot = len(labs)
        b = Counter()
        nb = []
        for l in labs:
            g = cm.resolve(l)[0]
            if is_nonbrain(l):
                b["NON-BRAIN/peripheral"] += 1
                nb.append(l)
            elif g in cm.NEURON_GROUPS:
                b["neuron"] += 1
            elif g in ("Astro", "Oligo", "OPC", "Ependymal"):
                b["glia/ependymal"] += 1
            elif g == "Micro":
                b["immune"] += 1
            elif g in cm.VASCULAR_UNION:
                b["vascular/meningeal"] += 1
            elif g == "Prog":
                b["progenitor"] += 1
            else:
                b["Other"] += 1
        P(f"\n  {os.path.basename(f)}  ({tot} cell-type columns)")
        for k in ["neuron", "glia/ependymal", "immune", "vascular/meningeal",
                  "progenitor", "NON-BRAIN/peripheral", "Other"]:
            if b[k]:
                P(f"      {k:22s}: {b[k]:2d}  ({100*b[k]/tot:4.1f}%)")
        P(f"      non-brain labels: {', '.join(sorted(nb))}")


def main():
    os.makedirs(OUT, exist_ok=True)
    files, lab_files, lab_bases, lab_atlas, pair_count = collect()
    labels = sorted(lab_files, key=lambda l: (-lab_files[l], l))
    res = {l: cm.resolve(l) for l in labels}
    all_bases = set().union(*lab_bases.values())

    # ---- TSVs ----
    with open(os.path.join(OUT, "cellgroup_mapping.tsv"), "w") as o:
        o.write("label\tn_files\tn_dataset_bases\tatlases\tgroup\trule_id\tnote\n")
        for l in labels:
            g, rid, note = res[l]
            o.write(f"{l}\t{lab_files[l]}\t{len(lab_bases[l])}\t"
                    f"{','.join(sorted(lab_atlas[l]))}\t{g}\t{rid}\t{note}\n")
    by_rule = defaultdict(list)
    for l in labels:
        g, rid, _ = res[l]
        by_rule[(g, rid)].append(l)
    with open(os.path.join(OUT, "cellgroup_rule_audit.tsv"), "w") as o:
        o.write("group\trule_id\tlabel\tn_files\n")
        for key in sorted(by_rule):
            for l in sorted(by_rule[key], key=lambda x: -lab_files[x]):
                o.write(f"{key[0]}\t{key[1]}\t{l}\t{lab_files[l]}\n")

    lines = []
    P = lines.append

    # ---- header + coverage ----
    n_lab = len(labels)
    other_lab = [l for l in labels if res[l][0] == "Other"]
    pairs_other = sum(lab_files[l] for l in other_lab)
    P("=" * 70)
    P(f"CELL-GROUP MAPPING REPORT — 681 files, {len(all_bases)} dataset-bases")
    P("=" * 70)
    P(f"unique labels        : {n_lab}")
    P(f"(file,label) pairs   : {pair_count}")
    P("\n1. COVERAGE (mapped to a real group = NOT Other):")
    P(f"   by label : {n_lab-len(other_lab):4d}/{n_lab}  "
      f"({100*(n_lab-len(other_lab))/n_lab:.1f}%)   [Other {len(other_lab)}]")
    P(f"   by pair  : {pair_count-pairs_other:5d}/{pair_count}  "
      f"({100*(pair_count-pairs_other)/pair_count:.1f}%)   [Other {pairs_other}]")

    # ---- risky-token audit ----
    P("\n" + "=" * 70)
    P("2. RISKY SHORT-TOKEN AUDIT (every label the token grabbed; check group)")
    P("=" * 70)
    for tok, hint in RISKY:
        rx = re.compile(tok)
        hits = [l for l in labels if rx.search("_" + cm.normalize(l) + "_")]
        P(f"\n  {tok:8s} (expect {hint}): {len(hits)} labels")
        for l in sorted(hits, key=lambda x: -lab_files[x]):
            g = res[l][0]
            flag = "" if g == hint else f"  <-- {g} (not {hint})"
            P(f"      {lab_files[l]:4d}  {l:40s} -> {g}{flag}")

    # ---- Other ----
    P("\n" + "=" * 70)
    P(f"3. OTHER — all {len(other_lab)} labels (by freq); suggest disposition")
    P("=" * 70)
    for l in sorted(other_lab, key=lambda x: -lab_files[x]):
        P(f"  {lab_files[l]:4d}  {l:42s} [{res[l][1]}] "
          f"({','.join(sorted(lab_atlas[l]))})")

    # ---- per-group + VASCULAR_UNION ----
    grp_lab = Counter(); grp_file = Counter(); grp_base = defaultdict(set)
    for l in labels:
        g = res[l][0]
        grp_lab[g] += 1; grp_file[g] += lab_files[l]; grp_base[g] |= lab_bases[l]
    P("\n" + "=" * 70)
    P("PER-GROUP STATS (labels / file-appearances / dataset-bases)")
    P("=" * 70)
    for g in cm.CANONICAL_GROUPS:
        P(f"  {g:14s}: {grp_lab[g]:5d} lab  {grp_file[g]:6d} files  {len(grp_base[g]):4d} bases")
    P("\n" + "=" * 70)
    P("4. VASCULAR_UNION (Endo ∪ Mural ∪ VLMC_Fib ∪ Vascular_gen)")
    P("=" * 70)
    ub = set()
    for g in ["Endo", "Mural", "VLMC_Fib", "Vascular_gen"]:
        P(f"  {g:14s}: {grp_lab[g]:4d} lab  {grp_file[g]:5d} files  {len(grp_base[g]):4d} bases")
        ub |= grp_base[g]
    P(f"  {'UNION':14s}: {'':4}      {'':5}      {len(ub):4d} bases "
      f"({100*len(ub)/len(all_bases):.1f}% of {len(all_bases)})")

    meninges_report(P)

    # ---- VLMC_Fib distribution ----
    P("\n" + "=" * 70)
    P("6. VLMC_Fib DATASET DISTRIBUTION (does it depend on the 3 meninges sets?)")
    P("=" * 70)
    vlmc_bases = grp_base["VLMC_Fib"]
    men = {"Linnarsson_Prenatal_Meninx", "Linnarsson_Prenatal_BrainMeninx",
           "Linnarsson_Prenatal_ForebrainMeninges"}
    by_at = Counter(atlas_of(b) for b in vlmc_bases)
    P(f"  VLMC_Fib appears in {len(vlmc_bases)} dataset-bases across "
      f"{len(by_at)} atlases:")
    for at, c in by_at.most_common():
        P(f"      {at:18s}: {c}")
    P(f"  of these, {len(vlmc_bases & men)}/{len(vlmc_bases)} are the 3 prenatal "
      f"meninges sets -> VLMC_Fib is {'BROAD' if len(vlmc_bases-men)>20 else 'NARROW'}, "
      f"not meninges-dependent" if len(vlmc_bases-men) > 20 else "")

    text = "\n".join(lines)
    with open(os.path.join(OUT, "cellgroup_report.txt"), "w") as o:
        o.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
