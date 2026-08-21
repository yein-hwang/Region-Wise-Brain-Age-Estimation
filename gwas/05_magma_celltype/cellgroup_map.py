#!/usr/bin/env python
"""
Cell-type label harmonization (v2, 681-dataset / 417-base scope).

MATCHING ENGINE
---------------
Each label is normalized (lowercase, separators -> `_`, trailing cluster index
`_<digits>` stripped) and PADDED: pad = "_" + norm + "_". Every rule pattern is
searched against `pad`. Short markers are matched as whole underscore-delimited
TOKENS (`_pv_`, `_ec_`, `_it_`), NEVER as substrings; multiword ontology phrases
(`oligodendrocyte_precursor`, `endothelial`, `pericyte`) match as substrings.
This avoids the `\\b` trap (`_` is a regex word char, so `\\bpv\\b` fails inside
`pv_scube3`).

`resolve(label)` -> (group, rule_id, note). FIRST matching rule wins; ORDER
encodes specificity:
  immune -> OPC -> Oligo -> Astro -> Ependymal
  -> Endo -> Mural -> VLMC_Fib -> Vascular_gen
  -> DA -> Sert -> Prog
  -> ExN -> InN            (ExN BEFORE InN so layer/CUX2/RORB excitatory
                            subclasses that carry an interneuron marker suffix,
                            e.g. GSE168408 `L2_CUX2_LAMP5`, are called ExN)
  -> Neuron_unspec -> junk

ATLAS-SPECIFIC ABBREVIATIONS folded in (verified against each atlas' full label
list, 2026-07-12):
  Seeker2023 (WM): AS_*=Astro, EC_*=Endo, COP_*=OPC, Oligo_*=Oligo,
    Mural_*/vSMC=Mural, Microglia_*/BAM/Immune=Micro, Ex_*/Neuron_Ex=ExN,
    In_*/Neuron_In=InN, Neur/Neuron=Neuron_unspec, RELN_*=Neuron_unspec
    (RELN+ interstitial neurons; E/I ambiguous), Unidentified=Other.
  Zhu2023: EN=ExN, IN_*/CGE|MGE interneuron=InN, RG=Prog, IPC=Prog.
  GSE168408: L*_CUX2/RORB/THEMIS/TLE4=ExN, PN=ExN (projection neuron),
    BKGR_NRGN=ExN, PV/SST/VIP/LAMP5/ID2/CCK/CGE/MGE=InN, Vas_*=Vascular_gen,
    Micro_*=Micro, Non.Neu=Other (aggregate non-neuronal, cannot be split).

USER-REVIEW DECISIONS (2026-07-12):
  * CR (`cr_N`, Bhaduri) -> Other (calretinin vs Cajal-Retzius ambiguous).
  * Cajal-Retzius -> ExN;  retinal_ganglion -> Other (peripheral).
  * dropped risky trailing marker tokens (npy/reln/sorcs1/syt6/scube3/ndnf/nos1
    as InN) and `_bg_` (Bergmann vs basal-ganglia) — subclass prefixes already
    catch those labels.
  * left to Other for user ruling: generic glia (`Glia`,`glial_cell`,
    `macroglial_cell`), `Mammillary_body`, aggregate `Non.Neu`/`non_neuronal`,
    Aldinger `H_*` cerebellar abbreviations, QC/blood labels.

Canonical groups: ExN InN DA Sert Neuron_unspec | Astro Oligo OPC Micro
Ependymal | Endo Mural VLMC_Fib Vascular_gen | Prog | Other.
VASCULAR_UNION = {Endo, Mural, VLMC_Fib, Vascular_gen}.
"""

import re

CANONICAL_GROUPS = [
    "ExN", "InN", "DA", "Sert", "Neuron_unspec",
    "Astro", "Oligo", "OPC", "Micro", "Ependymal",
    "Endo", "Mural", "VLMC_Fib", "Vascular_gen",
    "Prog", "Other",
]

VASCULAR_UNION = {"Endo", "Mural", "VLMC_Fib", "Vascular_gen"}
NEURON_GROUPS  = {"ExN", "InN", "DA", "Sert", "Neuron_unspec"}


def normalize(label: str) -> str:
    s = label.strip().lower()
    for a in "-./ ":
        s = s.replace(a, "_")
    s = s.replace("(", "").replace(")", "").replace("+", "")
    s = re.sub(r"_+", "_", s)
    s = re.sub(r"(_\d+)+$", "", s)   # drop trailing "_<digits>" cluster numbers
    return s.strip("_")


# Ordered rules, searched against pad = "_"+norm+"_".
# `_tok_` = exact token; bare multiword phrases = substring.
_RAW_RULES = [
    # ---- immune / microglia ----
    ("Micro", "immune",
     r"microglia|macrophage|leukocyte|myeloid|monocyte|granulocyte|neutrophil|"
     r"lymphocyte|lymphoid|dendritic_cell|mast_cell|natural_killer|plasma_cell|"
     r"border_associated|perivascular_macrophage|kupffer|antigen_presenting|"
     r"_immune_|_micro_|_macro_|_mg_|_mgl_|_pvm_|_bam_|_t_cell|_t_cells_|"
     r"_b_cell|_b_ebf1|_nk_"),

    # ---- oligodendrocyte precursor (before oligodendrocyte) ----
    ("OPC", "opc",
     r"oligodendrocyte_precursor|committed_oligodendrocyte|premyelinating|"
     r"_opc_|_opcs_|_opc$|_cop_|_cop$|_cops_"),
    # ---- oligodendrocyte ----
    # Jakel ImOlGs = "immune oligodendrocytes" (mature oligo lineage, not OPC).
    ("Oligo", "oligo",
     r"oligodendrocyte|myelinating|newly_formed_oligo|imolgs|"
     r"_oligo_|_oligo\d|_oligos_|_ol_|_olig_|_mol_|_nfol_|_mfol_|_odc_|_odc\d"),
    # ---- astrocyte (incl. Bergmann glia, Muller glia; Aldinger H_BG) ----
    ("Astro", "astro",
     r"astrocyte|bergmann|bergman|muller_glia|müller|"
     r"_astro_|_astros_|_ast_|_asc_|_asc\d|_as_|_h_bg_"),
    # ---- ependymal / choroid plexus epithelium ----
    ("Ependymal", "ependymal",
     r"ependymal|ependyma|choroid"),

    # ---- vascular sub-families (specific before generic) ----
    ("Endo", "endo",
     r"endothelial|capillary|venous|arterial|arteriole|_artery_|_vein_|"
     r"tip_cell|_ec_|_ec$|_endo_|_endos_|_end_"),
    # Per (PsychENCODE), Peric (Linnarsson midbrain) = pericyte abbreviations.
    ("Mural", "mural",
     r"pericyte|smooth_muscle|vascular_associated_smooth|mural|"
     r"_peri_|_peri$|_peric_|_per_|_vsmc_|_smc_|contractile"),
    ("VLMC_Fib", "vlmc_fib",
     r"_vlmc_|fibroblast|fibrocyte|meningeal|leptomeningeal|meninges|arachnoid|"
     r"_dura|mesothelial|perivascular_cell|mesenchymal|mesenchyme|stromal|"
     r"vascular_leptomeningeal"),
    ("Vascular_gen", "vascular_generic",
     r"vascular|blood_vessel|vasculature|_vas_|_vas$"),

    # ---- dopaminergic / serotonergic ----
    # Kamath2022 SNc DA subtypes: SOX6_* (ventral DA family), audited Kamath-
    # unique. `_calb1_` REMOVED: calbindin is promiscuous and mis-caught SST/Ex
    # interneuron subtypes (GSE168408 SST_CALB1*, Ex_*_CALB1). Kamath CALB1_*
    # DA subtypes therefore fall to Other (acceptable; correctness > coverage).
    ("DA", "da",
     r"dopaminergic|dopamine|_da\d|_da_|nigral_da|midbrain_da|_sox6_"),
    ("Sert", "sert",
     r"serotonergic|serotonin|_sert_|_raphe|5_ht"),

    # ---- progenitor / radial glia / dividing (before neuron subtypes) ----
    ("Prog", "prog",
     r"radial_glia|radial_glial|neuroepithel|progenitor|intermediate_prog|"
     r"neuroblast|neural_stem|stem_cell|dividing|proliferat|cycling|mitotic|"
     r"gliogenic|neurogenic|glioblast|neural_crest|neuronal_restricted_precursor|"
     r"_ipc_|_ipc$|_ipc\d|_nec_|_nsc_|_npc_|_nep_|_rgc_|_rgl_|_rgl\d|_rg_|_gcp_|"
     r"_nprog_|_progbp_|_progfp|_progm_|_nbm|_nbml"),

    # ---- EXPLICIT class prefixes — BEFORE the layer/marker heuristics ----
    # Allen/Bakken/Gabitto name cortical neurons by LAYER with an explicit
    # class prefix: `Inh_L5_6_PVALB_*` (177 labels), `Exc_L6_THEMIS_*` (123).
    # Anchored `^_inh_` / `^_exc_` = label STARTS with token inh/exc, so they
    # win over the ExN `_l\d_` heuristic. Exact tokens => gene names `INHBA`/
    # `INHA` (token `inhba`) do NOT match, so excitatory `L2_3_CUX2_..._INHBA`
    # stays ExN; `L2_CUX2_LAMP5` (no inh prefix) stays ExN via the layer rule.
    # (bare gaba / interneuron / markers remain in the InN heuristic below.)
    ("InN", "inn_explicit",
     r"^_inh_|_inhibitory_|_gabaergic_|_interneuron_"),
    ("ExN", "exn_explicit",
     r"^_exc_|_excitatory_|_glutamatergic_"),

    # ---- EXCITATORY / glutamatergic (layer + marker heuristic) ----
    ("ExN", "exn",
     r"glutamatergic|excitatory|pyramidal|cajal_retzius|intratelencephalic|"
     r"extratelencephalic|corticothalamic|near_projecting|granule|dentate|"
     r"hippocampal_ca|amygdala_excitatory|thalamic_excitatory|rhombic_lip|"
     r"mossy|unipolar_brush|projection_neuron|_glut\d|_glut_|exca|exdg|expfc|"
     r"l\dit|_ex_|_ex\d|_exc_|_exn_|_en_|_en\d|_it_|_it$|_et_|_et$|_ct_|_ct$|"
     r"_np_|_np$|_l6b_|_l\d_|_ca[1-4]_|_pn_|_pn$|_nrgn_|_eomes_|"
     r"_h_gn_|_h_ecn|_h_rl_|"
     r"_cux2|_rorb|_themis|_tle4|_prss12|_satb2|_fezf2|_tbr1|_slc17|_foxp2"),
    # ---- INHIBITORY / GABAergic ----
    # `_matrix_` targets Phan striatal MSN (D1_Matrix/D2_Matrix); flagged risky
    # -> audited in build_mapping_table RISKY list.
    ("InN", "inn",
     r"gaba|interneuron|inhibitory|medium_spiny|cerebellar_inhibitory|"
     r"midbrain_derived_inhibitory|purkinje|_basket_|_stellate_|_golgi_|"
     r"_intn_|striosome|_matrix_|_h_icn_|_h_pc_|_h_mli_|"
     r"_in\d|_in_|_inn_|_inh_|"
     r"_pvalb_|_pv_|_sst_|_vip_|_lamp5_|_sncg_|_pax6_|_adarb2_|_lhx6_|"
     r"_chandelier_|_chodl_|_cge_|_cge|_mge_|_mge|_id2_|_id2|_cck_|_cck|_calb2_"),

    # ---- generic neuron (last neuronal fallback) ----
    # OMTN (oculomotor/trochlear nuc.), RN (red nuc.): midbrain motor neurons.
    ("Neuron_unspec", "neuron_generic",
     r"_neuron_|_neurons_|_neuron\d|_neuronal|neural_cell|_neur_|_reln_|_reln$|"
     r"brainstem|_omtn_|_rn_|noradrenergic"),

    # ---- explicit junk / catch-all ----
    # Ma2022 RB / RB_HBA1_HBB = red blood (hemoglobin HBA1/HBB).
    ("Other", "junk",
     r"outlier|unknown|unassigned|doublet|low_quality|ambiguous|miscellaneous|"
     r"_misc|splatter|native_cell|_other_|not_available|_nan_|unclassified|"
     r"unidentified|contamination|debris|red_blood|erythrocyte|erythroblast|"
     r"erythroid|reticulocyte|_platelet|megakaryocyte|promyelocyte|metamyelocyte|"
     r"_rb_|_rb$|_hba\d|_hbb_"),
]

_RULES = [(grp, rid, re.compile(pat)) for (grp, rid, pat) in _RAW_RULES]

_NOTE = {
    "vascular_generic": "generic vascular; sub-type unresolved",
    "neuron_generic": "generic/ambiguous neuron; E/I not separated",
    "junk": "QC / blood-contamination / unclassifiable",
}


def resolve(label: str):
    """Return (group, rule_id, note) for one cell-type label."""
    norm = normalize(label)
    if not norm:
        return "Other", "empty", "empty label"
    pad = "_" + norm + "_"
    for grp, rid, rx in _RULES:
        if rx.search(pad):
            return grp, rid, _NOTE.get(rid, "")
    return "Other", "unmatched", "no rule matched"


if __name__ == "__main__":
    tests = [
        "oligodendrocyte_precursor_cell", "OPCs", "OPC_HOXD3", "COP_B", "Oligo_D",
        "oligodendrocyte", "astrocyte", "AS_11", "H_Ast", "ASC1",
        "microglial_cell", "central_nervous_system_macrophage", "leukocyte",
        "Micro", "Micro_PVM", "meningeal_macrophage", "BAM",
        "endothelial_cell", "EC_cap_1", "EC_art_2", "vascular_associated_smooth_muscle_cell",
        "pericyte", "Mural_vein_1", "vSMC", "fibroblast", "Vascular", "vascular_9",
        "brain_vascular_cell", "Vas_CLDN5", "VLMC", "leptomeningeal_cell",
        "arachnoid_barrier_cell", "H_Meninges",
        "ependymal_cell", "Choroid_plexus", "choroid_plexus_epithelial_cell",
        "glutamatergic_neuron", "L2_3_IT", "L6_CT", "L6b", "L5_6_NP", "IT",
        "L2_CUX2_LAMP5", "L4_RORB_MET", "L5.6_TLE4_SORCS1", "PN", "BKGR_NRGN",
        "Pvalb", "Sst_Chodl", "Vip", "Lamp5_Lhx6", "LAMP5_NOS1", "PV", "PV_SCUBE3",
        "In3", "In_1", "CGE_interneuron", "IN", "IN_MGE", "ID2_CSMD1", "CCK_RELN",
        "EN", "EN_fetal_early", "Neuron_Ex", "Neuron_In", "Neur",
        "IPC_EN", "IPC_Glia", "RG", "Upper_layer_intratelencephalic",
        "dopaminergic_neuron", "forebrain_radial_glial_cell", "dividing_8",
        "Cajal_Retzius_cell", "cr_1", "RELN_2", "RELN+_neurons",
        "neuron", "neural_cell", "native_cell", "Splatter", "Miscellaneous",
        "outlier", "Non.Neu", "Bergmann_glia", "Medium_spiny_neuron",
        "Purkinje_cell", "retinal_ganglion_cell", "Schwann_cell", "erythroblast",
        "Glia", "glial_cell", "Mammillary_body",
    ]
    for t in tests:
        g, rid, note = resolve(t)
        print(f"  {t:42s} -> {g:14s} [{rid}]")
