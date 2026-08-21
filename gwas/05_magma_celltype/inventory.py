#!/usr/bin/env python
"""
Inventory + reproducible classification of the FUMA_scRNA_data_v2 celltype
covar files into {mouse, human_nonbrain, human_brain}.

Only human_brain files are used as MAGMA gene-property covariates.

Selection rules (locked with user 2026-07-12, 679-dataset scope):
  MOUSE (exclude):
    - filename contains "_Mouse_", OR
    - base starts with one of the mouse atlases:
      MouseCellAtlas, TabulaMuris, DropViz, Linnarsson_MouseBrainAtlas
  HUMAN_NONBRAIN (exclude):
    - Xu_Human_2023_* peripheral tissues (everything EXCEPT *_Hippocampus)
    - Travaglini_2020_Blood, Travaglini_2020_Lung
    - PBMC_10x_*
    - GSE81547_Human_Pancreas, GSE84133_Human_Pancreas, GSE89232_Human_Blood
  HUMAN_BRAIN (include): everything else, i.e.
    - any "_Human_" tagged file not caught above (incl. Xu Hippocampus)
    - untagged brain sets: PsychENCODE_Adult, PsychENCODE_Developmental,
      Johansen_2023_Neocortex, Linnarsson_Prenatal_* (meninges)

`dataset_base` = filename with leading "<num>_" and trailing
"_level<k>[ _rank<j>].txt" stripped -> used for dataset-level dedup.
"""

import os
import re
import glob

def _env(name):
    import os as _os, sys as _sys
    v = _os.environ.get(name)
    if not v:
        _sys.exit(f"ERROR: {name} is not set (see gwas/config/paths.env.example)")
    return v


CELLTYPE_DIR = os.environ.get("FUMA_SCRNA_DIR") or (
    _env("MAGMA_ROOT") + "/data/FUMA_scRNA_data_v2/celltype")

_MOUSE_ATLAS_PREFIX = (
    "MouseCellAtlas", "TabulaMuris", "DropViz", "Linnarsson_MouseBrainAtlas",
)

# explicit human non-brain bases (not Xu; Xu handled by rule)
_HUMAN_NONBRAIN_EXACT = {
    "Travaglini_2020_Blood", "Travaglini_2020_Lung",
    "GSE81547_Human_Pancreas", "GSE84133_Human_Pancreas",
    "GSE89232_Human_Blood",
}
_HUMAN_NONBRAIN_PREFIX = ("PBMC_10x",)

# untagged files that ARE human brain
_UNTAGGED_BRAIN_PREFIX = (
    "PsychENCODE_Adult", "PsychENCODE_Developmental",
    "Johansen_2023_Neocortex", "Linnarsson_Prenatal_",
)


def strip_num_prefix(name: str) -> str:
    return re.sub(r"^\d+_", "", name)


def dataset_base(filename: str) -> str:
    """Filename -> dataset base (drop dir, leading num, trailing level/rank)."""
    b = os.path.basename(filename)
    b = re.sub(r"\.txt$", "", b)
    b = re.sub(r"_level\d+(_rank\d+)?$", "", b)
    return strip_num_prefix(b)


def classify(filename: str) -> str:
    """Return one of {'mouse','human_nonbrain','human_brain'}."""
    b = os.path.basename(filename)
    base = dataset_base(filename)

    # --- mouse ---
    if "_Mouse_" in b:
        return "mouse"
    if base.startswith(_MOUSE_ATLAS_PREFIX):
        return "mouse"

    # --- human non-brain ---
    if base.startswith("Xu_Human_2023_"):
        # Xu peripheral tissues excluded; hippocampus kept as brain
        if "Hippocampus" in base:
            return "human_brain"
        return "human_nonbrain"
    if base in _HUMAN_NONBRAIN_EXACT:
        return "human_nonbrain"
    if base.startswith(_HUMAN_NONBRAIN_PREFIX):
        return "human_nonbrain"

    # --- human brain ---
    if "_Human_" in b:
        return "human_brain"
    if base.startswith(_UNTAGGED_BRAIN_PREFIX):
        return "human_brain"

    # anything left untagged & unmatched -> flag as 'other' for manual review
    return "other_unmatched"


def all_files():
    return sorted(glob.glob(os.path.join(CELLTYPE_DIR, "*.txt")))


def human_brain_files():
    return [f for f in all_files() if classify(f) == "human_brain"]


if __name__ == "__main__":
    from collections import Counter, defaultdict
    files = all_files()
    cls = Counter()
    bases_by_cls = defaultdict(set)
    for f in files:
        c = classify(f)
        cls[c] += 1
        bases_by_cls[c].add(dataset_base(f))
    print(f"total files: {len(files)}")
    for c in ("human_brain", "human_nonbrain", "mouse", "other_unmatched"):
        print(f"  {c:18s}: {cls[c]:4d} files  |  {len(bases_by_cls[c]):4d} unique bases")
    if bases_by_cls["other_unmatched"]:
        print("\nUNMATCHED bases (need manual rule):")
        for b in sorted(bases_by_cls["other_unmatched"]):
            print("   ", b)
