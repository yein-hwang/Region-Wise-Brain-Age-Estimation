#!/usr/bin/env python
"""
Central configuration for the UK Biobank region-wise BAG GWAS
MAGMA cell-type enrichment (gene-property) pipeline.

681-dataset scope (locked with user 2026-07-12). See PROJECT_MEMORY.md for
full context. Key facts:
  * GWAS source = UK Biobank region-wise BAG, N = 41,067, 10 regions.
    SNP id column = MarkerID (rsID); p-value column = p.value; N column = N.
  * covar set = 681 human-brain single-cell expression files (417 unique
    dataset bases) from FUMA_scRNA_data_v2/celltype, selected by
    inventory.classify() (mouse + human-non-brain excluded). Auto-collected,
    NOT hard-coded.
  * Identical dataset set and identical MAGMA options for every region.
"""

import os
from . import inventory

def _env(name):
    import os as _os, sys as _sys
    v = _os.environ.get(name)
    if not v:
        _sys.exit(f"ERROR: {name} is not set (see gwas/config/paths.env.example)")
    return v


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
MAGMA_ROOT   = _env("MAGMA_ROOT")
MAGMA_BIN    = os.path.join(MAGMA_ROOT, "magma")
DATA_DIR     = os.path.join(MAGMA_ROOT, "data")

# Reference panel (1000G phase3 EUR, GRCh37/hg19) and gene annotation (ENSG).
BFILE        = os.path.join(DATA_DIR, "g1000_eur")
GENE_ANNOT   = os.path.join(DATA_DIR, "annot_g1000eur_ENSG.genes.annot")

# Single-cell expression matrices (gene x cell-type mean expression, ENSG ids).
# v2 = FUMA pre-processed resource downloaded 2026-07-12 (876 files).
SC_DATA_DIR  = os.path.join(DATA_DIR, "FUMA_scRNA_data_v2", "celltype")

# GWAS summary statistics: {region}/results/{region}_imputed_sumstats.txt
GWAS_ROOT    = _env("GWAS_UKB_DIR")

def gwas_path(region: str) -> str:
    return os.path.join(GWAS_ROOT, region, "results", f"{region}_imputed_sumstats.txt")

# Output layout
OUTPUT_DIR       = os.path.join(MAGMA_ROOT, "outputs")
REGIONS_DIR      = os.path.join(OUTPUT_DIR, "regions")        # gene analysis (.genes.raw)
GENEPROP_DIR     = os.path.join(OUTPUT_DIR, "gene_property")  # step3 marginal gsa.out
JOINTPAIRS_DIR   = os.path.join(OUTPUT_DIR, "jointpairs")     # conditional joint-pairs
COMBINED_DIR     = os.path.join(OUTPUT_DIR, "combined")       # final tables
DOCS_DIR         = os.path.join(OUTPUT_DIR, "docs")           # Methods docs
LOGS_DIR         = os.path.join(OUTPUT_DIR, "logs")           # run logs

# --------------------------------------------------------------------------
# GWAS / MAGMA column + option conventions  (verified 2026-07-12)
# --------------------------------------------------------------------------
SNP_COL   = "MarkerID"
PVAL_COL  = "p.value"     # NOTE: dot in the name; MAGMA accepts it literally
NCOL      = "N"

# gene-property model options (identical for every region x dataset)
GENEPROP_MODEL = ["condition-hide=Average", "direction=greater"]

# --------------------------------------------------------------------------
# 10 regions (whole-brain "global" + 9 regions)
# --------------------------------------------------------------------------
REGIONS = [
    "global",          # whole-brain
    "caudate",
    "cerebellum",
    "frontal_lobe",
    "insula",
    "occipital_lobe",
    "parietal_lobe",
    "putamen",
    "temporal_lobe",
    "thalamus",
]

# --------------------------------------------------------------------------
# 681 human-brain covar files (auto-collected from inventory; order-stable)
# --------------------------------------------------------------------------
def dataset_files():
    """Sorted list of absolute paths to the 681 human-brain covar files."""
    return inventory.human_brain_files()

def dataset_tag(path: str) -> str:
    """Stable per-file tag = basename without .txt (unique across the 681)."""
    return os.path.basename(path)[:-4] if path.endswith(".txt") else os.path.basename(path)

# significance threshold used throughout (BH-FDR primary; Bonferroni reported too)
ALPHA = 0.05

# separator used to join region and dataset in output filenames
TAG_SEP = "__"


def genes_raw(region: str) -> str:
    return os.path.join(REGIONS_DIR, f"{region}.genes.raw")


def ensure_dirs():
    for d in (REGIONS_DIR, GENEPROP_DIR, JOINTPAIRS_DIR,
              COMBINED_DIR, DOCS_DIR, LOGS_DIR):
        os.makedirs(d, exist_ok=True)
