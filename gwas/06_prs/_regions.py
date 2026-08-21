"""Region codes and display names, from config/regions.tsv.

Same convention as 03_postgwas: PRS_REGIONS overrides the list, regions.tsv
supplies the display names used in the published tables.
"""
import os

REGIONS = os.environ.get("PRS_REGIONS",
    "global,caudate,cerebellum,frontal_lobe,insula,"
    "occipital_lobe,parietal_lobe,putamen,temporal_lobe,thalamus").split(",")

_REG_TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "regions.tsv")
LABEL = {}
with open(_REG_TSV) as _f:
    for _line in _f:
        if _line.startswith("#"):
            continue
        _p = _line.rstrip("\n").split("\t")
        if len(_p) >= 2:
            LABEL[_p[0]] = _p[1]


def require(*names):
    """Return the named environment variables, or exit with the missing ones."""
    import sys
    vals, missing = [], []
    for n in names:
        v = os.environ.get(n)
        if not v:
            missing.append(n)
        vals.append(v)
    if missing:
        print("ERROR: not set in config/paths.env: " + ", ".join(missing), file=sys.stderr)
        print("       set -a; . gwas/config/paths.env; set +a", file=sys.stderr)
        sys.exit(1)
    return vals if len(vals) > 1 else vals[0]
