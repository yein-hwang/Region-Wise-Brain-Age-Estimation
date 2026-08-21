# ---------------------------------------------------------------------------
# Shared shell helpers. Sourced (not executed) by every script under gwas/.
#
# Responsibilities, and nothing more:
#   * load config/paths.env
#   * expose the region list
#   * turn "run this SAIGE script with these flags" into a docker / conda
#     invocation, locally or over ssh
#
# It never touches an analysis flag. Every SAIGE, PLINK, LDSC and MAGMA option
# lives verbatim in the calling script.
# ---------------------------------------------------------------------------

GWAS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# --- config ---------------------------------------------------------------
: "${GWAS_CONFIG:=$GWAS_ROOT/config/paths.env}"
if [ ! -f "$GWAS_CONFIG" ]; then
    echo "ERROR: config not found: $GWAS_CONFIG" >&2
    echo "       cp $GWAS_ROOT/config/paths.env.example $GWAS_ROOT/config/paths.env" >&2
    exit 1
fi
set -a
# shellcheck disable=SC1090
. "$GWAS_CONFIG"
set +a

# Abort early with a clear message rather than letting a tool fail on an
# empty path.
gwas_require() {
    local missing=0 v
    for v in "$@"; do
        if [ -z "${!v:-}" ]; then
            echo "ERROR: $v is not set in $GWAS_CONFIG" >&2
            missing=1
        fi
    done
    [ "$missing" -eq 0 ] || exit 1
}

# --- regions --------------------------------------------------------------
# gwas_regions [cohort]   ->   region codes, one per line, config order
gwas_regions() {
    local cohort="${1:-ukb}"
    awk -F'\t' -v c="$cohort" '
        /^#/ || NF < 3 { next }
        { n = split($3, a, ","); for (i = 1; i <= n; i++) if (a[i] == c) { print $1; next } }
    ' "$GWAS_ROOT/config/regions.tsv"
}

# gwas_region_label <code>  ->  display name
gwas_region_label() {
    awk -F'\t' -v k="$1" '!/^#/ && $1 == k { print $2; exit }' "$GWAS_ROOT/config/regions.tsv"
}

# --- compute nodes --------------------------------------------------------
# gwas_node <index>  ->  host for round-robin dispatch, or "" to run locally
gwas_node() {
    local nodes=($GWAS_NODES)
    [ "${#nodes[@]}" -eq 0 ] && return 0
    echo "${nodes[$(( $1 % ${#nodes[@]} ))]}"
}

# Block until fewer than $GWAS_MAXJOBS background jobs are running.
gwas_throttle() {
    local cap="${GWAS_MAXJOBS:-20}"
    while [ "$(jobs -rp | wc -l)" -ge "$cap" ]; do sleep 5; done
}

# --- SAIGE ----------------------------------------------------------------
# gwas_saige_cmd <script> <flag>...   ->  the command line as a single string
#
# SAIGE_RUNNER=docker : docker run --rm -v <mount> ... <image> <script> <flags>
# SAIGE_RUNNER=conda  : Rscript $SAIGE_ENV/bin/<script> <flags>
#
# The flags are passed through untouched.
gwas_saige_cmd() {
    local script="$1"; shift
    case "${SAIGE_RUNNER:-docker}" in
        docker)
            gwas_require SAIGE_IMAGE
            local mounts="" m
            for m in ${SAIGE_DOCKER_MOUNTS:-}; do mounts="$mounts -v $m:$m"; done
            echo "docker run --rm$mounts $SAIGE_IMAGE $script $*"
            ;;
        conda)
            gwas_require SAIGE_ENV
            echo "PATH=$SAIGE_ENV/bin:\$PATH R_LIBS_USER= $SAIGE_ENV/bin/Rscript $SAIGE_ENV/bin/$script $*"
            ;;
        *)
            echo "ERROR: SAIGE_RUNNER must be 'docker' or 'conda', got '${SAIGE_RUNNER}'" >&2
            exit 1
            ;;
    esac
}

# gwas_run <node> <command string>  ->  run here, or on <node> over ssh
gwas_run() {
    local node="$1"; shift
    if [ -z "$node" ]; then
        bash -c "$*"
    else
        ssh "$node" "cd $GWAS_ROOT && $*"
    fi
}
