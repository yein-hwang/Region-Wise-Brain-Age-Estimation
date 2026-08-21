#!/bin/bash
# Concatenate SAIGE per-chromosome results into one summary-statistics file
# per region: {region}/results/{region}_imputed_sumstats.txt
#
# A region is merged only if all 22 chromosomes are present AND each SAIGE
# .index file ends with "Have completed the analyses of all chunks."
#
# Usage: bash merge_chromosomes.sh <cohort_dir> [region ...]
set -uo pipefail
. "$(dirname "$0")/../config/common.sh"

cohort_dir="${1:?Usage: $0 <cohort_dir> [region ...]}"; shift
if [ "$#" -ge 1 ]; then regions=("$@"); else mapfile -t regions < <(gwas_regions ukb); fi

for region in "${regions[@]}"; do
    echo "Checking Region: $region ..."
    prefix="$cohort_dir/$region/results/chr"
    output="$cohort_dir/$region/results/${region}_imputed_sumstats.txt"

    all_valid=true
    for i in $(seq 1 22); do
        chr_file="${prefix}${i}.txt"
        idx_file="${chr_file}.index"
        if [ ! -f "$chr_file" ]; then
            echo "  Missing data file: $chr_file"; all_valid=false
        elif [ ! -f "$idx_file" ]; then
            echo "  Missing index file: $idx_file"; all_valid=false
        else
            last_line=$(tail -n 1 "$idx_file")
            if [[ "$last_line" != *"Have completed the analyses of all chunks."* ]]; then
                echo "  Incomplete chunk analysis in: $idx_file"; all_valid=false
            fi
        fi
    done

    if [ "$all_valid" = false ]; then
        echo "  Skipping $region due to incomplete or missing files."; continue
    fi

    echo "  All checks passed. Merging $region..."
    head -n 1 "${prefix}1.txt" > "$output"
    for i in $(seq 1 22); do
        tail -n +2 "${prefix}${i}.txt" >> "$output"
    done
    echo "  Done -> $output"
done
