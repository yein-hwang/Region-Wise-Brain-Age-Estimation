#!/bin/bash
# Build the gzipped, column-subset summary statistics uploaded to FUMA
# SNP2GENE: CHR POS MarkerID Allele1 Allele2 AF_Allele2 BETA SE p.value N
# (column order follows the SAIGE header, not the list above).
#
# The same merge/completeness checks as merge_chromosomes.sh apply.
# FUMA rejects uploads over 600 MB; the script warns when one is produced.
#
# Usage: bash make_fuma_input.sh <cohort_dir> [region ...]
set -uo pipefail
. "$(dirname "$0")/../config/common.sh"

cohort_dir="${1:?Usage: $0 <cohort_dir> [region ...]}"; shift
if [ "$#" -ge 1 ]; then regions=("$@"); else mapfile -t regions < <(gwas_regions ukb); fi

fuma_dir="$cohort_dir/fuma"
mkdir -p "$fuma_dir"

for region in "${regions[@]}"; do
    echo "Checking Region: $region ..."
    prefix="$cohort_dir/$region/results/chr"
    output="$fuma_dir/${region}_imputed_fuma.txt.gz"

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

    echo "  Merging & filtering for FUMA..."
    header=$(head -1 "${prefix}1.txt")
    col_indices=$(echo "$header" | awk -F'\t' '{
        for (i=1; i<=NF; i++) {
            if ($i=="CHR" || $i=="POS" || $i=="MarkerID" || $i=="Allele1" || $i=="Allele2" || $i=="AF_Allele2" || $i=="BETA" || $i=="SE" || $i=="p.value" || $i=="N")
                printf "%d,", i
        }
    }' | sed 's/,$//')

    (
        head -1 "${prefix}1.txt" | cut -f"$col_indices"
        for i in $(seq 1 22); do
            tail -n +2 "${prefix}${i}.txt" | cut -f"$col_indices"
        done
    ) | gzip > "$output"

    size=$(du -m "$output" | cut -f1)
    echo "  Done -> $output (${size}MB)"
    [ "$size" -gt 600 ] && echo "  WARNING: File exceeds the 600MB FUMA upload limit."
done
