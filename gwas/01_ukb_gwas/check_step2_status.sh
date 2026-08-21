#!/bin/bash
# Report per-chromosome Step 2 status for one region, by reading its logs.
# Usage: bash check_step2_status.sh <cohort_dir> <region>
#        bash check_step2_status.sh "$GWAS_UKB_DIR" caudate
set -uo pipefail

log_dir="${1:?Usage: $0 <cohort_dir> <region>}/${2:?Usage: $0 <cohort_dir> <region>}/logs"
[ -d "$log_dir" ] || { echo "ERROR: '$log_dir' is not a directory."; exit 1; }

echo "Checking log files in: $log_dir"
echo "----------------------------------------"

mapfile -t log_files < <(find "$log_dir" -maxdepth 1 -name "*chr*.log")
if [ ${#log_files[@]} -eq 0 ]; then
    echo "No step-2 logs found in $log_dir"; exit 0
fi

log_info=()
for logfile in "${log_files[@]}"; do
    chr=$(basename "$logfile" | grep -o 'chr[0-9]\{1,2\}' | grep -o '[0-9]\{1,2\}' | head -n 1)
    [[ "$chr" =~ ^[0-9]+$ ]] && log_info+=("$chr|$logfile")
done

IFS=$'\n' sorted_logs=($(sort -t '|' -k1,1n <<<"${log_info[*]}")); unset IFS

for entry in "${sorted_logs[@]}"; do
    chr_num="${entry%%|*}"; logfile="${entry#*|}"
    if grep -q "Analysis done" "$logfile"; then
        echo "OK       chr$chr_num - $(basename "$logfile")"
    elif grep -q "Execution halted" "$logfile"; then
        echo "ERROR    chr$chr_num - $(basename "$logfile")  (execution halted)"
    else
        echo "RUNNING  chr$chr_num - $(basename "$logfile")"
    fi
done
echo "----------------------------------------"
