#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Split the GWAS Catalog telomere-length summary statistics into per-chromosome
# LDSC inputs. External-trait counterpart of 01_preprocess.py.
#
# The input carries GWAS Catalog column names (rs_id / effect_allele /
# other_allele / effect_allele_frequency / beta / standard_error / chromosome).
# They are renamed to the SAIGE names the rest of 04_ldsc/ uses, so
# effect_allele becomes Allele2 and 02_munge.sh's --a1 Allele2 puts LDSC's A1
# on the effect allele.
#
# Source file: LDSC_TELOMERE_RAW_SUMSTATS in config/paths.env.
#
# Usage:
#   python 00_preprocess_telomere.py \
#       --save_path         "$LDSC_WORK_DIR/files/telomere_pc1" \
#       --save_file_name    telomere_pc1 \
#       --gwas_results_path "$LDSC_TELOMERE_RAW_SUMSTATS"
#

import pandas as pd
import os
from tqdm import tqdm
import argparse



def main(save_path, save_file_name, gwas_results_path):
    file_name = f"_{save_file_name}_GWAS_SummaryStatistics.txt"
    snp_name = f"_{save_file_name}_GWAS_SummaryStatistics_snplist.txt"


    df = pd.read_csv(gwas_results_path, sep="\t", low_memory=False)  
    print(f"Current Data Frame: {gwas_results_path}")
    print(f"Current Data Frame Columns: {df.columns.tolist()}")
    print(f"Dataset Size: {len(df)}")
    # rare variants filtering
        # df["AF_Allele2"] > 0.01 --> if the reference allele is too rare the estimate is unreliable
        # df["AF_Allele2"] < 0.99 --> drops extremely rare variants, alternative allele at or below 0.001 (0.1%)
        # rename chr_df columns: effect_allele -> Allele2, reference_allele -> Allele1
    # Create a copy after filtering to avoid SettingWithCopyWarning
    new_df = df[(df["effect_allele_frequency"] > 0.01) & (df["effect_allele_frequency"] < 0.99)].copy()
    new_df.rename(columns={'rs_id': 'MarkerID', 'other_allele': 'Allele1', 'effect_allele': 'Allele2'}, inplace=True)
    # find the column indices of Allele1 and Allele2
    cols = new_df.columns.tolist()
    idx1 = cols.index('Allele1')
    idx2 = cols.index('Allele2')

    # swap the positions of the two columns
    cols[idx1], cols[idx2] = cols[idx2], cols[idx1]
    new_df = new_df[cols]
    
    for i in tqdm(range(1, 23)):
        chr_df = new_df[new_df["chromosome"] == i]

        output_file = os.path.join(save_path, f"chr{i}{file_name}")
        snp_file = os.path.join(save_path, f"chr{i}{snp_name}")

        # Generate SNP list from a copy (the SNP column has already been created)
        snp = chr_df.copy()
        snp['SNP'] = snp["MarkerID"]
        # SAIGE/GWAS-Catalog effect allele -> LDSC A1. Do NOT swap back.
        snp["A1"] = snp["Allele2"]  # effect allele (GWAS-Catalog effect_allele)
        snp["A2"] = snp["Allele1"]  # non-effect allele (other_allele)
        snplist = snp[["SNP", "A1", "A2"]]

        # Save the files
        chr_df.to_csv(output_file, sep="\t", index=False, header=True)
        snplist.to_csv(snp_file, sep="\t", index=False, header=True)

    print(f"Preprocessing Finished.\n")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Process LDSC analysis")
    parser.add_argument("--save_path", type=str, required=True, help="save path for output files")
    parser.add_argument("--save_file_name", type=str, required=True, help="output file name prefix")
    parser.add_argument("--gwas_results_path", type=str, required=True, help="Summary Statistics file path")
    args = parser.parse_args()  # args

    main(args.save_path, args.save_file_name, args.gwas_results_path)