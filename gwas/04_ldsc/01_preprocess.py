"""Split one SAIGE summary-statistics file into per-chromosome LDSC inputs.

Applies the common-variant filter used throughout: 0.01 < AF_Allele2 < 0.99.
For each autosome it writes the filtered rows plus a three-column SNP list
(SNP/A1/A2) that `munge_sumstats.py --merge-alleles` consumes.

A1 = Allele2, A2 = Allele1. SAIGE reports BETA and AF_Allele2 on Allele2, so
Allele2 is the effect allele and must occupy the A1 slot that LDSC interprets
as the signed-statistic reference.
"""
import pandas as pd
import os
from tqdm import tqdm
import argparse


# Argument handling: take the UCB_BETA and TEMPERATURE values as input arguments
parser = argparse.ArgumentParser(description="Preprocessing for LDSC")
parser.add_argument("--save_path", type=str, default=".", help="output file save directory")
parser.add_argument("--save_file_name", type=str, default="default", help="output file prefix")
parser.add_argument("--gwas_results_path", type=str, default="", help="GWAS sumstat file directory")
args = parser.parse_args()

file_name = f"_{args.save_file_name}_GWAS_SummaryStatistics.txt"
snp_name = f"_{args.save_file_name}_GWAS_SummaryStatistics_snplist.txt"

df = pd.read_csv(args.gwas_results_path, sep="\t", low_memory=False)
# SAIGE writes CHR as it appears in the genotype files, which for some releases
# means zero-padded strings ("01"). Compare on the numeric value so the split
# below works either way; the filter and the split are otherwise unchanged.
df["CHR"] = pd.to_numeric(df["CHR"], errors="coerce")
# rare variants filtering
    # df["AF_Allele2"] > 0.01 --> if the reference allele is too rare the estimate is unreliable
    # df["AF_Allele2"] < 0.99 --> an alternative allele at or below 0.001 (0.1%) is an extremely rare variant (Rare Variant)
new_df = df[(df["AF_Allele2"] > 0.01) & (df["AF_Allele2"] < 0.99)]

for i in tqdm(range(1, 23)):
    chr_df = new_df[new_df["CHR"] == i]

    output_file = os.path.join(args.save_path, f"chr{i}{file_name}")
    snp_file = os.path.join(args.save_path, f"chr{i}{snp_name}")

    # generate the SNP list
    snp = chr_df.copy()
    snp["SNP"] = snp["MarkerID"]
    snp["A1"] = snp["Allele2"]  # effect allele (SAIGE BETA is on Allele2)
    snp["A2"] = snp["Allele1"]  # non-effect allele
    snplist = snp[["SNP", "A1", "A2"]]

    # save the files
    chr_df.to_csv(output_file, sep="\t", index=False, header=True, quoting=False)
    snplist.to_csv(snp_file, sep="\t", index=False, header=True, quoting=False)

print(f"Preprocessing Finished.\n")