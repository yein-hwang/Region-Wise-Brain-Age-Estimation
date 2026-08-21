"""Concatenate the 22 per-chromosome munged files into one .sumstats.gz.

Kept as it was run: the concatenate-and-write happens inside the per-file loop,
so the file is rewritten on every iteration and the last write is the complete
one. Left unchanged rather than tidied, because this is the code that produced
the published summary statistics.
"""
# -*- coding: utf-8 -*-

import pandas as pd
import gzip
import os
import argparse
from tqdm import tqdm


def main(input_dir, output_file, munge_file_name):

    input_dir = f"{input_dir}"
    output_file = f"{output_file}"

    print(f"Input File Path: {input_dir}")
    print(f"Output File Name: {output_file}")

    # build the per-chromosome file list (chr1 ~ chr22)
    filenames = ["{}/{}_chr{}_munge.sumstats.gz".format(input_dir, munge_file_name, i) for i in range(1, 23)]

    # add to the list only if the file exists
    data_list = []
    for file in tqdm(filenames):
        if os.path.exists(file):
            print("Found:", file)
            with gzip.open(file, 'rt') as f:
                df = pd.read_csv(f, sep="\t")
                data_list.append(df)
        else:
            print("Warning: File not found, skipping ->", file)

        # skip if there is no file to merge
        if not data_list:
            print("No valid files found")
            continue

        # combine all the data into a single DataFrame
        merged_data = pd.concat(data_list, ignore_index=True)

        # save the merged data as a .gz file
        with gzip.open(output_file, 'wt') as f:
            merged_data.to_csv(f, sep="\t", index=False)

        print("GWAS summary statistics successfully saved:", output_file)
        print()

if __name__ == "__main__":
    # Argument parser 
    parser = argparse.ArgumentParser(description="Merge GWAS Sumstats after Munge")
    parser.add_argument("--input_dir", type=str, required=True, help="output file directory")
    parser.add_argument("--output_file", type=str, required=True, help="output file")
    parser.add_argument("--munge_file_name", type=str, required=True, help="Munge file name prefix for reading")
    args = parser.parse_args()  # args

    main(args.input_dir, args.output_file, args.munge_file_name)