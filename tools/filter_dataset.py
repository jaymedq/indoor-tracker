import pandas as pd
import numpy  as np
import os
import argparse

import matplotlib.pyplot as plt

def filter_dataset(input_file, columns, threshold, output):
    df = pd.read_csv(input_file)
    original_count = len(df)
    combined_outlier_mask = pd.Series([False] * original_count, index=df.index)
    for col in columns:
        column_median = df[col].median()
        plt.plot(df.hist(col))
        deviation = np.abs(df[col] - column_median)
        outliers_in_col = deviation > args.threshold
        combined_outlier_mask |= outliers_in_col
    filtered_df = df[~combined_outlier_mask]
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count
    if original_count > 0:
        percentage_removed = (removed_count / original_count) * 100
        print(f"\nRemoved {removed_count} rows ({percentage_removed:.2f}%) identified as outliers")
    filtered_df.to_csv(args.output if args.output else args.input, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_file",type=str,required=True)
    parser.add_argument("--columns",type=str,nargs='+',required=True)
    parser.add_argument("--threshold",type=float,default=8.0)
    parser.add_argument("--output", nargs="?", type=str, default = "")
    args = parser.parse_args()
    if os.path.exists(args.input_file):
        filter_dataset(
            args.input_file, args.columns, args.threshold, args.output
        )