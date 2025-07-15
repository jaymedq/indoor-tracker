"Improve BLE AoA estimation by applying a moving median filter to remove outliers from one or more columns."

import pandas as pd
import numpy as np
import argparse
import os
from matplotlib import pyplot as plt
import math

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input",type=str,required=True)
    parser.add_argument("--columns",type=str,nargs='+',required=True)
    parser.add_argument("--window",type=int,default=7)
    parser.add_argument("--threshold",type=float,default=8.0)
    parser.add_argument("--output", nargs="?", type=str, default = "")

    args = parser.parse_args()
    if not os.path.exists(args.input):
        return
    print(f"Reading data from '{args.input}'...")
    df = pd.read_csv(args.input)
    original_count = len(df)

    missing_columns = [col for col in args.columns if col not in df.columns]
    if missing_columns:
        return
    print(f"N={args.window} T={args.threshold} to columns: {args.columns}...")

    combined_outlier_mask = pd.Series([False] * original_count, index=df.index)
    fig, axes = plt.subplots(math.ceil(len(args.columns)/2), 2, figsize=(12, 5 * len(args.columns)))
    plt.title(f'Histogram of Values ({args.input})')
    plt.ylabel('Frequency')
    plt.legend()
    for i, ax in  enumerate(axes.flatten()):
        if i >= len(args.columns):
            break
        col = args.columns[i]
        # plt.figure(figsize=(10, 5))
        ax.hist(df[col], bins=50, alpha=0.7, label=f'Kept {col} values', color='blue')
        # ax.xlabel(col)
        ax.grid(True)
        moving_median = df[col].rolling(window=args.window, min_periods=1, closed='left').median()
        deviation = np.abs(df[col] - moving_median)
        outliers_in_col = deviation > args.threshold
        combined_outlier_mask |= outliers_in_col
    plt.show()
    filtered_df = df[~combined_outlier_mask]
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count
    if original_count > 0:
        percentage_removed = (removed_count / original_count) * 100
        print(f"\nRemoved {removed_count} rows ({percentage_removed:.2f}%) identified as outliers")
    # subplot histogram of kept values for each column
    # for col in args.columns:
    #     if col in filtered_df.columns:
    #         #sub plot histogram of kept values
    #         plt.figure(figsize=(10, 5))
    #         plt.hist(filtered_df[col], bins=50, alpha=0.7, label=f'Kept {col} values', color='blue')
    #         plt.title(f'Histogram of Kept {col} Values ({args.input})')
    #         plt.xlabel(col)
    #         plt.ylabel('Frequency')
    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()
    filtered_df.to_csv(args.output if args.output else args.input, index=False)

if __name__ == "__main__":
    main()