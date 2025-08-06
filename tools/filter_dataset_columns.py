import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

def filter_dataset(input_file, columns, threshold, output):
    df_original = pd.read_csv(input_file)
    original_count = len(df_original)

    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 5 * len(columns)))
    if len(columns) == 1:
        axes = [axes]

    combined_outlier_mask = pd.Series([False] * original_count, index=df_original.index)

    # First pass: compute outliers across all columns
    for col in columns:
        # Drop NaN values for safety
        values = df_original[col].dropna()

        # Create histogram
        hist, bin_edges = np.histogram(values, bins=50)

        # Find the most frequent bin
        most_frequent_bin_index = np.argmax(hist)
        mode_bin_center = (bin_edges[most_frequent_bin_index] + bin_edges[most_frequent_bin_index + 1]) / 2

        # Filter based on distance from mode bin center
        deviation = np.abs(df_original[col] - mode_bin_center)
        outliers_in_col = deviation > threshold
        combined_outlier_mask |= outliers_in_col

    df_filtered = df_original[~combined_outlier_mask]

    # Plot before and after for each column
    for i, col in enumerate(columns):
        original_values = df_original[col].dropna()
        filtered_values = df_filtered[col].dropna()

        bins = np.histogram_bin_edges(original_values, bins=50)

        axes[i].hist(original_values, bins=bins, color='skyblue', alpha=0.6, label='Before Filter', zorder=1)
        axes[i].hist(filtered_values, bins=bins, color='lightcoral', alpha=0.6, label='After Filter', zorder=2)
        
        axes[i].set_title(f'Histogram for {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()

    fig.suptitle(f'TH: {threshold}, input: {os.path.basename(input_file)}', fontsize=16)
    plt.tight_layout()
    
    # Build path for saving histogram image
    input_folder = os.path.dirname(input_file)
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    output_hist_path = os.path.join(input_folder, f"histogram_{input_basename}.png")
    
    plt.savefig(output_hist_path)
    print(f"Histogram saved to: {output_hist_path}")
    
    plt.close()

    filtered_count = len(df_filtered)
    removed_count = original_count - filtered_count
    if original_count > 0:
        percentage_removed = (removed_count / original_count) * 100
        print(f"\nRemoved {removed_count} rows ({percentage_removed:.2f}%) identified as outliers")

    # Save the filtered data
    if output:
        df_filtered.to_csv(output, index=False)
    else:
        df_filtered.to_csv(input_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--columns", type=str, nargs='+', required=True)
    parser.add_argument("--threshold", type=float, default=8.0)
    parser.add_argument("--output", nargs="?", type=str, default="")
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        filter_dataset(
            args.input_file, args.columns, args.threshold, args.output
        )
