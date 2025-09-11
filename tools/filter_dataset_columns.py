import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

def filter_dataset(input_file, columns, threshold, output):
    df_original = pd.read_csv(input_file)
    df_filtered = df_original.copy()
    original_count = len(df_original)

    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 5 * len(columns)))
    if len(columns) == 1:
        axes = [axes]

    total_replaced = 0

    for i, col in enumerate(columns):
        values = df_original[col].dropna()

        # Histogram for reference
        hist, bin_edges = np.histogram(values, bins=50)
        most_frequent_bin_index = np.argmax(hist)
        mode_bin_center = (bin_edges[most_frequent_bin_index] + bin_edges[most_frequent_bin_index + 1]) / 2

        # Create new filtered column
        filtered_col = f"{col}_filter"
        df_filtered[filtered_col] = df_original[col].copy()

        moving_mean = []
        replaced_count = 0

        for idx, val in enumerate(df_original[col]):
            if pd.isna(val):
                continue

            deviation = abs(val - mode_bin_center)
            if deviation > threshold:
                if moving_mean:  # replace in the new column only
                    df_filtered.at[idx, filtered_col] = np.mean(moving_mean)
                    replaced_count += 1
            else:
                moving_mean.append(val)

        total_replaced += replaced_count

        # Plot before vs after
        bins = np.histogram_bin_edges(values, bins=50)
        axes[i].hist(values, bins=bins, color='skyblue', alpha=0.6, label='Original', zorder=1)
        axes[i].hist(df_filtered[filtered_col].dropna(), bins=bins, color='lightcoral', alpha=0.6, label='Filtered', zorder=2)
        axes[i].set_title(f'Histogram for {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()

    fig.suptitle(f'TH: {threshold}, input: {os.path.basename(input_file)}', fontsize=16)
    plt.tight_layout()

    # Save histogram
    input_folder = os.path.dirname(input_file)
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    output_hist_path = os.path.join(input_folder, f"histogram_{input_basename}.png")
    plt.savefig(output_hist_path)
    print(f"Histogram saved to: {output_hist_path}")
    plt.close()

    print(f"\nAdded filtered columns with '_filter' suffix. Replaced {total_replaced} outlier values.")

    #Add replace rate to dataset
    df_filtered['filter_replace_rate'] = round(((total_replaced/len(df_original))*100),2)

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
