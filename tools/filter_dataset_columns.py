import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

def filter_dataset(input_file, columns, threshold, output):
    df_original = pd.read_csv(input_file)
    subset = df_original[columns]
    mask_to_drop = subset.isna().any(axis=1)
    df_filtered = df_original[~mask_to_drop].copy()
    original_count = len(df_original)

    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 5 * len(columns)))
    if len(columns) == 1:
        axes = [axes]

    total_replaced = 0
    replaced_indexes = []

    for i, col in enumerate(columns):
        values = df_filtered[col].dropna()

        # Histogram for reference
        hist, bin_edges = np.histogram(values, bins=50)
        most_frequent_bin_index = np.argmax(hist)
        mode_bin_center = (bin_edges[most_frequent_bin_index] + bin_edges[most_frequent_bin_index + 1]) / 2

        # Create new filtered column
        drop_filtered_col = f"{col}_filter"
        df_filtered[drop_filtered_col] = df_filtered[col].copy()
        
        replace_filtered_col = f'{col}_replace_filter'
        df_filtered[replace_filtered_col] = df_filtered[col].copy()

        moving_mean = []
        

        for index_label, row in df_filtered.iterrows():
            val = row[col] # Get the value for the specific column from the row

            if pd.isna(val):
                print(f'isna found at index {index_label} : value = {val}')
                continue

            deviation = abs(val - mode_bin_center)
            if deviation > threshold:
                if moving_mean:
                    # Use the correct 'index_label' for .loc
                    df_filtered.loc[index_label, replace_filtered_col] = np.mean(moving_mean)
                    replaced_indexes.append(index_label)
                # Use the correct 'index_label' for .loc
                df_filtered.loc[index_label, drop_filtered_col] = np.nan
            else:
                moving_mean.append(val)
        # deviation = abs(df_filtered[col] - mode_bin_center)
        # mask = deviation <= threshold   # keep only rows within threshold
        # df_filtered = df_filtered[mask]
        # replaced_count = (~mask).sum()  # number of removed rows

        # Plot before vs after
        bins = np.histogram_bin_edges(values, bins=50)
        axes[i].hist(values, bins=bins, color='skyblue', alpha=0.6, label='Original', zorder=1)
        axes[i].hist(df_filtered[replace_filtered_col].dropna(), bins=bins, color='lightcoral', alpha=0.6, label='Filtered', zorder=2)
        axes[i].set_title(f'Histogram for {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    total_replaced = len(set(replaced_indexes))

    fig.suptitle(f'TH: {threshold}, input: {os.path.basename(input_file)}', fontsize=16)
    plt.tight_layout()

    # Save histogram
    input_folder = os.path.dirname(input_file)
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    output_hist_path = os.path.join(input_folder, f"histogram_{input_basename}.png")
    plt.savefig(output_hist_path)
    print(f"Histogram saved to: {output_hist_path}")
    plt.close()

    print(f"\nAdded filtered columns with '_filter' suffix. Replaced {total_replaced} outlier values out of {original_count} total ({round(((total_replaced/original_count)*100),2)}%).")

    #Add replace rate to dataset
    df_filtered['filter_replace_rate'] = round(((total_replaced/original_count)*100),2)

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
