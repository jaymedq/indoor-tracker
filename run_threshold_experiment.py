
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools.constants import RADAR_PLACEMENT
from tools.calculate_mse_mae_rmse import calculate_rmse, calculate_mse

def run_command(command):
    """Runs a command and prints output/errors."""
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"--- ERROR running command: {' '.join(command)} ---")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        print("--- END ERROR ---")
    return result

def calculate_errors(group):
    """Calculates MSE and RMSE for different sensor types."""
    real_points = np.vstack(group['real_xyz'].apply(np.array))
    ble_points = np.column_stack((group['x_ble'], group['y_ble'], np.full(len(group), 1.78)))
    mmw_points = np.column_stack((group['mmw_x'], group['mmw_y'], np.full(len(group), 1.78)))
    fusion_points = np.vstack(group['sensor_fused_xyz_filter'].apply(np.array))
    filter_replace_rate = group['filter_replace_rate'].values[-1]
    
    real_x = real_points[0, 0]
    real_y = real_points[0, 1]

    return pd.Series({
        'MSE_BLE': calculate_mse(real_points, ble_points),
        'RMSE_BLE': calculate_rmse(real_points, ble_points),
        'MSE_MMW': calculate_mse(real_points, mmw_points),
        'RMSE_MMW': calculate_rmse(real_points, mmw_points),
        'MSE_Fusion': calculate_mse(real_points, fusion_points),
        'RMSE_Fusion': calculate_rmse(real_points, fusion_points),
        'x': real_x,
        'y': real_y,
        'filter_replace_rate': filter_replace_rate
    })

def safe_eval_list(s):
    """
    Safely evaluate a string representation of a list, correctly handling 'nan'.
    """
    try:
        # Provide a scope to eval where 'nan' is defined as numpy.nan
        return eval(s, {"nan": np.nan})
    except NameError:
        # If eval fails for any reason, return a list of NaNs
        return [np.nan, np.nan, np.nan]

thresholds = np.arange(0.05, 0.7, 0.05)
window_sizes = [3, 5, 7, 9, 11, 13, 15, 17] #must be odd for median filter

def main():
    """Main function to run the threshold experiment."""
    python_executable = sys.executable

    all_results = []

    # These are the files processed by the original .bat file
    test_files = [
        "Results/T136_MMW_A5_BLE_C2P2/T136_MMW_A5_BLE_C2P2_ble_data.csv",
        "Results/T137_MMW_A5_BLE_C2P3/T137_MMW_A5_BLE_C2P3_ble_data.csv",
        "Results/T138_MMW_A5_BLE_C2P4/T138_MMW_A5_BLE_C2P4_ble_data.csv",
        "Results/T139_MMW_A5_BLE_C2P5/T139_MMW_A5_BLE_C2P5_ble_data.csv",
        "Results/T146_MMW_A5_BLE_C3P2/T146_MMW_A5_BLE_C3P2_ble_data.csv",
        "Results/T147_MMW_A5_BLE_C3P3/T147_MMW_A5_BLE_C3P3_ble_data.csv",
        "Results/T148_MMW_A5_BLE_C3P4/T148_MMW_A5_BLE_C3P4_ble_data.csv",
        "Results/T149_MMW_A5_BLE_C3P5/T149_MMW_A5_BLE_C3P5_ble_data.csv",
        "Results/T153_MMW_A5_BLE_C3P2/T153_MMW_A5_BLE_C3P2_ble_data.csv",
        "Results/T154_MMW_A5_BLE_C3P3/T154_MMW_A5_BLE_C3P3_ble_data.csv",
        "Results/T155_MMW_A5_BLE_C3P4/T155_MMW_A5_BLE_C3P4_ble_data.csv",
        "Results/T156_MMW_A5_BLE_C3P5/T156_MMW_A5_BLE_C3P5_ble_data.csv",
        "Results/T157_MMW_A5_BLE_C2P2/T157_MMW_A5_BLE_C2P2_ble_data.csv",
        "Results/T158_MMW_A5_BLE_C2P3/T158_MMW_A5_BLE_C2P3_ble_data.csv",
        "Results/T159_MMW_A5_BLE_C2P4/T159_MMW_A5_BLE_C2P4_ble_data.csv",
        "Results/T160_MMW_A5_BLE_C2P5/T160_MMW_A5_BLE_C2P5_ble_data.csv"
    ]

    for window in window_sizes:
        print(f"- Starting experiment for WINDOW size: {window}")
        
        for threshold in thresholds:
            threshold = round(threshold, 2)
            print(f"- Running experiment for threshold: {threshold}")
            print("Running filter scripts...")
            for file_path in test_files:
                command = [
                    python_executable,
                    "tools/filter_dataset_columns.py",
                    "--input_file", file_path,
                    "--columns", "x", "y",
                    "--threshold", str(threshold),
                    "--window", str(window)
                ]
                run_command(command)

            # 2. Run pre-process and fuse scripts
            print("Running pre-process and fuse scripts...")
            run_command([python_executable, "tools/pre_process_dataset.py"])
            run_command([python_executable, "tools/fuse_sensor_data.py"])

            # 3. Calculate RMSE and store results
            print("Calculating RMSE...")
            try:
                data = pd.read_csv("fused_dataset.csv", sep=';')
                data["centroid_xyz"] = data["centroid_xyz"].apply(eval)
                data["real_xyz"] = data["real_xyz"].apply(eval)
                data['sensor_fused_xyz_filter'] = data['sensor_fused_xyz_filter'].apply(safe_eval_list)
                data['mmw_x'] = data['centroid_xyz'].apply(lambda x: x[0])
                data['mmw_y'] = data['centroid_xyz'].apply(lambda y: y[1])
                
                data['distance'] = data.apply(lambda row: np.linalg.norm(np.array(row["real_xyz"]) - RADAR_PLACEMENT), axis=1)
                
                results = data.groupby('distance').apply(calculate_errors).reset_index()
                results['threshold'] = threshold
                results['window'] = window
                all_results.append(results)
            except FileNotFoundError:
                print("Could not find fused_dataset.csv. Skipping RMSE calculation for this threshold.")
            except Exception as e:
                print(f"An error occurred during RMSE calculation: {e}")
                print("Reverting changes...")
                run_command(["git", "checkout", "--", "Results/"])
                raise e

        # 4. Revert changes
        print("Reverting changes...")
        run_command(["git", "checkout", "--", "Results/"])
    
    # 5. Combine and save all results
    print("\nSaving combined results...")
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        final_results.to_csv("threshold_window_experiment_results.csv", index=False)
        
        # 6. Plot errors by distance (now iterating over window sizes)
        print("Plotting results...")
        
        # Get the first distance value to plot (as in your original script)
        first_distance = final_results['distance'].min() # Or use a specific distance
        
        for window in window_sizes:
            window_data = final_results[final_results['window'] == window]
            distance_data = window_data[window_data['distance'] == first_distance]
            
            if distance_data.empty:
                 print(f"No data to plot for Window {window} at distance {first_distance}")
                 continue
                 
            # Create a new figure for each window size
            fig, ax1 = plt.subplots(figsize=(4.5, 3.5))
            ax2 = ax1.twinx()

            # Plotting lines
            # Discard Rate (Red, secondary Y-axis)
            ax2.plot(distance_data['threshold'] * 100, 
                     distance_data['filter_replace_rate'], 
                     'r-', marker='d', 
                     label='Discard rate')
                     
            # RMSE (Blue, primary Y-axis)
            ax1.plot(distance_data['threshold'] * 100, 
                     distance_data['RMSE_Fusion'], 
                     'b-', marker='o', 
                     label='RMSE')

            ax1.set_title(f"RMSE vs Discard Rate (Window = {window})", fontsize=14)
            ax1.set_xlabel('Threshold (cm)', fontsize=12)
            ax1.set_ylabel('RMSE (m)', color='b', fontsize=12)
            ax2.set_ylabel('Discard rate (%)', color='r', fontsize=12)
            
            # Combine legends and display
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper right')
            
            ax1.grid(True)
            ax1.tick_params(axis='y', colors='b')
            ax2.tick_params(axis='y', colors='r')
            fig.tight_layout()
            
            # Save the figure with window size in the filename
            fig.savefig(f"RMSE_DiscardRate_W{window}.eps", format='eps')
            fig.savefig(f"RMSE_DiscardRate_W{window}.png")
            print(f"Plot saved to RMSE_DiscardRate_W{window}.png")
    else:
        print("No results were generated.")

    print("--- Experiment finished ---")
    print("Results can be found in threshold_experiment_results.csv")
    print("Plot saved to RMSE_DiscardRate.eps")
    print("Plot saved to RMSE_DiscardRate.png")

if __name__ == "__main__":
    main()
