
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

def main():
    """Main function to run the threshold experiment."""
    python_executable = sys.executable
    thresholds = np.arange(0.05, 0.7, 0.05)
    all_results = []

    # These are the files processed by the original .bat file
    test_files = [
        # "Results/T029_MMW_A1_BLE_C3P1/exported_T029_MMW_A1_BLE_C3P1.txt",
        # "Results/T030_MMW_A1_BLE_C3P2/exported_T030_MMW_A1_BLE_C3P2.txt",
        # "Results/T031_MMW_A1_BLE_C3P3/exported_T031_MMW_A1_BLE_C3P3.txt",
        "Results/T032_MMW_A1_BLE_C3P4/exported_T032_MMW_A1_BLE_C3P4.txt",
        "Results/T033_MMW_A1_BLE_C3P5/exported_T033_MMW_A1_BLE_C3P5.txt",
        "Results/T034_MMW_A1_BLE_C4PA/exported_T034_MMW_A1_BLE_C4PA.txt",
        # "Results/T035_MMW_A1_BLE_CVP1/exported_T035_MMW_A1_BLE_CVP1.txt",
        # "Results/T036_MMW_A1_BLE_CVP2/exported_T036_MMW_A1_BLE_CVP2.txt",
        # "Results/T037_MMW_A1_BLE_CVP3/exported_T037_MMW_A1_BLE_CVP3.txt",
        # "Results/T038_MMW_A1_BLE_CVP4/exported_T038_MMW_A1_BLE_CVP4.txt",
        # "Results/T038_MMW_A1_BLE_CVP5/exported_T038_MMW_A1_BLE_CVP5.txt",
        # "Results/T040_MMW_A1_BLE_C4PV/exported_T040_MMW_A1_BLE_C4PV.txt",
        # "Results/T047_MMW_A1_BLE_C2P1/exported_T047_MMW_A1_BLE_C2P1.txt",
        # "Results/T048_MMW_A1_BLE_C2P2/exported_T048_MMW_A1_BLE_C2P2.txt",
        # "Results/T049_MMW_A1_BLE_C2P3/exported_T049_MMW_A1_BLE_C2P3.txt",
        # "Results/T050_MMW_A1_BLE_C2P4/exported_T050_MMW_A1_BLE_C2P4.txt",
        # "Results/T051_MMW_A1_BLE_C2P5/exported_T051_MMW_A1_BLE_C2P5.txt",
        # "Results/T052_MMW_A1_BLE_C4P4/exported_T052_MMW_A1_BLE_C4P4.txt",
        # "Results/T053_MMW_A1_BLE_C4P5/exported_T053_MMW_A1_BLE_C4P5.txt",
        # "Results/T054_MMW_A1_BLE_C4P6/exported_T054_MMW_A1_BLE_C4P6.txt",
        # "Results/T055_MMW_A1_BLE_C3P5/exported_T055_MMW_A1_BLE_C3P5.txt",
        # "Results/T056_MMW_A1_BLE_C3P4/exported_T056_MMW_A1_BLE_C3P4.txt",
        # "Results/T057_MMW_A1_BLE_C3P3/exported_T057_MMW_A1_BLE_C3P3.txt",
        # "Results/T058_MMW_A1_BLE_C3P2/exported_T058_MMW_A1_BLE_C3P2.txt",
        # "Results/T059_MMW_A1_BLE_C4P1/exported_T059_MMW_A1_BLE_C4P1.txt",
        # "Results/T060_MMW_A1_BLE_C1P5/exported_T060_MMW_A1_BLE_C1P5.txt",
    ]

    for threshold in thresholds:
        threshold = round(threshold, 2)
        print(f"--- Running experiment for threshold: {threshold} ---")

        # 1. Run filter scripts
        print("Running filter scripts...")
        for file_path in test_files:
            command = [
                python_executable,
                "tools/filter_dataset_columns.py",
                "--input_file", file_path,
                "--columns", "x", "y",
                "--threshold", str(threshold)
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
    print("Saving combined results...")
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        final_results.to_csv("threshold_experiment_results.csv", index=False)
        final_results = pd.read_csv("threshold_experiment_results.csv")

        # 6. Plot errors by distance
        print("Plotting results...")
        plt.figure(figsize=(4.5,3.5))
        colors = plt.cm.jet(np.linspace(0, 1, len(thresholds)))
        
        distance_data = final_results[final_results['distance'] == final_results['distance'][0]]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax2.plot(distance_data['threshold']*100, distance_data['filter_replace_rate'], 'r-', marker='d', label= 'Discard rate')
        ax1.plot(distance_data['threshold']*100, distance_data['RMSE_Fusion'], 'b-', marker='o', label= 'RMSE')

        # plt.title(f"RMSE (Fusion) by Threshold for Distance = {final_results['distance'][2]}")
        ax1.set_xlabel('Threshold (cm)', fontsize=14)
        ax1.set_ylabel('RMSE (m)', color='b', fontsize=14)
        ax2.set_ylabel('Discard rate (%)', color='r', fontsize=14)
        # fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.8))
        ax1.grid(True)
        fig.tight_layout()
        fig.savefig("RMSE_DiscardRate.eps", format = 'eps')
        fig.savefig("RMSE_DiscardRate.png")
        # plt.show() # Commented out to prevent blocking in a non-interactive environment
    else:
        print("No results were generated.")

    print("--- Experiment finished ---")
    print("Results can be found in threshold_experiment_results.csv")
    print("Plot saved to RMSE_DiscardRate.eps")
    print("Plot saved to RMSE_DiscardRate.png")

if __name__ == "__main__":
    main()
