import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calculate_mse_mae_rmse import calculate_rmse, calculate_mse
from constants import EXPERIMENT_POINTS, RADAR_PLACEMENT
from plot_room_2d import POINTS_TO_CONSIDER

# Load dataset
data = pd.read_csv("fused_dataset.csv", sep=';')

def safe_eval_list(s):
    """
    Safely evaluate a string representation of a list, correctly handling 'nan'.
    """
    try:
        return eval(s, {"nan": np.nan})
    except NameError:
        return [np.nan, np.nan, np.nan]

# Ensure stringified lists are parsed correctly
data["centroid_xyz"] = data["centroid_xyz"].apply(eval)
data["real_xyz"] = data["real_xyz"].apply(eval)
data['sensor_fused_xyz'] = data['sensor_fused_xyz'].apply(eval)
data['sensor_fused_xyz_filter'] = data['sensor_fused_xyz_filter'].apply(safe_eval_list)
if "dl_sensor_fused_xyz" in data.columns:
    data['dl_sensor_fused_xyz'] = data['dl_sensor_fused_xyz'].apply(safe_eval_list)
data['mmw_x'] = data['centroid_xyz'].apply(lambda x: x[0])
data['mmw_y'] = data['centroid_xyz'].apply(lambda y: y[1])

def calculate_errors_by_point(group):
    real_points = np.vstack(group['real_xyz'].apply(np.array))
    ble_points = np.column_stack((group['x_ble'], group['y_ble'], np.full(len(group), 1.78)))
    mmw = np.column_stack((group['mmw_x'], group['mmw_y'], np.full(len(group), 1.78)))
    fusion_points = np.vstack(group['sensor_fused_xyz_filter'].apply(np.array))
    fusion_points_without_sliding_window_median_filter = np.vstack(group['sensor_fused_xyz'].apply(np.array))
    if "dl_sensor_fused_xyz" in group.columns:
        fusion_points_deep_learning = np.vstack(group['dl_sensor_fused_xyz'].apply(np.array))
    else:
        fusion_points_deep_learning = real_points
    real_x = real_points[0, 0]
    real_y = real_points[0, 1]

    mse_ble = calculate_mse(real_points, ble_points)
    mse_mmw = calculate_mse(real_points, mmw)
    mse_fusion = calculate_mse(real_points, fusion_points)
    mse_fusion_without_sliding_window_median_filter = calculate_mse(real_points, fusion_points_without_sliding_window_median_filter)
    mse_fusion_points_deep_learning = calculate_mse(real_points, fusion_points_deep_learning)

    rmse_ble = calculate_rmse(real_points, ble_points)
    rmse_mmw = calculate_rmse(real_points, mmw)
    rmse_fusion = calculate_rmse(real_points, fusion_points)
    rmse_fusion_without_sliding_window_median_filter = calculate_rmse(real_points, fusion_points_without_sliding_window_median_filter)
    rmse_fusion_points_deep_learning = calculate_rmse(real_points, fusion_points_deep_learning)

    return pd.Series({
        'MSE_BLE': mse_ble,
        'RMSE_BLE': rmse_ble,
        'MSE_MMW': mse_mmw,
        'RMSE_MMW': rmse_mmw,
        'MSE_Fusion': mse_fusion,
        'RMSE_Fusion': rmse_fusion,
        'MSE_FusionWOSWMF': mse_fusion_without_sliding_window_median_filter,
        'RMSE_FusionWOSWMF': rmse_fusion_without_sliding_window_median_filter,
        'MSE_DeepFusion': mse_fusion_points_deep_learning,
        'RMSE_DeepFusion': rmse_fusion_points_deep_learning,
        'x': real_x,
        'y': real_y,
        "samples": len(group)
    })

results = data.groupby('experiment_point').apply(calculate_errors_by_point).reset_index()
results = results.sort_values(by='experiment_point').reset_index(drop=True)
results.to_csv("error_by_experiment_point.csv", index=False)
point_labels = [f"{point.experiment_point}\nN={point.samples}" for point in results.itertuples(index=False)]

# Define methods and map for plotting
# methods = ['BLE', 'MMW', 'FusionWOSWMF', 'Fusion', 'DeepFusion']
methods = ['BLE', 'MMW', 'FusionWOSWMF', 'Fusion']
method_label_map = {
    "BLE": "BLE only",
    "MMW": "mmWave only",
    "FusionWOSWMF": "T2TF without SWMF",
    "Fusion": "Proposed T2TF scheme",
    "DeepFusion": "Deep Learning Fusion"
}
method_marker_map = {
    "BLE": 'd',
    "MMW": '*',
    "FusionWOSWMF": 'x',
    "Fusion": 'o',
    "DeepFusion": '.',
}

def get_hallway_index(point_label):
    """Extracts the hallway index (X in CXPY) from the experiment point label."""
    try:
        # Find 'C' and 'P' and extract the number in between
        start_index = point_label.find('C') + 1
        end_index = point_label.find('P')
        return int(point_label[start_index:end_index])
    except:
        return np.nan

results['hallway_index'] = results['experiment_point'].apply(get_hallway_index)

unique_hallways = sorted(results['hallway_index'].dropna().unique())

fig, ax = plt.subplots(figsize=(12, 6))
multiplier = 0
width = 0.15
x = np.arange(len(results))  # the label locations

for method in methods:
    offset = width * multiplier
    rects = ax.bar(x + offset, results[f'RMSE_{method}'], width, label=method_label_map.get(method, method))
    ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)
    multiplier += 1

print(f'Absolute improvement in RMSE from BLE:', results['RMSE_BLE'].mean() - results['RMSE_Fusion'].mean())
print(f'Absolute improvement in RMSE from MMW:', results['RMSE_MMW'].mean() - results['RMSE_Fusion'].mean())
print(f"Percentage improvement in RMSE from BLE: {(((results['RMSE_Fusion'].mean() - results['RMSE_BLE'].mean()) / results['RMSE_BLE'].mean()))*100:.2f}%")
print(f"Percentage improvement in RMSE from MMW: {(((results['RMSE_Fusion'].mean() - results['RMSE_MMW'].mean()) / results['RMSE_BLE'].mean()))*100:.2f}%")

ax.set_title('Root Mean Squared Error (RMSE) by Experiment Point', fontsize=16)
ax.set_xlabel('Experiment Point', fontsize=14)
ax.set_ylabel('RMSE [m]', fontsize=14)
ax.set_ylim((0, 1.4))
# Set x-ticks to be centered under the groups of bars and labeled with point_labels
ax.set_xticks(x + width * (len(methods) - 1) / 2)
ax.set_xticklabels(point_labels, rotation=45, ha='right')
ax.legend(loc='upper left')
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
fig.tight_layout()

# Save the figure with a dedicated filename for ALL points
fig.savefig("Resultado.eps", format='eps')
fig.savefig("Resultado.png")
print("Figure for All Points saved to Resultado.png")

for hallway in unique_hallways:
    hallway_data = results[results['hallway_index'] == hallway].copy()
    hallway_point_labels = hallway_data['experiment_point'].tolist()

    if hallway_data.empty:
        print(f"No data for Hallway C{hallway}. Skipping.")
        continue

    fig, ax = plt.subplots(figsize=(10, 6)) # New figure for each hallway
    multiplier = 0
    width = 0.15
    x = np.arange(len(hallway_data))  
    for method in methods:
        offset = width * multiplier
        ax.bar(x + offset, hallway_data[f'RMSE_{method}'], width, label=method_label_map.get(method, method))
        multiplier += 1

    print(f'\n--- Hallway C{hallway} RMSE Summary ---')
    for method in methods:
        print(f'Mean RMSE_{method} for C{hallway}:', hallway_data[f'RMSE_{method}'].mean())

    ax.set_title(f'Root Mean Squared Error (RMSE) by Experiment Point - Hallway C{hallway}', fontsize=16)
    ax.set_xlabel('Experiment Point', fontsize=14)
    ax.set_ylabel('RMSE [m]', fontsize=14)
    ax.set_ylim((0, 1.4)) # Keep consistent Y limit
    
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(hallway_point_labels, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    
    filename_eps = f"Resultado_Hallway_C{hallway}.eps"
    filename_png = f"Resultado_Hallway_C{hallway}.png"
    fig.savefig(filename_eps, format='eps')
    fig.savefig(filename_png)
    print(f"Figure for Hallway C{hallway} saved to {filename_png}")

plt.show()