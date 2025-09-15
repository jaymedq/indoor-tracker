import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calculate_mse_mae_rmse import calculate_rmse, calculate_mse
from constants import RADAR_PLACEMENT

# Load dataset
data = pd.read_csv("fused_dataset.csv", sep=';')


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

# Ensure stringified lists are parsed correctly
data["centroid_xyz"] = data["centroid_xyz"].apply(eval)
data["real_xyz"] = data["real_xyz"].apply(eval)
data['sensor_fused_xyz'] = data['sensor_fused_xyz'].apply(eval)
data['sensor_fused_xyz_filter'] = data['sensor_fused_xyz_filter'].apply(safe_eval_list)
data['mmw_x'] = data['centroid_xyz'].apply(lambda x: x[0])
data['mmw_y'] = data['centroid_xyz'].apply(lambda y: y[1])

def calculate_distance(row):
    return np.linalg.norm(np.array(row["real_xyz"]) - RADAR_PLACEMENT)

def calculate_errors(group):
    real_points = np.vstack(group['real_xyz'].apply(np.array))
    ble_points = np.column_stack((group['x_ble'], group['y_ble'], np.full(len(group), 1.78)))
    mmw = np.column_stack((group['mmw_x'], group['mmw_y'], np.full(len(group), 1.78)))
    fusion_points = np.vstack(group['sensor_fused_xyz_filter'].apply(np.array))
    fusion_points_without_sliding_window_median_filter = np.vstack(group['sensor_fused_xyz'].apply(np.array))
    real_x = real_points[:, 0][0]
    real_y = real_points[0, 1]

    mse_ble = calculate_mse(real_points, ble_points)
    mse_mmw = calculate_mse(real_points, mmw)
    mse_fusion = calculate_mse(real_points, fusion_points)
    mse_fusion_without_sliding_window_median_filter = calculate_mse(real_points, fusion_points_without_sliding_window_median_filter)

    rmse_ble = calculate_rmse(real_points, ble_points)
    rmse_mmw = calculate_rmse(real_points, mmw)
    rmse_fusion = calculate_rmse(real_points, fusion_points)
    rmse_fusion_without_sliding_window_median_filter = calculate_rmse(real_points, fusion_points_without_sliding_window_median_filter)

    return pd.Series({
        'MSE_BLE': mse_ble,
        'RMSE_BLE': rmse_ble,
        'MSE_MMW': mse_mmw,
        'RMSE_MMW': rmse_mmw,
        'MSE_Fusion': mse_fusion,
        'RMSE_Fusion': rmse_fusion,
        'MSE_FusionWOSWMF': mse_fusion_without_sliding_window_median_filter,
        'RMSE_FusionWOSWMF': rmse_fusion_without_sliding_window_median_filter,
        'x': real_x,
        'y': real_y
    })

# Calculate distances
data['distance'] = data.apply(calculate_distance, axis=1)

# Group by discrete distance and calculate errors
results = data.groupby('distance').apply(calculate_errors).reset_index()

# Save results to CSV
results.to_csv("error_by_distance.csv", index=False)

# Plotting MSE by Distance
plt.figure(figsize=(15, 8))
methods = ['BLE', 'MMW', 'FusionWOSWMF', 'Fusion']
method_label_map = {
    "BLE": "BLE only",
    "MMW": "mmWave only",
    "FusionWOSWMF": "T2TF without SWMF",
    "Fusion": "Proposed T2TF scheme"
}
method_marker_map = {
    "BLE": 'd',
    "MMW": '*',
    "FusionWOSWMF": 'x',
    "Fusion": 'o',
}
for method in methods:
    plt.plot(results['distance'], results[f'MSE_{method}'], marker='o', label=f'{method} MSE')
    # print(f'MIN MSE_{method}:', np.min(results[f'MSE_{method}']))
    # print(f'MAX MSE_{method}:', np.max(results[f'MSE_{method}']))

# plt.title('Mean Squared Error (MSE) by Distance')
plt.xlabel('Distance', fontsize = 14)
plt.ylabel('MSE', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# Plotting RMSE by Distance
fig, ax = plt.subplots()
for method in methods:
    ax.plot(results['distance'], results[f'RMSE_{method}'], label=f'{method_label_map.get(method)} RMSE', marker= method_marker_map.get(method))
    print(f'MIN RMSE_{method}:', np.min(results[f'RMSE_{method}']))
    print(f'MAX RMSE_{method}:', np.max(results[f'RMSE_{method}']))
print(f'Absolute improvement in RMSE from BLE:', results['RMSE_BLE'].mean() - results['RMSE_Fusion'].mean())
print(f'Absolute improvement in RMSE from MMW:', results['RMSE_MMW'].mean() - results['RMSE_Fusion'].mean())
print(f"Percentage improvement in RMSE from BLE: {(((results['RMSE_Fusion'].mean() - results['RMSE_BLE'].mean()) / results['RMSE_Fusion'].mean()))*100}%")
print(f"Percentage improvement in RMSE from MMW: {(((results['RMSE_Fusion'].mean() - results['RMSE_MMW'].mean()) / results['RMSE_Fusion'].mean()))*100}%")

# plt.title('Root Mean Squared Error (RMSE) by Distance')
ax.set_xlabel('Distance [m]', fontsize=14)
ax.set_ylabel('RMSE [m]', fontsize=14)
ax.grid(True)
fig.legend(loc="upper right")
fig.tight_layout()
fig.show()
fig.savefig("Resultado.png")

from matplotlib import cm
from scipy.interpolate import griddata

# --- 3D Surface Plot of RMSE ---

fig = plt.figure(figsize=plt.figaspect(0.5))
methods = ['BLE', 'MMW', 'Fusion']

# Add small noise to y to avoid singular matrix error in griddata
results['y'] += np.random.normal(0, 1e-4, len(results['y']))

# Prepare data for interpolation
points = results[['x', 'y']].values
grid_x, grid_y = np.mgrid[
    0:9:100j,
    results['y'].min():results['y'].max():100j
]

for i, method in enumerate(methods):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    values = results[f'RMSE_{method}'].values

    # Interpolate the Z values (RMSE) onto the grid
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

    # Plot the surface
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=cm.viridis_r, linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.4, location='left')

    # Formatting the plot
    ax.set_title(f'3D Surface of {method} RMSE', fontsize=14, pad=20)
    ax.set_xlabel('X Position (m)', fontsize=10, labelpad=10)
    ax.set_ylabel('Y Position (m)', fontsize=10, labelpad=10)
    ax.set_zlabel('RMSE (m)', fontsize=10, labelpad=10)

    # Adjust view angle
    ax.view_init(elev=35, azim=-45)

plt.tight_layout(w_pad=7)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)
plt.show()