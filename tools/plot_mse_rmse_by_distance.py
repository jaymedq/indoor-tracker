import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calculate_mse_mae_rmse import calculate_rmse, calculate_mse

# Load dataset
data = pd.read_csv("fused_dataset.csv", sep=';')

# Ensure stringified lists are parsed correctly
data["centroid_xyz"] = data["centroid_xyz"].apply(eval)
data["real_xyz"] = data["real_xyz"].apply(eval)
data['sensor_fused_xyz'] = data['sensor_fused_xyz'].apply(eval)
data['mmw_x'] = data['centroid_xyz'].apply(lambda x: x[0])
data['mmw_y'] = data['centroid_xyz'].apply(lambda y: y[1])

# Radar origin
radar_placement = np.array([0.995, -7.825, 1.70])

def calculate_distance(row):
    return np.linalg.norm(np.array(row["real_xyz"]) - radar_placement)

def calculate_errors(group):
    real_points = np.vstack(group['real_xyz'].apply(np.array))
    ble_kf_points = np.column_stack((group['x_ble'], group['y_ble'], np.full(len(group), 1.78)))
    mmw_kf = np.column_stack((group['mmw_x'], group['mmw_y'], np.full(len(group), 1.78)))
    fusion_points = np.vstack(group['sensor_fused_xyz'].apply(np.array))
    real_x = real_points[:, 0][0]
    real_y = real_points[0, 1]

    mse_ble_kf = calculate_mse(real_points, ble_kf_points)
    mse_mmw_kf = calculate_mse(real_points, mmw_kf)
    mse_fusion = calculate_mse(real_points, fusion_points)

    rmse_ble_kf = calculate_rmse(real_points, ble_kf_points)
    rmse_mmw_kf = calculate_rmse(real_points, mmw_kf)
    rmse_fusion = calculate_rmse(real_points, fusion_points)

    return pd.Series({
        'MSE_BLE_KF': mse_ble_kf,
        'RMSE_BLE_KF': rmse_ble_kf,
        'MSE_MMW_KF': mse_mmw_kf,
        'RMSE_MMW_KF': rmse_mmw_kf,
        'MSE_TTFKF_MMW_BLE_Fusion': mse_fusion,
        'RMSE_TTFKF_MMW_BLE_Fusion': rmse_fusion,
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
plt.figure(figsize=(12, 6))
methods = ['BLE_KF', 'MMW_KF', 'TTFKF_MMW_BLE_Fusion']
for method in methods:
    plt.plot(results['distance'], results[f'MSE_{method}'], marker='o', label=f'{method} MSE')
    print(f'MIN MSE_{method}:', np.min(results[f'MSE_{method}']))
    print(f'MAX MSE_{method}:', np.max(results[f'MSE_{method}']))

plt.title('Mean Squared Error (MSE) by Distance')
plt.xlabel('Distance')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

# Plotting RMSE by Distance
plt.figure(figsize=(12, 6))
for method in methods:
    plt.plot(results['distance'], results[f'RMSE_{method}'], marker='o', label=f'{method} RMSE')

plt.title('Root Mean Squared Error (RMSE) by Distance')
plt.xlabel('Distance')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()

from matplotlib import cm
from scipy.interpolate import griddata

# --- 3D Surface Plot of RMSE ---

fig = plt.figure(figsize=(21, 7))
methods = ['BLE_KF', 'MMW_KF', 'TTFKF_MMW_BLE_Fusion']

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
    fig.colorbar(surf, shrink=0.5, aspect=10, label='RMSE (meters)')

    # Formatting the plot
    ax.set_title(f'3D Surface of {method} RMSE', fontsize=14, pad=20)
    ax.set_xlabel('X Position (meters)', fontsize=10, labelpad=10)
    ax.set_ylabel('Y Position (meters)', fontsize=10, labelpad=10)
    ax.set_zlabel('RMSE (meters)', fontsize=10, labelpad=10)

    # Adjust view angle
    ax.view_init(elev=35, azim=-45)

plt.tight_layout()
plt.show()