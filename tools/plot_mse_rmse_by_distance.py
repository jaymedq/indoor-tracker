import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calculate_mse_mae_rmse import calculate_rmse, calculate_mse

# Load dataset
data = pd.read_csv("fused_dataset.csv", sep=';')

# Ensure stringified lists are parsed correctly
data["centroid_xyz"] = data["centroid_xyz"].apply(eval)
data["real_xyz"] = data["real_xyz"].apply(eval)
data["sensor_fused_xyz"] = data["sensor_fused_xyz"].apply(eval)

# Radar origin
radar_placement = np.array([0.995, -7.825, 1.70])

def calculate_distance(row):
    return np.linalg.norm(np.array(row["real_xyz"]) - radar_placement)

def calculate_errors(group):
    real_points = np.vstack(group['real_xyz'].apply(np.array))
    ble_kf_points = np.column_stack((group['x_ble_kf'], group['y_ble_kf'], np.full(len(group), 1.78)))
    mmw_kf = np.column_stack((group['x_mmw_kf'], group['y_mmw_kf'], np.full(len(group), 1.78)))
    fusion_points = np.vstack(group['sensor_fused_xyz'].apply(np.array))

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
        'RMSE_TTFKF_MMW_BLE_Fusion': rmse_fusion
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