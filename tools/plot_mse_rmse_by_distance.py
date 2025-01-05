import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calculate_mse_mae_rmse import calculate_rmse, calculate_mse

# Load dataset
data = pd.read_csv("Results/ble_mmwave_fusion_all.csv")

# Ensure stringified lists are parsed correctly
data["centroid_xyz"] = data["centroid_xyz"].apply(eval)
data["real_xyz"] = data["real_xyz"].apply(eval)

def calculate_distance(row):
    real = np.array(row["real_xyz"])
    return np.linalg.norm(real)

def calculate_errors(group):
    real_points = np.vstack(group['real_xyz'].apply(np.array))
    centroid_points = np.vstack(group['centroid_xyz'].apply(np.array))
    triang_points = np.column_stack((group['X_est_TRIANG_KF'], group['Y_est_TRIANG_KF'], np.full(len(group), 1.78)))
    fusion_points = np.column_stack((group['X_est_FUSAO'], group['Y_est_FUSAO'], np.full(len(group), 1.78)))
    
    mse_centroid = calculate_mse(real_points, centroid_points)
    mse_triang = calculate_mse(real_points, triang_points)
    mse_fusion = calculate_mse(real_points, fusion_points)
    
    rmse_centroid = calculate_rmse(real_points, centroid_points)
    rmse_triang = calculate_rmse(real_points, triang_points)
    rmse_fusion = calculate_rmse(real_points, fusion_points)
    
    return pd.Series({
        'MSE_MMW_Centroid': mse_centroid,
        'RMSE_MMW_Centroid': rmse_centroid,
        'MSE_BLE_Triang': mse_triang,
        'RMSE_BLE_Triang': rmse_triang,
        'MSE_BLE_Fusion': mse_fusion,
        'RMSE_BLE_Fusion': rmse_fusion
    })

# Calculate distances
data['distance'] = data.apply(calculate_distance, axis=1)

# Group by discrete distance and calculate errors
results = data.groupby('distance').apply(calculate_errors).reset_index()

# Save results to CSV
results.to_csv("Results/error_by_distance.csv", index=False)

import matplotlib.pyplot as plt

# Plotting MSE by Distance
plt.figure(figsize=(12, 6))
methods = ['MMW_Centroid', 'BLE_Triang', 'BLE_Fusion']
for method in methods:
    plt.plot(results['distance'], results[f'MSE_{method}'], marker='o', label=f'{method} MSE')

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
