import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calculate_mse_mae_rmse import calculate_rmse, calculate_mse

# Load dataset
data = pd.read_csv("FUSAO_PROCESSADA.csv", sep=';')

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
    mmw_kf = np.column_stack((group['X_mmwave_kf'], group['Y_mmwave_kf'], np.full(len(group), 1.78)))
    fusion_points = np.column_stack((group['X_fused'], group['Y_fused'], np.full(len(group), 1.78)))

    mse_centroid = calculate_mse(real_points, centroid_points)
    mse_triang = calculate_mse(real_points, triang_points)
    mse_mmw_kf = calculate_mse(real_points, mmw_kf)
    mse_fusion = calculate_mse(real_points, fusion_points)

    rmse_centroid = calculate_rmse(real_points, centroid_points)
    rmse_triang = calculate_rmse(real_points, triang_points)
    rmse_fusion = calculate_rmse(real_points, fusion_points)
    rmse_mmw_kf = calculate_rmse(real_points, mmw_kf)

    return pd.Series({
        'MSE_MMW_Centroid': mse_centroid,
        'RMSE_MMW_Centroid': rmse_centroid,
        'MSE_BLE_Triang': mse_triang,
        'RMSE_BLE_Triang': rmse_triang,
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

import matplotlib.pyplot as plt

# Plotting MSE by Distance
plt.figure(figsize=(12, 6))
methods = ['MMW_Centroid', 'BLE_Triang', 'MMW_KF', 'TTFKF_MMW_BLE_Fusion']
for method in methods:
    plt.plot(results['distance'], results[f'MSE_{method}'], marker='o', label=f'{method} MSE')
    print(f'MIN MSE_{method}:',np.min(results[f'MSE_{method}']))
    print(f'MAX MSE_{method}:',np.max(results[f'MSE_{method}']))

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
