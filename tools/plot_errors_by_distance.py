import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calculate_mse_mae_rmse import calculate_rmse, calculate_mae

# Load dataset
data = pd.read_csv("ble_mmwave_fusion_all.csv")

# Ensure stringified lists are parsed correctly
data["centroid_xyz"] = data["centroid_xyz"].apply(eval)
data["real_xyz"] = data["real_xyz"].apply(eval)

# Calculate distance and error metrics
def calculate_distance_and_error(row):
    real = np.array(row["real_xyz"])
    centroid = np.array(row["centroid_xyz"])
    triang = np.array([row["X_est_TRIANG_KF"], row["Y_est_TRIANG_KF"], 1.78])
    fusion = np.array([row["X_est_FUSAO"], row["Y_est_FUSAO"], 1.78])

    # Calculate distance to real point
    distance = np.linalg.norm(real)

    # Calculate RMSE
    error_centroid = calculate_rmse(real, centroid)
    error_triang = calculate_rmse(real, triang)
    error_fusion = calculate_rmse(real, fusion)

    return pd.Series([distance, error_centroid, error_triang, error_fusion],
                     index=["distance", "error_centroid", "error_triang", "error_fusion"])

# Apply the function
data[["distance", "error_centroid", "error_triang", "error_fusion"]] = data.apply(
    calculate_distance_and_error, axis=1
)

# Plot Error vs Distance
plt.figure(figsize=(12, 8))
plt.scatter(data["distance"], data["error_centroid"], label="mmWave Centroid Error", color='red')
plt.scatter(data["distance"], data["error_triang"], label="BLE Triangulation Error", color='blue')
plt.scatter(data["distance"], data["error_fusion"], label="BLE Fusion Error", color='green')

plt.title("Error vs Distance for Different Estimation Methods")
plt.xlabel("Distance (m)")
plt.ylabel("Error (RMSE)")
plt.legend()
plt.grid(True)
plt.show()
