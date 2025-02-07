from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


import numpy as np


def remove_nan_index(
    listA: np.array, listB: np.array, listC: np.array, listD: np.array
) -> tuple:
    # Stack the arrays along a new axis for easier indexing
    stacked = np.stack((listA, listB, listC, listD))  # Shape: (3, 3, 297)

    # Create a mask for valid columns (no NaNs across all three arrays)
    valid_mask = ~np.isnan(stacked).any(axis=(0, 1))

    # Apply the mask to filter valid columns
    return (
        listA[:, valid_mask],
        listB[:, valid_mask],
        listC[:, valid_mask],
        listD[:, valid_mask],
    )


def calculate_rmse(A: np.ndarray, B: np.ndarray) -> float:
    return np.sqrt(np.nanmean((A - B) ** 2))


def calculate_mse(A: np.ndarray, B: np.ndarray) -> float:
    return np.nanmean((A - B) ** 2)


def calculate_mae(A: np.ndarray, B: np.ndarray) -> float:
    return np.nanmean((A - B).__abs__())


def calculate_mse_mae_rmse(row: pd.DataFrame) -> dict:
    # Extract real positions and estimated positions
    real_x = [x[0] for x in row["real_xyz"].values]
    real_y = [x[1] for x in row["real_xyz"].values]
    real_z = [x[2] for x in row["real_xyz"].values]

    centroid_x = [x[0] for x in row["centroid_xyz"].values]
    centroid_y = [x[1] for x in row["centroid_xyz"].values]
    centroid_z = [x[2] for x in row["centroid_xyz"].values]

    triang_kf_x = row["X_est_TRIANG_KF"].values
    triang_kf_y = row["Y_est_TRIANG_KF"].values
    triang_kf_z = 1.78  # Static z for estimation

    fusao_x = row["X_est_FUSAO"].values
    fusao_y = row["Y_est_FUSAO"].values
    fusao_z = 1.78  # Static z for estimation

    # Convert to numpy arrays
    real_3d = np.array([real_x, real_y, real_z])
    triang_kf_3d = np.array([triang_kf_x, triang_kf_y, [triang_kf_z] * len(real_z)])
    fusao_3d = np.array([fusao_x, fusao_y, [fusao_z] * len(real_z)])
    centroid_3d = np.array([centroid_x, centroid_y, centroid_z])

    # Make sure NAN does not count.
    real_3d, triang_kf_3d, fusao_3d, centroid_3d = remove_nan_index(
        real_3d, triang_kf_3d, fusao_3d, centroid_3d
    )

    # Calculate MSE and MAE for CENTROID
    mse_centroid = calculate_mse(real_3d, centroid_3d)
    mae_centroid = calculate_mae(real_3d, centroid_3d)
    rmse_centroid = calculate_rmse(real_3d, centroid_3d)

    # Calculate MSE and MAE for TRIANG_KF
    mse_triang = calculate_mse(real_3d, triang_kf_3d)
    mae_triang = calculate_mae(real_3d, triang_kf_3d)
    rmse_triang = calculate_rmse(real_3d, triang_kf_3d)

    # Calculate MSE and MAE for FUSAO
    mse_fusao = calculate_mse(real_3d, fusao_3d)
    mae_fusao = calculate_mae(real_3d, fusao_3d)
    rmse_fusao = calculate_rmse(real_3d, fusao_3d)

    return {
        "mse_centroid": mse_centroid,
        "rmse_centroid": rmse_centroid,
        "mae_centroid": mae_centroid,
        "mse_triang": mse_triang,
        "rmse_triang": rmse_triang,
        "mae_triang": mae_triang,
        "mse_fusao": mse_fusao,
        "rmse_fusao": rmse_fusao,
        "mae_fusao": mae_fusao,
    }


if __name__ == "__main__":
    data = pd.read_csv("ble_mmwave_fusion_all.csv")
    if "real_xyz" in data.columns:
        data["real_xyz"] = data["real_xyz"].apply(eval)
    if "centroid_xyz" in data.columns:
        data["centroid_xyz"] = data["centroid_xyz"].apply(eval)

    data = calculate_mse_mae_rmse(data)
    # data.apply(calculate_mse_mae_rmse, axis=1)
    for key,value in data.items():
        print(key, value)
