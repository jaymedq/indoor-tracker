import pandas as pd
import numpy as np
from datetime import datetime

from pykalman import KalmanFilter

# --- CONFIGURATION ---
# BLE and mmWave dataset filenames
BLE_DATASET_FILES = [
    "resultado_RADAR_C3_PRD_P12_178_JM",
    "resultado_RADAR_C3_PRD_P13_178_JM",
    "resultado_RADAR_C3_PRD_P14_178_JM",
    "resultado_RADAR_C3_PRD_P15_178_JM",
    "resultado_RADAR_C3_PRD_PA_178_JM",
]
MMWAVE_DATASET_FILE = "output_lab_tag_14_10_24"
FINAL_MERGED_FILENAME = "ble_mmwave_fusion_all.csv"
CENTROID_OUTPUT_FILE = "output_transformed_centroid.csv"

# Experiment points
EXPERIMENT_POINTS = {
    "PA": [7.1, -6.865, 1.78],
    "P11": [1.102, -6.865, 1.78],
    "P12": [2.308, -6.865, 1.78],
    "P13": [3.503, -6.865, 1.78],
    "P14": [4.7, -6.865, 1.78],
    "P15": [5.9, -6.865, 1.78],
}

# Radar origin
radar_placement = np.array([0.995, -7.825, 1.70])


# --- STEP 1: SENSOR FUSION ---
def createTimeToDt(row):
    try:
        epoch_in_seconds = row["CreateTime"]
        return datetime.fromtimestamp(epoch_in_seconds).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(row)
        raise e

def fix_x_axis(row):
    return [row['X_est_TRIANG_KF'] * -1, row['Y_est_TRIANG_KF'] * -1, row['X_est_FUSAO'] * -1, row['Y_est_FUSAO'] * -1]


def fuse_datasets():
    all_fused_data = []
    mmwave_data = pd.read_csv(f"Results/{MMWAVE_DATASET_FILE}.csv")
    mmwave_data["timestamp"] = pd.to_datetime(
        mmwave_data["timestamp"], format="%d/%m/%Y %H:%M:%S"
    )

    for ble_file in BLE_DATASET_FILES:
        ble_data = pd.read_csv(f"Results/lab-experiment-results/{ble_file}.csv")
        ble_data["CreateTime"] = ble_data.apply(createTimeToDt, axis=1)

        ble_data["CreateTime"] = pd.to_datetime(
            ble_data["CreateTime"], format="%Y-%m-%d %H:%M:%S"
        )
        ble_data[["X_est_TRIANG_KF", "Y_est_TRIANG_KF", "X_est_FUSAO", "Y_est_FUSAO"]] =  ble_data.apply(fix_x_axis, axis=1, result_type='expand')

        fusion_data = pd.merge(
            ble_data, mmwave_data, left_on="CreateTime", right_on="timestamp", how="inner"
        )

        for point in EXPERIMENT_POINTS.keys():
            if f"_{point}_" in ble_file:
                fusion_data["real_xyz"] = f"{EXPERIMENT_POINTS[point]}"

        BLE_MMWAVE_FUSION_FILENAME = f"{ble_file}_mmwave_fusion.csv"
        fusion_data.to_csv(BLE_MMWAVE_FUSION_FILENAME, index=False)
        print(f"Fusion dataset saved as {BLE_MMWAVE_FUSION_FILENAME}")
        all_fused_data.append(fusion_data)

    final_fused_dataset = pd.concat(all_fused_data, ignore_index=True)
    final_fused_dataset.to_csv(FINAL_MERGED_FILENAME, index=False)
    print(f"Final merged dataset saved as {FINAL_MERGED_FILENAME}")
    return final_fused_dataset


# --- STEP 2: COORDINATE TRANSFORMATION & CENTROID CALCULATION ---
def transform_coordinates(row):
    points = np.array([row['x'], row['y'], row['z']]).T
    transformed = np.array([
        radar_placement[0] + points[:, 0],  # Add radar x
        radar_placement[1] - points[:, 1],  # Subtract radar y
        radar_placement[2] + points[:, 2]   # Add radar z
    ])
    return [transformed[0].tolist(), transformed[1].tolist(), transformed[2].tolist()]


def calculate_centroid(row):
    centroid_x = round(np.mean(row['x']), 2)
    centroid_y = round(np.mean(row['y']), 2)
    centroid_z = round(np.mean(row['z']), 2)
    return [centroid_x, centroid_y, centroid_z]


def process_centroids(fusion_data):
    df = fusion_data.copy()
    df['x'] = df['x'].apply(eval)
    df['y'] = df['y'].apply(eval)
    df['z'] = df['z'].apply(eval)

    df[['x', 'y', 'z']] = df.apply(transform_coordinates, axis=1, result_type='expand')
    df['centroid_xyz'] = df.apply(calculate_centroid, axis=1)

    df.to_csv(CENTROID_OUTPUT_FILE, index=False)
    print(f"Centroid calculation added to data at {CENTROID_OUTPUT_FILE}")
    return df


# --- STEP 3: ERROR METRICS CALCULATION ---
def remove_nan_index(listA: np.array, listB: np.array, listC: np.array, listD: np.array) -> tuple:
    stacked = np.stack((listA, listB, listC, listD))
    valid_mask = ~np.isnan(stacked).any(axis=(0, 1))
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


transition_matrix = [[1, 0], [0, 1]]  # Constant movement
observation_matrix = [[1, 0], [0, 1]]  # LOS

def apply_kalman_filter(df, x_col, y_col):
    observations = df[[x_col, y_col]].values
    kf = KalmanFilter(transition_matrices=transition_matrix,
                       observation_matrices=observation_matrix,
                       initial_state_mean=observations[0],
                       observation_covariance=np.eye(2),
                       transition_covariance=np.eye(2) * 0.01)
    smoothed_states, _ = kf.smooth(observations)
    return smoothed_states

def evaluate_metrics(data):
    data["real_xyz"] = data["real_xyz"].apply(eval)
    # data["centroid_xyz"] = data["centroid_xyz"].apply(eval)

    # Extract real positions and estimated positions
    real_x = [x[0] for x in data["real_xyz"].values]
    real_y = [x[1] for x in data["real_xyz"].values]
    real_z = [x[2] for x in data["real_xyz"].values]
    real_3d = np.array([real_x, real_y, real_z])

    triang_kf_x = data["X_est_TRIANG_KF"].values
    triang_kf_y = data["Y_est_TRIANG_KF"].values
    triang_kf_z = 1.78  # Static z for estimation

    centroid_x = [x[0] for x in data["centroid_xyz"].values]
    centroid_y = [x[1] for x in data["centroid_xyz"].values]
    centroid_z = [x[2] for x in data["centroid_xyz"].values]

    # Convert to numpy arrays
    fusao_x = data["X_est_FUSAO"].values
    fusao_y = data["Y_est_FUSAO"].values
    fusao_z = 1.78  # Static z for estimation
    real_z = [x[2] for x in data["real_xyz"].values]
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

    metrics = {
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

    print("Error Metrics:")
    for key,value in metrics.items():
        print(f"{key}: {value}")
    print("\n")

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

def apply_kalman_filter(df, x_col, y_col):
    observations = df[[x_col, y_col]].values
    kf = KalmanFilter(transition_matrices=transition_matrix,
                       observation_matrices=observation_matrix,
                       initial_state_mean=observations[0],
                       observation_covariance=np.eye(2),
                       transition_covariance=np.eye(2) * 0.01)
    smoothed_states, _ = kf.smooth(observations)
    return smoothed_states

def track_to_track_fusion(df):
    fusion_x = []
    fusion_y = []
    for i in range(len(df)):
        # After improving KF applications, optimize these weights.
        cov_ble = np.array([[0.5, 0], [0, 0.5]])
        cov_mmwave = np.array([[0.3, 0], [0, 0.3]])

        w_ble = np.linalg.inv(cov_ble)
        w_mmwave = np.linalg.inv(cov_mmwave)

        total_weight = w_ble + w_mmwave
        w_ble /= total_weight
        w_mmwave /= total_weight
        
        fused_x = w_ble[0, 0] * df.loc[i, "X_est_TRIANG_KF"] + w_mmwave[0, 0] * df.loc[i, "X_mmwave_kf"]
        fused_y = w_ble[1, 1] * df.loc[i, "Y_est_TRIANG_KF"] + w_mmwave[1, 1] * df.loc[i, "Y_mmwave_kf"]
        
        fusion_x.append(fused_x)
        fusion_y.append(fused_y)
    
    return fusion_x, fusion_y

# --- MAIN PIPELINE EXECUTION ---
if __name__ == "__main__":
    print("Starting Sensor Fusion...")
    fused_data = fuse_datasets()

    print("\nProcessing Centroids...")
    centroid_data = process_centroids(fused_data)

    print("\nApplying Kalman Filter...")
    centroid_data["X_mmw_centroid"] = [x[0] for x in centroid_data["centroid_xyz"].values]
    centroid_data["Y_mmw_centroid"] = [x[1] for x in centroid_data["centroid_xyz"].values]

    ble_kf = apply_kalman_filter(centroid_data, "X_est_TRIANG_KF", "Y_est_TRIANG_KF")
    mmwave_kf = apply_kalman_filter(centroid_data, "X_mmw_centroid", "Y_mmw_centroid")
    # Melhoria:
    # Calcular a aplicação do filtro para ter uma variavel de "peso de kalman" para cada distancia.
    # segmentar por distancia
    # aplicar o filtro de kalman para cada distancia
    # após aplicar a melhoria, otimizar os pesoss do ttf.
    # ponderar com RSSI

    centroid_data["X_mmwave_kf"], centroid_data["Y_mmwave_kf"] = mmwave_kf[:, 0], mmwave_kf[:, 1]

    fusion_x, fusion_y = track_to_track_fusion(centroid_data)
    centroid_data["X_fused"], centroid_data["Y_fused"] = fusion_x, fusion_y

    centroid_data.to_csv("FUSAO_PROCESSADA.csv", sep=';', index=False)

    print("\nEvaluating Error Metrics...")
    evaluate_metrics(centroid_data)
