import pandas as pd
import numpy as np
from datetime import datetime

from pykalman import KalmanFilter
from sklearn.model_selection import ParameterGrid

# --- CONFIGURATION ---
# BLE and mmWave dataset filenames
BLE_DATASETS_PREFIX = "exported_"
MMW_DATASETS_SUFFIX = "_mmwave_data"
TEST_NAMES = [
    # "T029_MMW_A1_BLE_C3P1", #OUT OF FOV, DISCARD!!!
    "T030_MMW_A1_BLE_C3P2",
    "T031_MMW_A1_BLE_C3P3",
    "T032_MMW_A1_BLE_C3P4",
    "T033_MMW_A1_BLE_C3P5",
    "T034_MMW_A1_BLE_C4PA",
    # "T035_MMW_A1_BLE_CVP1",
    # "T036_MMW_A1_BLE_CVP2",
    # "T037_MMW_A1_BLE_CVP3",
    # "T038_MMW_A1_BLE_CVP4",
    # "T038_MMW_A1_BLE_CVP5",
    # "T040_MMW_A1_BLE_C4PV",
    # "T047_MMW_A1_BLE_C2P1",
    # "T048_MMW_A1_BLE_C2P2",
    # "T049_MMW_A1_BLE_C2P3",
    # "T050_MMW_A1_BLE_C2P4",
    # "T051_MMW_A1_BLE_C2P5",
    # "T052_MMW_A1_BLE_C4P4",
    # "T053_MMW_A1_BLE_C4P5",
    # "T054_MMW_A1_BLE_C4P6",
    # "T055_MMW_A1_BLE_C3P5",
    # "T056_MMW_A1_BLE_C3P4",
    # "T057_MMW_A1_BLE_C3P3",
    # "T058_MMW_A1_BLE_C3P2",
    # "T059_MMW_A1_BLE_C4P1",
    # "T060_MMW_A1_BLE_C1P5"
]
FINAL_MERGED_FILENAME = "ble_mmwave_fusion_all.csv"
CENTROID_OUTPUT_FILE = "output_transformed_centroid.csv"

# Experiment points
EXPERIMENT_POINTS = {
    "C1P1": [ 1.15 , -0.4, 1.78],
    "C1P2": [ 2.35 , -0.4, 1.78],
    "C1P3": [ 3.55 , -0.4, 1.78],
    "C1P4": [ 4.75 , -0.4, 1.78],
    "C1P5": [ 5.95 , -0.4, 1.78],
    "C2P1": [ 1.143, -4.462, 1.78],
    "C2P2": [ 2.343, -4.462, 1.78],
    "C2P3": [ 3.543, -4.462, 1.78],
    "C2P4": [ 4.745, -4.462, 1.78],
    "C2P5": [ 5.944, -4.462, 1.78],
    "C3P1": [ 1.102, -6.865, 1.78],
    "C3P2": [ 2.308, -6.865, 1.78],
    "C3P3": [ 3.503, -6.865, 1.78],
    "C3P4": [ 4.7, -6.865, 1.78],
    "C3P5": [ 5.9, -6.865, 1.78],
    "C4PA": [ 7.1, -6.865, 1.78],
    "CVP1": [ 1.102, -7.165, 1.78],
    "CVP2": [ 2.308, -7.165, 1.78],
    "CVP3": [ 3.503, -7.165, 1.78],
    "CVP4": [ 4.7, -7.165, 1.78],
    "C4P1":	[7.144, -0.863, 1.78],
    "C4P2":	[7.143, -2.015, 1.78],
    "C4P3":	[7.1, -3.215, 1.78],
    "C4P4":	[7.13, -4.462, 1.78],
    "C4P5":	[7.14, -5.618, 1.78],
    "C4P6":	[7.1, -6.865, 1.78],
    "CVP5": [ 5.9, -7.165, 1.78],
    "C4PV": [ 7.1, -7.165, 1.78],
    "PORTA": [ 8.61, -7.473, 1.78]
}

# Radar origin
radar_placement = np.array([0.995, -7.88, 1.70])
# radar_placement = np.array([0.98, -4.5, 1.78])


# --- STEP 1: SENSOR FUSION ---
def createTimeToDt(row):
    try:
        epoch_in_seconds = row["create_time"]
        return datetime.fromtimestamp(epoch_in_seconds).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(row)
        raise e

def fix_x_axis(row):
    return [row['x_ble'] * -1, row['y_ble'] * -1, row['x_ble_filter'] * -1, row['y_ble_filter'] * -1]


def fuse_datasets():

    BLE_DATASET_FILES = []
    MMWAVE_DATASET_FILES = []
    for test_name in TEST_NAMES:
        ble_file = f"{BLE_DATASETS_PREFIX}{test_name}"
        mmwave_file = f"{test_name}{MMW_DATASETS_SUFFIX}"
        BLE_DATASET_FILES.append(ble_file)
        MMWAVE_DATASET_FILES.append(mmwave_file)
    all_fused_data = []
    all_mmw_data = pd.DataFrame()

    for mmwave_file in MMWAVE_DATASET_FILES:
        mmwave_data = pd.read_csv(f"Results/{mmwave_file.replace(MMW_DATASETS_SUFFIX,'')}/{mmwave_file}.csv")
        mmwave_data["timestamp"] = pd.to_datetime(
            mmwave_data["timestamp"], format="%d/%m/%Y %H:%M:%S"
        )
        all_mmw_data = pd.concat([all_mmw_data, mmwave_data], ignore_index=True)

    for ble_file in BLE_DATASET_FILES:
        ble_data = pd.read_csv(f"Results/{ble_file.replace(BLE_DATASETS_PREFIX,'')}/{ble_file}.txt")
        if ble_data.empty:
            print(f"No data found in {ble_file}. Skipping fusion for this file.")
            continue
        ble_data["create_time"] = ble_data.apply(createTimeToDt, axis=1)

        ble_data["create_time"] = pd.to_datetime(
            ble_data["create_time"], format="%Y-%m-%d %H:%M:%S"
        )
        # rename x,y,z columns to x_ble, y_ble, z_ble
        ble_data.rename(columns={"x": "x_ble", "y": "y_ble", "x_filter": "x_ble_filter", "y_filter": "y_ble_filter", "z": "z_ble"}, inplace=True)
        ble_data[["x_ble", "y_ble", "x_ble_filter", "y_ble_filter"]] =  ble_data.apply(fix_x_axis, axis=1, result_type='expand')

        fusion_data = pd.merge(
            ble_data, all_mmw_data, left_on="create_time", right_on="timestamp", how="inner"
        )

        for point in EXPERIMENT_POINTS.keys():
            if f"BLE_{point}" in ble_file:
                fusion_data["real_xyz"] = [EXPERIMENT_POINTS[point]] * len(fusion_data)
                fusion_data["distance"] = np.linalg.norm(np.array(EXPERIMENT_POINTS[point]) - radar_placement)

        BLE_MMWAVE_FUSION_FILENAME = f"{ble_file}_mmwave_fusion.csv"
        fusion_data.to_csv(BLE_MMWAVE_FUSION_FILENAME, index=False)
        print(f"Fusion dataset saved as {BLE_MMWAVE_FUSION_FILENAME}")
        all_fused_data.append(fusion_data)

    final_fused_dataset = pd.concat(all_fused_data, ignore_index=True)
    final_fused_dataset['ble_xyz'] = final_fused_dataset.apply(lambda row: (row["x_ble"], row["y_ble"], 1.78), axis=1)
    final_fused_dataset['ble_xyz_filter'] = final_fused_dataset.apply(lambda row: (row["x_ble_filter"], row["y_ble_filter"], 1.78), axis=1)
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

def drop_static_points(row):
    # there are n x,y,z,velocity points in each row
    velocities = np.array(row['velocity'])
    x_members = np.array(row['x'])
    y_members = np.array(row['y'])
    z_members = np.array(row['z'])
    # Check if velocities are zero, if nth velocity is zero, drop x,y,z,velocity nth member.
    i = 0
    while (i < velocities.size):
        if velocities[i] == 0:
            x_members = np.delete(x_members, i)
            y_members = np.delete(y_members, i)
            z_members = np.delete(z_members, i)
            velocities = np.delete(velocities, i)
        else:
            i += 1
    row['x'] = x_members.tolist()
    row['y'] = y_members.tolist()
    row['z'] = z_members.tolist()
    row['velocity'] = velocities.tolist()
    return row[['x', 'y', 'z', 'velocity']]

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
    df['velocity'] = df['velocity'].apply(eval)

    df[['x', 'y', 'z']] = df.apply(transform_coordinates, axis=1, result_type='expand')
    #drop rows where x,y,z are empty lists
    df = df[df['x'].apply(lambda x: len(x) > 0) & df['y'].apply(lambda y: len(y) > 0) & df['z'].apply(lambda z: len(z) > 0)]
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

def evaluate_metrics(data):
    # Extract real positions and estimated positions
    real_x = [x[0] for x in data["real_xyz"].values]
    real_y = [x[1] for x in data["real_xyz"].values]
    real_z = [x[2] for x in data["real_xyz"].values]
    real_3d = np.array([real_x, real_y, real_z])

    triang_kf_x = data["x_ble"].values
    triang_kf_y = data["y_ble"].values
    triang_kf_z = 1.78  # Static z for estimation

    centroid_x = [x[0] for x in data["centroid_xyz"].values]
    centroid_y = [x[1] for x in data["centroid_xyz"].values]
    centroid_z = [x[2] for x in data["centroid_xyz"].values]

    # Convert to numpy arrays
    fusao_x = data["x_ble"].values
    fusao_y = data["y_ble"].values
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

def apply_kalman_filter(df, x_col, y_col, params=None):
    if params is None:
        params = {}
    observations = df[[x_col, y_col]].values
    kf = KalmanFilter(transition_matrices= params.get('transition_matrices', transition_matrix),
                       observation_matrices= params.get('observation_matrices', observation_matrix),
                       initial_state_mean= params.get('initial_state_mean', df[[x_col, y_col]].values[0]),
                       observation_covariance= params.get('observation_covariance', np.eye(2) * 0.01),
                       transition_covariance= params.get('transition_covariance', np.eye(2) * 0.01))
    kf.em(observations, n_iter=5)
    smoothed_states, _ = kf.smooth(observations)
    return smoothed_states

def track_to_track_fusion(df, mmw_weight, ble_weight):
    fusion_x = []
    fusion_y = []
    for i in range(len(df)):
        # mmwave worsens with distance, so multiply by inverse distance factor between 0.8 and 1
        mmw_distance_weight = 0.8 + 0.2 / (1 + df.loc[i, 'distance'])
        fused_x = mmw_distance_weight * mmw_weight * df.loc[i, "X_mmwave_kf"] + ble_weight * df.loc[i, "x_ble"]
        fused_y = mmw_distance_weight * mmw_weight * df.loc[i, "Y_mmwave_kf"] + ble_weight * df.loc[i, "y_ble"]
        
        fusion_x.append(fused_x)
        fusion_y.append(fused_y)
    
    return fusion_x, fusion_y

def optimize_kalman_filter(df, x_col, y_col, real_xyz_col):
    param_grid = {
        'transition_covariance': [np.eye(2) * 0.01, np.eye(2) * 0.1, np.eye(2) * 1],
        'observation_covariance': [np.eye(2) * 0.01, np.eye(2) * 0.1, np.eye(2) * 1]
    }
    best_params = None
    best_rmse = float('inf')

    for params in ParameterGrid(param_grid):
        kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            initial_state_mean=df[[x_col, y_col]].values[0],
            transition_covariance=params['transition_covariance'],
            observation_covariance=params['observation_covariance']
        )
        kf.em(df[[x_col, y_col]].values, n_iter=5)
        smoothed_states, _ = kf.smooth(df[[x_col, y_col]].values)
        df['X_kf'], df['Y_kf'] = smoothed_states[:, 0], smoothed_states[:, 1]

        real_xyz = np.array(df[real_xyz_col].values.tolist())
        kf_xyz = np.array([df['X_kf'], df['Y_kf'], [1.78] * len(df)]).T  # Assuming static z

        rmse = calculate_rmse(real_xyz, kf_xyz)
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    print(f"Best Kalman Filter parameters: {best_params}")
    return best_params

def optimize_track_to_track_fusion(df):
    param_grid = {
        'mmw_weight': [0.1, 0.3, 0.5, 0.7, 0.9],
        'ble_weight': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    best_weights = None
    best_rmse = float('inf')

    for params in ParameterGrid(param_grid):
        fusion_x, fusion_y = track_to_track_fusion(df, params['mmw_weight'], params['ble_weight'])
        df["X_fused_opt"], df["Y_fused_opt"] = fusion_x, fusion_y

        real_xyz = np.array(df["real_xyz"].values.tolist())
        fused_xyz = np.array([df["X_fused_opt"], df["Y_fused_opt"], [1.78] * len(df)]).T  # Assuming static z

        rmse = calculate_rmse(real_xyz, fused_xyz)
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = params

    print(f"Best Track-to-Track Fusion weights: {best_weights}")
    return best_weights

# --- MAIN PIPELINE EXECUTION ---
if __name__ == "__main__":
    print("Starting Sensor Fusion...")
    fused_data = fuse_datasets()

    print("\nProcessing Centroids...")
    centroid_data = process_centroids(fused_data)

    centroid_data["X_mmw_centroid"] = [x[0] for x in centroid_data["centroid_xyz"].values]
    centroid_data["Y_mmw_centroid"] = [x[1] for x in centroid_data["centroid_xyz"].values]

    print("\nOptimizing Kalman Filter...")
    # best_mmw_params = optimize_kalman_filter(centroid_data, "X_mmw_centroid", "Y_mmw_centroid", "real_xyz")

    print("\nApplying Kalman Filter...")
    # mmwave_kf = apply_kalman_filter(centroid_data, "X_mmw_centroid", "Y_mmw_centroid", best_mmw_params)

    # centroid_data["X_mmwave_kf"], centroid_data["Y_mmwave_kf"] = mmwave_kf[:, 0], mmwave_kf[:, 1]

    print("\nOptimizing Track-to-Track Fusion...")
    # best_fusion_weights = optimize_track_to_track_fusion(centroid_data)

    print("\nApplying Track-to-Track Fusion...")
    # fusion_x, fusion_y = track_to_track_fusion(centroid_data, best_fusion_weights['mmw_weight'], best_fusion_weights['ble_weight'])
    # centroid_data["X_fused"], centroid_data["Y_fused"] = fusion_x, fusion_y

    centroid_data.to_csv("FUSAO_PROCESSADA.csv", sep=';', index=False)

    print("\nEvaluating Error Metrics...")
    evaluate_metrics(centroid_data)
