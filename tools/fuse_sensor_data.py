import numpy as np
import pandas as pd
from constants import RADAR_PLACEMENT

def track_to_track_fusion(mean1, cov1, mean2, cov2):
    """
    Fuse two position estimates with covariance weighting.
    
    mean1, mean2: 2D position estimates (x, y)
    cov1, cov2: 2x2 covariance matrices
    """
    cov_inv1 = np.linalg.inv(cov1)
    cov_inv2 = np.linalg.inv(cov2)

    # Combined covariance
    fused_cov = np.linalg.inv(cov_inv1 + cov_inv2)

    # Weighted sum of means
    fused_mean = np.dot(fused_cov, np.dot(cov_inv1, mean1) + np.dot(cov_inv2, mean2))

    return fused_mean, fused_cov

df = pd.read_csv("FUSAO_PROCESSADA.csv", sep=";")
# Drop np.isnan values
df: pd.DataFrame = df[~np.isnan(df[["x_ble"]]).any(axis=1)]
df = df.reset_index(drop=True)
df["centroid_xyz"] = df["centroid_xyz"].apply(eval)
df["ble_xyz_filter"] = df["ble_xyz_filter"].apply(eval)
df["real_xyz"] = df["real_xyz"].apply(eval)

fused_values = []

# Process each row (assumed to be sequential in time)
def fuse_sensor_data(row, mmw_cov, ble_cov):
    # try:
    # Parse the string representations of the measurements
    mm_meas = row["centroid_xyz"]
    ble_meas = row["ble_xyz_filter"]
    if np.nan in mm_meas or np.nan in ble_meas:
        raise ValueError("Invalid measurements")

    # Get estimates and covariances
    mean_mmwave = mm_meas[:2]
    mean_ble = ble_meas[:2]

    # Fuse estimates
    fused_position, fused_covariance = track_to_track_fusion(mean_mmwave, mmw_cov, mean_ble, ble_cov)
    fused_position = fused_position.tolist()
    fused_position.append(1.78)  # Add Z coordinate

    print("Fused Position:", fused_position)
    print("Fused Covariance:\n", fused_covariance)

    return fused_position, fused_covariance

def calculate_distance(row):
    return np.linalg.norm(np.array(row["real_xyz"]) - RADAR_PLACEMENT)

# Append the fused data to the dataframe.
df['distance'] = df.apply(calculate_distance, axis=1)

grouped_dict = {key: group for key, group in df.groupby("distance")}
for key, group in grouped_dict.items():
    mmw_x = group['centroid_xyz'].apply(lambda x: x[0])
    mmw_y = group['centroid_xyz'].apply(lambda y: y[1])
    mmw_xy_cov = mmw_x.cov(mmw_y)
    mmw_yx_cov = mmw_x.cov(mmw_y)

    mmw_cov = np.array([[mmw_x.var(), 0], [0, mmw_y.var()]])
    ble_xy_cov = group['x_ble_filter'].cov(group['y_ble_filter'])
    ble_yx_cov = group['y_ble_filter'].cov(group['x_ble_filter'])
    ble_cov = np.array([[group['x_ble_filter'].var(), 0], [0, group['y_ble_filter'].var()]])

    group[["sensor_fused_xyz", "sensor_fused_cov"]] = group.apply(fuse_sensor_data, mmw_cov = mmw_cov, ble_cov = ble_cov, axis=1, result_type='expand')
    df.loc[group.index, 'sensor_fused_xyz'] = group['sensor_fused_xyz']
    df.loc[group.index, 'sensor_fused_cov'] = group['sensor_fused_cov']
    # Print the results
    print(f"Distance: {key}")
    print("Fused Position:", df['sensor_fused_xyz'].values)
    print("Covariance Matrix:\n", group[['sensor_fused_cov']].values)
    print("===================================")

# Save the updated dataframe to a new CSV file.
df.to_csv("fused_dataset.csv", sep=';', index=False)
print("Fused dataset saved to 'fused_dataset.csv'")

# plotting fusion_ttf_cov_matrix by distance
import matplotlib.pyplot as plt
import numpy as np

# Group by discrete distance and timestamp and plot covariance
def plot_covariance_by_distance(df, arg1):
    plt.figure()
    plt.plot(df[arg1], df["sensor_fused_cov"].apply(lambda x: x[0,0]), marker='o', label='cov_xx', alpha=0.5)
    plt.plot(df[arg1], df["sensor_fused_cov"].apply(lambda x: x[0,1]), marker='*', label='cov_xy', alpha=0.5)
    plt.plot(df[arg1], df["sensor_fused_cov"].apply(lambda x: x[1,1]), marker='h', label='cov_yy', alpha=0.5)
    plt.title(f"Covariance over {arg1}")
    plt.xlabel(f"{arg1}")
    plt.ylabel("Covariance Component")
    plt.legend()
    plt.show()

plot_covariance_by_distance(df, 'timestamp')
plot_covariance_by_distance(df, 'distance')
