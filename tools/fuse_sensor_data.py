import numpy as np
import pandas as pd

class KalmanFilter2D:
    def __init__(self):
        # State vector [x, y]
        self.x = np.array([0, 0], dtype=float)
        
        # State transition matrix (Assuming static model)
        self.dt = 1  # Time step
        self.F = np.eye(2)
        
        # Process noise covariance (assumed small)
        self.Q = np.eye(2) * 0.01
        
        # Measurement matrix (Only x and y)
        self.H = np.eye(2)

        # Measurement noise covariance (Will be updated dynamically)
        self.R = np.eye(2)

        # Initial covariance matrix (High uncertainty initially)
        self.P = np.eye(2) * 1.0

    def predict(self):
        self.x = np.dot(self.F, self.x)  # State prediction
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # Covariance prediction

    def update(self, measurement):
        """
        measurement: [x, y]
        R: 2x2 covariance matrix for measurement
        """
        z = np.array(measurement)

        # Innovation
        y = z - np.dot(self.H, self.x)

        # Calculate measurement noise covariance using the covariance of the measurement noise
        self.R = np.eye(2) * np.array([0.01, 0.5])  # Example covariance for the measurement

        # Innovation covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # State update
        self.x = self.x + np.dot(K, y)

        # Covariance update
        I = np.eye(2)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def get_state(self):
        return self.x

    def get_covariance(self):
        return self.P

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
df["ble_xyz"] = df["ble_xyz"].apply(eval)
df["real_xyz"] = df["real_xyz"].apply(eval)

fused_values = []

# Process each row (assumed to be sequential in time)
def fuse_sensor_data(row, kf_mmwave, kf_ble, R_mmwave, R_ble):
    # try:
    # Parse the string representations of the measurements
    mm_meas = row["centroid_xyz"]
    ble_meas = row["ble_xyz"]
    if np.nan in mm_meas or np.nan in ble_meas:
        raise ValueError("Invalid measurements")
    # except Exception as e:
    #     mm_meas = [0.0, 0.0, 0.0]
    #     ble_meas = [0.0, 0.0, 0.0]

    # Run Kalman filters
    kf_mmwave.predict()
    kf_mmwave.update(mm_meas[:2])

    kf_ble.predict()
    kf_ble.update(ble_meas[:2])

    # Get estimates and covariances
    mean_mmwave = kf_mmwave.get_state()
    cov_mmwave = kf_mmwave.get_covariance()

    mean_ble = kf_ble.get_state()
    cov_ble = kf_ble.get_covariance()

    # Fuse estimates
    fused_position, fused_covariance = track_to_track_fusion(mean_mmwave, cov_mmwave, mean_ble, cov_ble)
    fused_position = fused_position.tolist()
    fused_position.append(1.78)  # Add Z coordinate

    print("Fused Position:", fused_position)
    print("Fused Covariance:\n", fused_covariance)

    return fused_position, fused_covariance[0][0], fused_covariance[0][1], fused_covariance[1][0], fused_covariance[1][1], mean_ble[0], mean_ble[1], 1.78, mean_mmwave[0], mean_mmwave[1], 1.78

# Radar origin
radar_placement = np.array([0.995, -7.88, 1.78])

def calculate_distance(row):
    return np.linalg.norm(np.array(row["real_xyz"]) - radar_placement)

# Append the fused data to the dataframe.
df['distance'] = df.apply(calculate_distance, axis=1)

grouped_dict = {key: group for key, group in df.groupby("distance")}
for key, group in grouped_dict.items():
    kf_mmwave = KalmanFilter2D()
    kf_ble = KalmanFilter2D()
    # Example measurement noise covariance for each sensor
    R_mmwave = np.array([[0.01, 0], [0, 0.01]])  # More precise
    R_ble = np.array([[0.2, 0], [0, 0.2]])  # Less precise

    group[["sensor_fused_xyz", "cov_xx", "cov_xy", "cov_yx", "cov_yy", "x_ble_kf", "y_ble_kf", "z_ble_kf", "x_mmw_kf", "y_mmw_kf", "z_mmw_kf"]] = group.apply(fuse_sensor_data, kf_mmwave=kf_mmwave, kf_ble=kf_ble, R_mmwave=R_mmwave, R_ble=R_ble, axis=1, result_type='expand')
    df.loc[group.index, 'sensor_fused_xyz'] = group['sensor_fused_xyz']
    df.loc[group.index, 'cov_xx'] = group['cov_xx']
    df.loc[group.index, 'cov_xy'] = group['cov_xy']
    df.loc[group.index, 'cov_yx'] = group['cov_yx']
    df.loc[group.index, 'cov_yy'] = group['cov_yy']
    df.loc[group.index, 'x_ble_kf'] = group['x_ble_kf']
    df.loc[group.index, 'y_ble_kf'] = group['y_ble_kf']
    df.loc[group.index, 'z_ble_kf'] = group['z_ble_kf']
    df.loc[group.index, 'x_mmw_kf'] = group['x_mmw_kf']
    df.loc[group.index, 'y_mmw_kf'] = group['y_mmw_kf']
    df.loc[group.index, 'z_mmw_kf'] = group['z_mmw_kf']
    # Print the results
    print(f"Distance: {key}")
    print("Fused Position:", df['sensor_fused_xyz'].values)
    print("Covariance Matrix:\n", group[['cov_xx', 'cov_xy', 'cov_yx', 'cov_yy']].values)
    print("===================================")

# Save the updated dataframe to a new CSV file.
df.to_csv("fused_dataset.csv", sep=';', index=False)
print("Fused dataset saved to 'fused_dataset.csv'")

# plotting fusion_kf_cov_matrix by distance
import matplotlib.pyplot as plt
import numpy as np

# Group by discrete distance and timestamp and plot covariance
def plot_covariance_by_distance(df, arg1):
    plt.figure()
    plt.plot(df[arg1], df["cov_xx"], marker='o', label='cov_xx', alpha=0.5)
    plt.plot(df[arg1], df["cov_xy"], marker='*', label='cov_xy', alpha=0.5)
    plt.plot(df[arg1], df["cov_yx"], marker='p', label='cov_yx', alpha=0.5)
    plt.plot(df[arg1], df["cov_yy"], marker='h', label='cov_yy', alpha=0.5)
    plt.title(f"Kalman Filter Covariance over {arg1}")
    plt.xlabel(f"{arg1}")
    plt.ylabel("Covariance Component")
    plt.legend()
    plt.show()

plot_covariance_by_distance(df, 'timestamp')
plot_covariance_by_distance(df, 'distance')
