import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KalmanFilter2D:
    def __init__(self, noise_covariance=None):
        self.x = np.array([0, 0], dtype=float)  # [x, y]

        # Static model (identity transition)
        self.F = np.eye(2)
        self.Q = np.eye(2) * 0.01
        self.H = np.eye(2)

        # Measurement noise covariance
        self.R = noise_covariance if noise_covariance is not None else np.eye(2) * 0.1

        # Initial state covariance
        self.P = np.eye(2) * 1.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        z = np.array(measurement)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        return self.x

    def get_covariance(self):
        return self.P


def track_to_track_fusion(mean1, cov1, mean2, cov2):
    """ Fuse two 2D estimates with covariance weighting """
    cov_inv1 = np.linalg.inv(cov1)
    cov_inv2 = np.linalg.inv(cov2)

    fused_cov = np.linalg.inv(cov_inv1 + cov_inv2)
    fused_mean = fused_cov @ (cov_inv1 @ mean1 + cov_inv2 @ mean2)

    return fused_mean, fused_cov


# === Load dataset ===
df = pd.read_csv("FUSAO_PROCESSADA.csv", sep=";")
df = df[~np.isnan(df[["x_ble"]]).any(axis=1)].reset_index(drop=True)

df["centroid_xyz"] = df["centroid_xyz"].apply(eval)
df["ble_xyz_filter"] = df["ble_xyz_filter"].apply(eval)
df["real_xyz"] = df["real_xyz"].apply(eval)
df["sensor_fused_xyz"] = [[0, 0, 0]] * len(df)

# Radar placement
radar_placement = np.array([0.995, -7.825, 1.70])

def fuse_sensor_data(row, kf_mmwave, kf_ble, mmw_cov, ble_cov):
    mm_meas = row["centroid_xyz"][:2]
    ble_meas = row["ble_xyz_filter"][:2]

    kf_mmwave.predict()
    kf_mmwave.update(mm_meas)

    kf_ble.predict()
    kf_ble.update(ble_meas)

    mean_mmwave = kf_mmwave.get_state()
    mean_ble = kf_ble.get_state()

    fused_position, fused_covariance = track_to_track_fusion(mean_mmwave, mmw_cov,
                                                             mean_ble, ble_cov)
    fused_position = fused_position.tolist()
    fused_position.append(1.78)  # fixed Z

    return (fused_position,
            fused_covariance[0][0], fused_covariance[0][1],
            fused_covariance[1][0], fused_covariance[1][1],
            mean_ble[0], mean_ble[1], 1.78,
            mean_mmwave[0], mean_mmwave[1], 1.78)


# === Fusion Loop ===
fused_records = []
for i, row in df.iterrows():
    # mmWave covariance: trust X (small), weak Y
    mmw_cov = np.array([[0.02, 0], [0, 10.0]])
    # BLE covariance: trust Y (small), weak X
    ble_cov = np.array([[10.0, 0], [0, 0.02]])

    kf_mmwave = KalmanFilter2D(noise_covariance=mmw_cov)
    kf_ble = KalmanFilter2D(noise_covariance=ble_cov)

    fused_results = fuse_sensor_data(row, kf_mmwave, kf_ble, mmw_cov, ble_cov)
    fused_records.append(fused_results)

df[["sensor_fused_xyz", "cov_xx", "cov_xy", "cov_yx", "cov_yy",
    "x_ble_kf", "y_ble_kf", "z_ble_kf",
    "x_mmw_kf", "y_mmw_kf", "z_mmw_kf"]] = pd.DataFrame(fused_records, index=df.index)

# Save
df.to_csv("fused_dataset.csv", sep=";", index=False)
print("Fused dataset saved to 'fused_dataset_t2t.csv'")


# === Plotting ===
def plot_covariance_by_distance(df, arg1):
    plt.figure()
    plt.plot(df[arg1], df["cov_xx"], marker='o', label='cov_xx', alpha=0.5)
    plt.plot(df[arg1], df["cov_xy"], marker='*', label='cov_xy', alpha=0.5)
    plt.plot(df[arg1], df["cov_yx"], marker='p', label='cov_yx', alpha=0.5)
    plt.plot(df[arg1], df["cov_yy"], marker='h', label='cov_yy', alpha=0.5)
    plt.title(f"Track-to-Track Fusion Covariance over {arg1}")
    plt.xlabel(f"{arg1}")
    plt.ylabel("Covariance Component")
    plt.legend()
    plt.show()

if "timestamp" in df.columns:
    plot_covariance_by_distance(df, 'timestamp')
plot_covariance_by_distance(df, 'distance')
