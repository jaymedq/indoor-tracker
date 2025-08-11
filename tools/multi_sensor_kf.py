import numpy as np
import pandas as pd

class MultiSensorKalman:
    def __init__(self):
        # State vector [x, y, vx, vy] so we can handle movement
        self.dt = 1.0
        self.x = np.zeros(4)

        # State transition
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4) * 0.01

        # Covariance matrix
        self.P = np.eye(4) * 1.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, H, R=None, adaptive=False):
        z = np.array(z)
        y = z - H @ self.x
        S = H @ self.P @ H.T + (R if R is not None else self.R)

        if adaptive:
            # Adapt R based on residual statistics
            alpha = 0.2
            meas_cov_est = np.outer(y, y)
            R_adapted = (1 - alpha) * (R if R is not None else self.R) + alpha * meas_cov_est
            S = H @ self.P @ H.T + R_adapted
        else:
            R_adapted = R if R is not None else self.R

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P
        return R_adapted  # return for logging


    def get_state(self):
        return self.x[:2]  # position only

kf = MultiSensorKalman()

df = pd.read_csv("FUSAO_PROCESSADA.csv", sep=";")
# Drop np.isnan values
df: pd.DataFrame = df[~np.isnan(df[["x_ble"]]).any(axis=1)]
df = df.reset_index(drop=True)
df["centroid_xyz"] = df["centroid_xyz"].apply(eval)
df["ble_xyz"] = df["ble_xyz"].apply(eval)
df["real_xyz"] = df["real_xyz"].apply(eval)
df["sensor_fused_xyz"] = [[0,0,0]] * len(df)

fused_values = []

# Radar origin
radar_placement = np.array([0.995, -7.88, 1.78])

def calculate_distance(row):
    return np.linalg.norm(np.array(row["real_xyz"]) - radar_placement)

# Append the fused data to the dataframe.
df['distance'] = df.apply(calculate_distance, axis=1)


R_mm = np.eye(2) * 0.1
R_ble = np.eye(2) * 0.05


for i, row in df.iterrows():
    mm_meas = np.array(row["centroid_xyz"][:2])
    ble_meas = np.array(row["ble_xyz"][:2])

    kf.predict()

    # Update with mmWave
    H_mm = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    R_mm = np.array([[0.1, 0],
                     [0, 0.2]])
    kf.update(mm_meas, H_mm, R_mm, adaptive=True)

    # Update with BLE
    H_ble = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
    R_ble = np.array([[0.01, 0],
                      [0, 0.5]])
    kf.update(ble_meas, H_ble, R_ble, adaptive=True)

    print("Fused position:", kf.get_state())

    fused_position = kf.get_state().tolist()
    row['sensor_fused_xyz'] = [fused_position[:2] + [1.78]][0]
    df.at[i, 'sensor_fused_xyz'] = row['sensor_fused_xyz']

# Save the updated dataframe to a new CSV file.
df.to_csv("fused_dataset.csv", sep=';', index=False)
print("Fused dataset saved to 'fused_dataset.csv'")
