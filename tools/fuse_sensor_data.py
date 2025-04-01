import numpy as np
import pandas as pd

class KalmanFilter2D:
    def __init__(self):
        # State vector [x, y, vx, vy]
        self.x = np.array([0, 0, 0, 0], dtype=float)
        
        # State transition matrix (Assuming constant velocity model)
        self.dt = 1  # Time step
        self.F = np.array([
            [1, 0, self.dt, 0],  # x' = x + vx*dt
            [0, 1, 0, self.dt],  # y' = y + vy*dt
            [0, 0, 1, 0],  # vx' = vx
            [0, 0, 0, 1]   # vy' = vy
        ], dtype=float)
        
        # Process noise covariance (assumed small)
        self.Q = np.eye(4) * 0.01
        
        # Measurement matrix (Only x and y)
        self.H = np.array([
            [1, 0, 0, 0],  # Measure x
            [0, 1, 0, 0]   # Measure y
        ], dtype=float)

        # Measurement noise covariance (Will be updated dynamically)
        self.R = np.eye(2)  

        # Initial covariance matrix (High uncertainty in velocity)
        self.P = np.eye(4) * 10

    def predict(self):
        self.x = np.dot(self.F, self.x)  # State prediction
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # Covariance prediction

    def update(self, measurement, R):
        """
        measurement: [x, y]
        R: 2x2 covariance matrix for measurement
        """
        self.R = R  # Update measurement noise
        z = np.array(measurement)

        # Innovation
        y = z - np.dot(self.H, self.x)

        # Innovation covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman Gain

        # State update
        self.x = self.x + np.dot(K, y)

        # Covariance update
        I = np.eye(4)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def get_state(self):
        return self.x[:2]  # Return only (x, y)

    def get_covariance(self):
        return self.P[:2, :2]  # Return only 2x2 covariance

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


kf_mmwave = KalmanFilter2D()
kf_ble = KalmanFilter2D()

# Example measurement noise covariance for each sensor
R_mmwave = np.array([[0.01, 0], [0, 0.01]])  # More precise
R_ble = np.array([[0.2, 0], [0, 0.2]])  # Less precise

df = pd.read_csv("FUSAO_PROCESSADA.csv", sep=";")
# Drop np.isnan values
df = df[~np.isnan(df[["X_est_TRIG"]]).any(axis=1)]
df = df.reset_index(drop=True)
df["centroid_xyz"] = df["centroid_xyz"].apply(eval)
df["ble_xyz"] = df["ble_xyz"].apply(eval)

fused_values = []

# Process each row (assumed to be sequential in time)
for idx, row in df.iterrows():
    try:
        # Parse the string representations of the measurements
        mm_meas = row["centroid_xyz"]
        ble_meas = row["ble_xyz"]
        if np.nan in mm_meas or np.nan in ble_meas:
            raise ValueError("Invalid measurements")
    except Exception as e:
        print(f"Error parsing row {idx}: {e}")
        mm_meas = [0.0, 0.0, 0.0]
        ble_meas = [0.0, 0.0, 0.0]

    # Run Kalman filters
    kf_mmwave.predict()
    kf_mmwave.update(mm_meas[:2], R_mmwave)

    kf_ble.predict()
    kf_ble.update(ble_meas[:2], R_ble)

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

    fused_values.append(fused_position)

# Append the fused data to the dataframe.
df["sensor_fused_xyz"] = fused_values

# Save the updated dataframe to a new CSV file.
df.to_csv("fused_dataset.csv", sep=';', index=False)
print("Fused dataset saved to 'fused_dataset.csv'")
