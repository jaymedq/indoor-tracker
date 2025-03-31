import numpy as np
import pandas as pd
import ast

class KalmanFilter:
    def __init__(self, dt, initial_state, initial_covariance, process_noise, measurement_noise):
        """
        Initializes the Kalman filter for a constant velocity model.
        
        Parameters:
          dt: Time step
          initial_state: (6,1) vector [x, y, z, vx, vy, vz]^T
          initial_covariance: (6,6) initial state covariance matrix
          process_noise: (6,6) process noise covariance matrix
          measurement_noise: (3,3) measurement noise covariance matrix (for the position measurements)
        """
        self.dt = dt
        self.x = initial_state  # state vector: [x, y, z, vx, vy, vz]^T
        self.P = initial_covariance  # state covariance matrix
        
        # State transition matrix (constant velocity model)
        self.F = np.block([[np.eye(3), dt * np.eye(3)],
                           [np.zeros((3, 3)), np.eye(3)]])
        
        # Measurement matrix: we only measure position
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])
        
        self.Q = process_noise  # process noise covariance matrix
        self.R = measurement_noise  # measurement noise covariance matrix
    
    def predict(self):
        """Performs the prediction step of the Kalman filter."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z):
        """
        Performs the measurement update step.
        
        Parameters:
          z: Measurement vector (3-element list or array)
          
        Returns:
          position: Updated position estimate (first 3 elements of state)
          position_cov: (3,3) covariance of the position estimate
        """
        z = np.array(z).reshape(3, 1)
        y = z - self.H @ self.x  # innovation or measurement residual
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
        
        # Extract the position and its covariance (upper-left 3x3 block)
        position = self.x[:3].flatten()
        position_cov = self.P[:3, :3]
        return position, position_cov

# --- Kalman Filter Setup ---

# Define the time step (modify as needed)
dt = 1.0

# Initial state: assume starting at zero position and velocity.
initial_state = np.zeros((6, 1))

# Initial covariance: set high uncertainty initially.
initial_covariance = np.eye(6) * 10.0

# Process noise covariance (tuning parameter)
q = 0.1  # process noise level
Q = np.block([
    [np.eye(3) * q, np.zeros((3, 3))],
    [np.zeros((3, 3)), np.eye(3) * q]
])

# Measurement noise covariance for each sensor:
# mmWave is assumed to be more accurate than BLE.
R_mm = np.diag([0.1, 0.1, 0.1])  # mmWave sensor measurement noise
R_ble = np.diag([0.5, 0.5, 0.5])  # BLE sensor measurement noise

# Create Kalman filter instances for each sensor
kf_mm = KalmanFilter(dt, initial_state.copy(), initial_covariance.copy(), Q, R_mm)
kf_ble = KalmanFilter(dt, initial_state.copy(), initial_covariance.copy(), Q, R_ble)

# --- Read the Dataset ---
df = pd.read_csv("FUSAO_PROCESSADA.csv", sep=";")

fused_values = []

# Process each row (assumed to be sequential in time)
for idx, row in df.iterrows():
    try:
        # Parse the string representations of the measurements
        mm_meas = ast.literal_eval(row["centroid_xyz"])
        ble_meas = ast.literal_eval(row["ble_xyz"])
    except Exception as e:
        print(f"Error parsing row {idx}: {e}")
        mm_meas = [0.0, 0.0, 0.0]
        ble_meas = [0.0, 0.0, 0.0]
    
    # --- Kalman Filter Prediction Step ---
    kf_mm.predict()
    kf_ble.predict()
    
    # --- Kalman Filter Update Step ---
    state_mm, cov_mm = kf_mm.update(mm_meas)
    state_ble, cov_ble = kf_ble.update(ble_meas)
    
    # --- Track-to-Track Fusion ---
    # Use the updated position covariances from the Kalman filters.
    try: 
        inv_cov_mm = np.linalg.inv(cov_mm)
        inv_cov_ble = np.linalg.inv(cov_ble)
        fused_cov_inv = inv_cov_mm + inv_cov_ble
        fused_cov = np.linalg.inv(fused_cov_inv)
        fused_state = fused_cov.dot(inv_cov_mm.dot(np.array(state_mm)) + inv_cov_ble.dot(np.array(state_ble)))
        fused_state = fused_state.flatten().tolist()
    except np.linalg.LinAlgError as e:
        print(f"Error inverting covariance matrices at row {idx}: {e}")
        fused_state = [0.0, 0.0, 0.0]
    
    fused_values.append(fused_state)

# Append the fused data to the dataframe.
df["sensor_fused_xyz"] = fused_values

# Save the updated dataframe to a new CSV file.
df.to_csv("fused_dataset.csv", sep=';', index=False)
print("Fused dataset saved to 'fused_dataset.csv'")
