import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of time steps for the synthetic dataset
num_steps = 50

# True trajectory: starting at [0, 0, 0] with constant velocity [0.5, 0.2, 0.1]
velocity = np.array([0.5, 0.2, 0.1])
true_positions = np.array([velocity * t for t in range(num_steps)])

# Define noise levels for each sensor
# mmWave sensor is assumed to be more accurate (lower standard deviation)
std_mm = 0.1
# BLE sensor is less accurate (higher standard deviation)
std_ble = 0.5

# Generate noisy measurements for each sensor
mmwave_measurements = true_positions + np.random.normal(0, std_mm, (num_steps, 3))
ble_measurements = true_positions + np.random.normal(0, std_ble, (num_steps, 3))

# Convert each row to a string representation of a list, e.g., "[x, y, z]"
mmwave_str = [str(list(row)) for row in mmwave_measurements]
ble_str = [str(list(row)) for row in ble_measurements]

# Create a dataframe
df = pd.DataFrame({
    "mmwave_xyz": mmwave_str,
    "ble_xyz": ble_str
})

# Save the synthetic dataset to a CSV file
csv_filename = "your_dataset.csv"
df.to_csv(csv_filename, index=False)
print(f"Synthetic dataset saved to '{csv_filename}'")
