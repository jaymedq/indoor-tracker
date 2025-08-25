import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge
import time

from tools.constants import RADAR_PLACEMENT

# Radar parameters
radar_facing_angle = 0  # facing right along +X axis (0 degrees)
radar_fov = 120  # degrees
fov_radius = 10

csv_file = "Results\\T032_MMW_A1_BLE_C3P4\\T032_MMW_A1_BLE_C3P4_mmwave_data.csv"

# Keep global dataframe that grows as new rows arrive
data = pd.DataFrame()
last_row = 0   # track how many rows we have processed

trajectories = {
    "centroid": {"x": [], "y": []},
    "sensor_fused": {"x": [], "y": []},
    "real": {"x": [], "y": []},
    "ble": {"x": [], "y": []},
}

fig, ax = plt.subplots()

def transform_coordinates(row):
    points = np.array([row['x'], row['y'], row['z']]).T
    transformed = np.array([
        RADAR_PLACEMENT[0] + points[:, 0],  # Add radar x
        RADAR_PLACEMENT[1] - points[:, 1],  # Subtract radar y
        RADAR_PLACEMENT[2] + points[:, 2]   # Add radar z
    ])
    return [transformed[0].tolist(), transformed[1].tolist(), transformed[2].tolist()]

def plot_radar_fov(ax):
    theta1 = radar_facing_angle - radar_fov / 2
    theta2 = radar_facing_angle + radar_fov / 2
    wedge = Wedge(
        center=(RADAR_PLACEMENT[0], RADAR_PLACEMENT[1]),
        r=fov_radius,
        theta1=theta1,
        theta2=theta2,
        facecolor="cyan",
        alpha=0.1,
        edgecolor="cyan",
        linestyle="--"
    )
    ax.add_patch(wedge)
    ax.scatter(RADAR_PLACEMENT[0], RADAR_PLACEMENT[1], c="k", marker="x", s=80, label="Radar")

def get_new_data():
    """Generator that yields one timestamp per new row in the CSV file."""
    global data, last_row

    while True:
        try:
            # Read all rows in file
            new_data = pd.read_csv(csv_file, sep=",")

            # Process only the rows we haven't yet
            if len(new_data) > last_row:
                for i in range(last_row, len(new_data)):
                    row = new_data.iloc[[i]].copy()

                    # Parse timestamp
                    row["timestamp"] = pd.to_datetime(
                        row["timestamp"], format="%d/%m/%Y %H:%M:%S"
                    )

                    # Parse lists
                    for col in ["x", "y", "z", "centroid_xyz", "real_xyz", "sensor_fused_xyz"]:
                        if col in row.columns:
                            row[col] = row[col].apply(eval)
                    
                    row[['x', 'y', 'z']] = row.apply(transform_coordinates, axis=1, result_type='expand')

                    # Append into global df
                    data = pd.concat([data, row], ignore_index=True)

                    # Yield this row’s timestamp
                    ts = row["timestamp"].iloc[0]
                    yield ts

                last_row = len(new_data)  # update after processing batch
            else:
                time.sleep(0.2)
        except Exception as e:
            print("Error reading CSV:", e)
            time.sleep(0.5)

def update(frame):
    ax.clear()
    plot_radar_fov(ax)

    timestamp_data = data[data["timestamp"] == frame]

    xs = timestamp_data["x"].values[0]
    ys = timestamp_data["y"].values[0]

    centroid = timestamp_data["centroid_xyz"].values[0] if "centroid_xyz" in timestamp_data.columns else None
    real_points = timestamp_data["real_xyz"].values[0] if "real_xyz" in timestamp_data.columns else None
    sensor_fused = timestamp_data["sensor_fused_xyz"].values[0] if "sensor_fused_xyz" in timestamp_data.columns else None

    x_ble, y_ble = None, None
    if "x_ble" in timestamp_data.columns and "y_ble" in timestamp_data.columns:
        x_ble = timestamp_data["x_ble"].values[0]
        y_ble = timestamp_data["y_ble"].values[0]

    ax.scatter(xs, ys, c="b", marker="o", label="Point Cloud", alpha=0.7)

    if centroid:
        ax.scatter(centroid[0], centroid[1], c="r", marker="^", s=50, label="Centroid")
        trajectories["centroid"]["x"].append(centroid[0])
        trajectories["centroid"]["y"].append(centroid[1])

    if sensor_fused:
        ax.scatter(sensor_fused[0], sensor_fused[1], c="y", marker="^", s=50, label="Sensor Fused")
        trajectories["sensor_fused"]["x"].append(sensor_fused[0])
        trajectories["sensor_fused"]["y"].append(sensor_fused[1])
        ax.plot(trajectories["sensor_fused"]["x"], trajectories["sensor_fused"]["y"], "y--", alpha=0.4)

    if real_points:
        ax.scatter(real_points[0], real_points[1], c="g", marker="s", s=100, label="Real Point")
        trajectories["real"]["x"].append(real_points[0])
        trajectories["real"]["y"].append(real_points[1])
        ax.plot(trajectories["real"]["x"], trajectories["real"]["y"], "g--", alpha=0.4)

    if x_ble is not None and y_ble is not None:
        ax.scatter(x_ble, y_ble, c="m", marker="D", s=60, label="BLE Position")
        trajectories["ble"]["x"].append(x_ble)
        trajectories["ble"]["y"].append(y_ble)

    ax.set_xlim([0, 10])
    ax.set_ylim([-10, 0])
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_title(f"Timestamp: {frame}")

    ax.legend(loc="upper right")
    return ax

ani = FuncAnimation(fig, update, frames=get_new_data, blit=False, interval=100)
plt.show()
