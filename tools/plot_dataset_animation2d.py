import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Wedge

from calculate_mse_mae_rmse import calculate_mse_mae_rmse

# Radar parameters
# radar_placement = np.array([0.98, -4.5])  # only X, Y for 2D
radar_placement = np.array([0.995, -7.88])
radar_facing_angle = 0  # facing right along +X axis (0 degrees)
radar_fov = 120  # total degrees
fov_radius = 10  # how far to show FOV (adjust as needed)

# Load data
data = pd.read_csv("fused_dataset.csv", sep=";")
data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")

# Ensure stringified lists are parsed correctly
data["x"] = data["x"].apply(eval)
data["y"] = data["y"].apply(eval)
data["z"] = data["z"].apply(eval)

if "centroid_xyz" in data.columns:
    data["centroid_xyz"] = data["centroid_xyz"].apply(eval)
if "real_xyz" in data.columns:
    data["real_xyz"] = data["real_xyz"].apply(eval)
if "sensor_fused_xyz" in data.columns:
    data["sensor_fused_xyz"] = data["sensor_fused_xyz"].apply(eval)

errors = calculate_mse_mae_rmse(data)

fig, ax = plt.subplots()

trajectories = {
    "centroid": {"x": [], "y": []},
    "sensor_fused": {"x": [], "y": []},
    "real": {"x": [], "y": []},
    "ble": {"x": [], "y": []},
}

def plot_radar_fov(ax):
    """Plot radar FOV sector."""
    # Define start and end angles
    theta1 = radar_facing_angle - radar_fov / 2
    theta2 = radar_facing_angle + radar_fov / 2

    wedge = Wedge(
        center=(radar_placement[0], radar_placement[1]),
        r=fov_radius,
        theta1=theta1,
        theta2=theta2,
        facecolor="cyan",
        alpha=0.1,
        edgecolor="cyan",
        linestyle="--"
    )
    ax.add_patch(wedge)

    # Also plot radar position
    ax.scatter(radar_placement[0], radar_placement[1], c="k", marker="x", s=80, label="Radar")

def update(frame):
    ax.clear()

    # Plot radar FOV
    plot_radar_fov(ax)

    # Filter data for the current timestamp
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

    # Plot point cloud
    ax.scatter(xs, ys, c="b", marker="o", label="Point Cloud", alpha=0.7)

    # Plot centroid
    if centroid:
        ax.scatter(centroid[0], centroid[1], c="r", marker="^", s=50, label="Centroid")
        trajectories["centroid"]["x"].append(centroid[0])
        trajectories["centroid"]["y"].append(centroid[1])
        # ax.plot(trajectories["centroid"]["x"], trajectories["centroid"]["y"], "r--", alpha=0.4)

    # Sensor fused
    if sensor_fused:
        ax.scatter(sensor_fused[0], sensor_fused[1], c="y", marker="^", s=50, label="Sensor Fused")
        trajectories["sensor_fused"]["x"].append(sensor_fused[0])
        trajectories["sensor_fused"]["y"].append(sensor_fused[1])
        ax.plot(trajectories["sensor_fused"]["x"], trajectories["sensor_fused"]["y"], "y--", alpha=0.4)

    # Real point
    error = 0
    if real_points:
        ax.scatter(real_points[0], real_points[1], c="g", marker="s", s=100, label="Real Point")
        trajectories["real"]["x"].append(real_points[0])
        trajectories["real"]["y"].append(real_points[1])
        ax.plot(trajectories["real"]["x"], trajectories["real"]["y"], "g--", alpha=0.4)
        if centroid:
            real_2d = np.array([real_points[0], real_points[1]])
            centroid_2d = np.array([centroid[0], centroid[1]])
            error = np.sqrt(((real_2d - centroid_2d) ** 2).sum())

    # BLE point
    if x_ble is not None and y_ble is not None:
        ax.scatter(x_ble, y_ble, c="m", marker="D", s=60, label="BLE Position")
        trajectories["ble"]["x"].append(x_ble)
        trajectories["ble"]["y"].append(y_ble)
        # ax.plot(trajectories["ble"]["x"], trajectories["ble"]["y"], "m--", alpha=0.4)

    # Set limits
    ax.set_xlim([0, 10])
    ax.set_ylim([-10, 0])

    # Labels and title
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_title(
        f"Timestamp: {frame}. MSE: {errors['mse_centroid']:.2f} "
        f"RMSE: {errors['rmse_centroid']:.2f} Error:{error:.2f}"
    )

    ax.legend(loc="upper right")
    return ax

# Create animation
ani = FuncAnimation(
    fig,
    update,
    frames=sorted(data["timestamp"].unique()),
    blit=False,
    repeat=True,
    interval=250,
    repeat_delay=3000,
)

video_writer = FFMpegWriter(fps=15)
ani.save("plot_dataset_animation_output_with_fov.mp4", writer=video_writer)

plt.show()
