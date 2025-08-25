import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from constants import RADAR_PLACEMENT, EXPERIMENT_POINTS
from calculate_mse_mae_rmse import calculate_mse_mae_rmse
from matplotlib.collections import LineCollection
import matplotlib as mpl

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

def plot_colored_points(ax, x, y, times, cmap, label, size=10, marker='.'):
    """Plot points with fading colors (no connecting lines)."""
    if len(x) < 1:
        return
    if label == "Real":
        ax.scatter(x, y, marker = marker, c="black", s=size, label=label)
    else:
        norm = mpl.colors.Normalize(vmin=times.min()-1200, vmax=times.max())
        ax.scatter(x, y, marker = marker, c=times, cmap=cmap, norm=norm, s=size, alpha=0.8, label=label)


# Prepare trajectories
trajectories = {
    "centroid": {"x": [], "y": []},
    "sensor_fused": {"x": [], "y": []},
    "real": {"x": [], "y": []},
    "ble": {"x": [], "y": []},
}

def plot_radar_fov(ax):
    """Plot radar FOV sector and radar position."""
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
    ax.scatter(RADAR_PLACEMENT[0], RADAR_PLACEMENT[1], c="k", marker="x", s=80, label="Radar Position")

def plot_experiment_points(ax):
    for label, coords in EXPERIMENT_POINTS.items():
        if "V" in label:
            continue
        if "ANCHOR" in label:
            ax.text(coords[0], coords[1], label.replace("ANCHOR","A"), fontsize=8, weight='bold', ha='right', va='bottom', color='r')
            ax.scatter(coords[0], coords[1], c="r", marker="v", s=20)
        else:
            ax.text(coords[0], coords[1], label, fontsize=8, weight='bold', ha='right', va='bottom', color='dimgray')
            ax.scatter(coords[0], coords[1], c="gray", marker=".", s=20)

# Iterate through dataset to accumulate trajectories
for _, row in data.iterrows():
    if "centroid_xyz" in row and row["centroid_xyz"]:
        trajectories["centroid"]["x"].append(row["centroid_xyz"][0])
        trajectories["centroid"]["y"].append(row["centroid_xyz"][1])
    if "sensor_fused_xyz" in row and row["sensor_fused_xyz"]:
        trajectories["sensor_fused"]["x"].append(row["sensor_fused_xyz"][0])
        trajectories["sensor_fused"]["y"].append(row["sensor_fused_xyz"][1])
    if "real_xyz" in row and row["real_xyz"]:
        trajectories["real"]["x"].append(row["real_xyz"][0])
        trajectories["real"]["y"].append(row["real_xyz"][1])
    if "x_ble" in row and "y_ble" in row and not pd.isna(row["x_ble"]) and not pd.isna(row["y_ble"]):
        trajectories["ble"]["x"].append(row["x_ble"])
        trajectories["ble"]["y"].append(row["y_ble"])

# Plot static routes
fig, ax = plt.subplots(figsize=(8, 6))

plot_radar_fov(ax)

# Extract time for gradient coloring
times = data["timestamp"].astype(np.int64) // 10**9  # convert to seconds

if trajectories["sensor_fused"]["x"]:
    plot_colored_points(ax, trajectories["sensor_fused"]["x"], trajectories["sensor_fused"]["y"],
                        times[:len(trajectories["sensor_fused"]["x"])],
                        plt.cm.Greens, "Sensor Fused", size=12, marker='D')
    
if trajectories["centroid"]["x"]:
    plot_colored_points(ax, trajectories["centroid"]["x"], trajectories["centroid"]["y"],
                        times[:len(trajectories["centroid"]["x"])],
                        plt.cm.Purples, "Centroid", size=8)

if trajectories["ble"]["x"]:
    plot_colored_points(ax, trajectories["ble"]["x"], trajectories["ble"]["y"],
                        times[:len(trajectories["ble"]["x"])],
                        plt.cm.Blues, "BLE", size=8)

plot_experiment_points(ax)

if trajectories["real"]["x"]:
    plot_colored_points(ax, trajectories["real"]["x"], trajectories["real"]["y"],
                        times[:len(trajectories["real"]["x"])],
                        "black", "Real", size=25)  # maybe a bit bigger for clarity


# Axis setup
ax.set_xlim([0, 10])
ax.set_ylim([-10, 0])
legend_handles = [
    plt.Line2D([0], [0], marker='.', color='w', markerfacecolor="purple", markersize=8, label="mmWave"),
    plt.Line2D([0], [0], marker='.', color='w', markerfacecolor="blue", markersize=8, label="BLE"),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor="limegreen", markersize=8, label="Sensor Fusion"),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="black", markersize=8, label="Ground Truth"),
    plt.Line2D([0], [0], marker='<', color='w', markerfacecolor="cyan", markersize=8, label="Radar FOV"),
    plt.Line2D([0], [0], marker='X', color='w', markerfacecolor="black", markersize=8, label="Radar Position"),
    plt.Line2D([0], [0], marker='v', color='w', markerfacecolor="r", markersize=8, label="BLE Anchor Position"),
]
plt.legend(handles=legend_handles)

plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Standalone sensors and fusion color fading over time")
plt.show()

plt.show()
