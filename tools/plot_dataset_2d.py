import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from constants import RADAR_PLACEMENT, EXPERIMENT_POINTS
from ast import literal_eval
from plot_room_2d import plot_obstacles, plot_radar_fov, plot_experiment_points

# Load data
data = pd.read_csv("fused_dataset.csv", sep=";")
data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")

def safe_eval_list(s):
    """
    Safely evaluate a string representation of a list, correctly handling 'nan'.
    """
    try:
        return eval(s, {"nan": np.nan})
    except NameError:
        return [np.nan, np.nan, np.nan]

data["centroid_xyz"] = data["centroid_xyz"].apply(eval)
data["real_xyz"] = data["real_xyz"].apply(eval)
data["sensor_fused_xyz_filter"] = data["sensor_fused_xyz_filter"].apply(safe_eval_list)

def plot_colored_points(ax, x, y, times, cmap, label, size=10, marker='.'):
    """Plot points with fading colors (no connecting lines)."""
    if len(x) < 1:
        return
    if label == "Real":
        ax.scatter(x, y, marker=marker, c="black", s=size, label=label)
    else:
        norm = mpl.colors.Normalize(vmin=times.min()-50, vmax=times.max())
        ax.scatter(x, y, marker=marker, c=times, cmap=cmap, norm=norm, s=size, alpha=0.8, label=label)


# Prepare trajectories
trajectories = {k: {"x": [], "y": []} for k in ["centroid","sensor_fused","dl_sensor_fused","real","ble"]}

# Fill trajectories
for _, row in data.iterrows():
    if row.get("centroid_xyz"):
        trajectories["centroid"]["x"].append(row["centroid_xyz"][0])
        trajectories["centroid"]["y"].append(row["centroid_xyz"][1])
    if row.get("sensor_fused_xyz_filter") and not np.isnan(row["sensor_fused_xyz_filter"]).any():
        trajectories["sensor_fused"]["x"].append(row["sensor_fused_xyz_filter"][0])
        trajectories["sensor_fused"]["y"].append(row["sensor_fused_xyz_filter"][1])
    if row.get("dl_sensor_fused_xyz"):
        trajectories["dl_sensor_fused"]["x"].append(row["dl_sensor_fused_xyz"][0])
        trajectories["dl_sensor_fused"]["y"].append(row["dl_sensor_fused_xyz"][1])
    if row.get("real_xyz"):
        trajectories["real"]["x"].append(row["real_xyz"][0])
        trajectories["real"]["y"].append(row["real_xyz"][1])
    if "x_ble" in row and "y_ble" in row and not pd.isna(row["x_ble"]) and not pd.isna(row["y_ble"]):
        trajectories["ble"]["x"].append(row["x_ble"])
        trajectories["ble"]["y"].append(row["y_ble"])

# Extract time for gradient coloring
times = data["timestamp"].astype(np.int64) // 10**9  # convert to seconds


def make_subplot(ax, title, to_plot):
    plot_radar_fov(ax)
    plot_obstacles(ax)
    plot_experiment_points(ax)

    if "centroid" in to_plot and trajectories["centroid"]["x"]:
        plot_colored_points(ax, trajectories["centroid"]["x"], trajectories["centroid"]["y"],
                            times[:len(trajectories["centroid"]["x"])],
                            plt.cm.Purples, "mmWave", size=8)

    if "sensor_fused" in to_plot and trajectories["sensor_fused"]["x"]:
        plot_colored_points(ax, trajectories["sensor_fused"]["x"], trajectories["sensor_fused"]["y"],
                            times[:len(trajectories["sensor_fused"]["x"])],
                            plt.cm.Reds, "T2TF", size=12, marker='D')

    if "dl_sensor_fused" in to_plot and trajectories["dl_sensor_fused"]["x"]:
        plot_colored_points(ax, trajectories["dl_sensor_fused"]["x"], trajectories["dl_sensor_fused"]["y"],
                            times[:len(trajectories["dl_sensor_fused"]["x"])],
                            plt.cm.Greens, "DLFusion", size=12, marker='D')

    if "ble" in to_plot and trajectories["ble"]["x"]:
        plot_colored_points(ax, trajectories["ble"]["x"], trajectories["ble"]["y"],
                            times[:len(trajectories["ble"]["x"])],
                            plt.cm.Blues, "BLE", size=8)

    if trajectories["real"]["x"]:
        plot_colored_points(ax, trajectories["real"]["x"], trajectories["real"]["y"],
                            times[:len(trajectories["real"]["x"])],
                            "black", "Real", size=25)

    ax.set_xlim([0, 10])
    ax.set_ylim([-10, 0])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    ax.legend(loc="upper right")


# Create one figure with 3 subplots side by side
fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # wide figure with 3 panels

make_subplot(axes[0], "mmWave Centroid estimate", ["centroid"])
make_subplot(axes[1], "Sensor Fusion estimate", ["sensor_fused", "dl_sensor_fused"])
make_subplot(axes[2], "BLE estimate", ["ble"])

plt.tight_layout()
plt.savefig("trajectories_comparison.png")
plt.show()
