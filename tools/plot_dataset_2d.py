import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def get_size(width_pt, fraction=1, subplots=(1, 1), aspect_ratio=None):
    """Set figure dimensions to avoid scaling in LaTeX."""
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    fig_width_in = fig_width_pt * inches_per_pt
    if aspect_ratio is None:
        golden_ratio = (5**.5 - 1) / 2
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    else:
        fig_height_in = fig_width_in * aspect_ratio
    return (fig_width_in, fig_height_in)

from constants import RADAR_PLACEMENT, EXPERIMENT_POINTS
from ast import literal_eval
from plot_room_2d import plot_obstacles, plot_radar_fov, plot_experiment_points

# Load data
data = pd.read_csv("fused_dataset.csv", sep=";")
try:
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")
except:
    # 13/10/2025 16:41:41
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%d/%m/%Y %H:%M:%S")

def safe_eval_list(s):
    """
    Safely evaluate a string representation of a list, correctly handling 'nan'.
    """
    try:
        return eval(s, {"nan": np.nan})
    except NameError:
        return [np.nan, np.nan, np.nan]

def calculate_centroid(row):
    centroid_x = round(np.mean(row['x']), 2)
    centroid_y = round(np.mean(row['y']), 2)
    centroid_z = round(np.mean(row['z']), 2)
    return [centroid_x, centroid_y, centroid_z]

def transform_coordinates(row):
    transformed = np.array([
        RADAR_PLACEMENT[0] + row['x'],  # Add radar x
        RADAR_PLACEMENT[1] - row['y'],  # Subtract radar y
        RADAR_PLACEMENT[2] + row['z']   # Add radar z
    ])
    return [transformed[0].tolist(), transformed[1].tolist(), transformed[2].tolist()]


def process_centroids(df):
    df['x'] = df['x'].apply(eval)
    df['y'] = df['y'].apply(eval)
    df['z'] = df['z'].apply(eval)
    df['velocity'] = df['velocity'].apply(eval)

    df[['x', 'y', 'z']] = df.apply(transform_coordinates, axis=1, result_type='expand')
    #drop rows where x,y,z are empty lists
    df = df[df['x'].apply(lambda x: len(x) > 0) & df['y'].apply(lambda y: len(y) > 0) & df['z'].apply(lambda z: len(z) > 0)]
    df['centroid_xyz'] = df.apply(calculate_centroid, axis=1)
    return df

if "centroid_xyz" in data.columns:
    data["centroid_xyz"] = data["centroid_xyz"].apply(eval)
    data["real_xyz"] = data["real_xyz"].apply(eval)
    data["sensor_fused_xyz_filter"] = data["sensor_fused_xyz_filter"].apply(safe_eval_list)
    if "dl_sensor_fused_xyz" in data.columns:
        data["dl_sensor_fused_xyz"] = data["dl_sensor_fused_xyz"].apply(safe_eval_list)
else:
    data = process_centroids(data)

def plot_colored_points(ax, x, y, times, cmap, label, size=10, marker='.'):
    """Plot points with fading colors (no connecting lines)."""
    if len(x) < 1:
        return
    if label == "Real":
        ax.scatter(x, y, marker=marker, c="black", s=size, label=label, rasterized=True)
    else:
        norm = mpl.colors.Normalize(vmin=times.min(), vmax=times.max())
        ax.scatter(x, y, marker=marker, c=times, cmap=cmap, norm=norm, s=size, alpha=0.8, label=label, rasterized=True)

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
times = np.arange(len(data))


def make_subplot(ax, title, to_plot):
    plot_radar_fov(ax)
    plot_obstacles(ax)
    plot_experiment_points(ax)

    if "centroid" in to_plot and trajectories["centroid"]["x"]:
        plot_colored_points(ax, trajectories["centroid"]["x"], trajectories["centroid"]["y"],
                            times[:len(trajectories["centroid"]["x"])],
                            plt.cm.Purples, "mmWave", size=25)

    if "sensor_fused" in to_plot and trajectories["sensor_fused"]["x"]:
        plot_colored_points(ax, trajectories["sensor_fused"]["x"], trajectories["sensor_fused"]["y"],
                            times[:len(trajectories["sensor_fused"]["x"])],
                            plt.cm.Reds, "T2TF", size=35, marker='D')

    if "dl_sensor_fused" in to_plot and trajectories["dl_sensor_fused"]["x"]:
        plot_colored_points(ax, trajectories["dl_sensor_fused"]["x"], trajectories["dl_sensor_fused"]["y"],
                            times[:len(trajectories["dl_sensor_fused"]["x"])],
                            plt.cm.Greens, "DLFusion", size=35, marker='D')

    if "ble" in to_plot and trajectories["ble"]["x"]:
        plot_colored_points(ax, trajectories["ble"]["x"], trajectories["ble"]["y"],
                            times[:len(trajectories["ble"]["x"])],
                            plt.cm.Blues, "BLE", size=25)

    if trajectories["real"]["x"]:
        plot_colored_points(ax, trajectories["real"]["x"], trajectories["real"]["y"],
                            times[:len(trajectories["real"]["x"])],
                            "black", "Real", size=60)

    ax.set_xlim([0, 10])
    ax.set_ylim([-10, 0])
    ax.set_aspect('equal') # Forces the 10x10 limit to render as a perfect square
    ax.set_xlabel(r"X [m]", fontsize=12)
    ax.set_ylabel(r"Y [m]", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)


if "dl_sensor_fused_xyz" in data.columns:
        
    # Create one figure with 4 subplots side by side
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

    make_subplot(axes[0], "mmWave Centroid", ["centroid"])
    make_subplot(axes[1], "Sensor Fusion", ["sensor_fused"])
    make_subplot(axes[2], "BLE estimate", ["ble"])
    make_subplot(axes[3], "Deep Learning", ["dl_sensor_fused"])

else:
    # Create one figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    make_subplot(axes[0], "BLE estimate", ["ble"])
    make_subplot(axes[1], "mmWave Centroid", ["centroid"])
    make_subplot(axes[2], "Sensor Fusion", ["sensor_fused"])

handles, labels = [], []
for ax in axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)

from matplotlib.lines import Line2D
override_dict = {
    "mmWave": Line2D([0], [0], marker='.', color='w', markerfacecolor='purple', markersize=15),
    "T2TF": Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=10),
    "DLFusion": Line2D([0], [0], marker='D', color='w', markerfacecolor='green', markersize=15),
    "BLE": Line2D([0], [0], marker='.', color='w', markerfacecolor='dodgerblue', markersize=15),
    "Real": Line2D([0], [0], marker='.', color='w', markerfacecolor='black', markersize=15)
}

for i, l in enumerate(labels):
    if l in override_dict:
        handles[i] = override_dict[l]

fig.subplots_adjust(top=0.85, wspace=0.1)
fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0.86), fontsize=18)

fig.savefig("trajectories_comparison.png")
# --- PGF CONFIGURATION ---
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",     # Matches LaTeX default
    "text.usetex": True,        # Let LaTeX handle the rendering
    "pgf.rcfonts": False,       # Ignore Matplotlib's internal fonts
})
fig.savefig("trajectories_comparison.pgf", backend='pgf', bbox_inches='tight', dpi=400)
fig.savefig("trajectories_comparison.pdf", bbox_inches='tight', dpi=400)
print("Saved trajectories_comparison.pgf and .pdf")