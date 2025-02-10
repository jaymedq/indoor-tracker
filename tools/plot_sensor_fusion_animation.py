import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from calculate_mse_mae_rmse import calculate_mse_mae_rmse

# Load data using os.path to ensure the script can be run from any directory
data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..',"FUSAO_PROCESSADA.csv"), sep=';')

data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")

# Ensure stringified lists are parsed correctly
data["x"] = data["x"].apply(eval)
data["y"] = data["y"].apply(eval)
data["z"] = data["z"].apply(eval)

# Parse centroids and real_xyz if they are present in the dataset
if "centroid_xyz" in data.columns:
    data["centroid_xyz"] = data["centroid_xyz"].apply(eval)
if "real_xyz" in data.columns:
    data["real_xyz"] = data["real_xyz"].apply(eval)

# Calculate mse mae rmse
errors = calculate_mse_mae_rmse(data)

# Initialize the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=45, azim=225)

def update(frame_timestamp):
    ax.clear()
    # Filter data for the current timestamp
    timestamp_data = data[data["timestamp"] == frame_timestamp]

    # Extract point cloud
    xs = timestamp_data["x"].values[0]
    ys = timestamp_data["y"].values[0]
    zs = timestamp_data["z"].values[0]

    # Extract centroid (if present)
    centroid = (
        timestamp_data["centroid_xyz"].values[0]
        if "centroid_xyz" in timestamp_data.columns
        else None
    )

    # Extract real points (if present)
    real_points = (
        timestamp_data["real_xyz"].values[0]
        if "real_xyz" in timestamp_data.columns
        else None
    )

    # Plot the point cloud in blue
    ax.scatter(xs, ys, zs, c="b", marker="o", label="MMW Point Cloud")

    # Plot the centroid in red
    if centroid:
        ax.scatter(
            centroid[0],
            centroid[1],
            centroid[2],
            c="r",
            marker="^",
            s=50,
            label="MMW Centroid",
        )
    
    #plot BLE triangulation in yellow
    triangulation = [timestamp_data["X_est_TRIANG_KF"], timestamp_data["Y_est_TRIANG_KF"], 1.78]
    ax.scatter(
        triangulation[0],
        triangulation[1],
        triangulation[2],
        c="y",
        marker="^",
        s=50,
        label="BLE Triangulation",
    )

    #plot BLE fusion in purple
    ble_fusion = [timestamp_data["X_est_FUSAO"], timestamp_data["Y_est_FUSAO"], 1.78]
    ax.scatter(
        ble_fusion[0],
        ble_fusion[1],
        ble_fusion[2],
        c="purple",
        marker="^",
        s=50,
        label="BLE Fusion",
    )

    # # plot BLE and mmw centroid fusion by adding the two centroids and dividing by 2
    # if centroid:
    #     ble_mmw_fusion_centroid = [(centroid[0] + ble_fusion[0]) / 2, (centroid[1] + ble_fusion[1]) / 2, (centroid[2] + ble_fusion[2]) / 2]
    #     ax.scatter(
    #         ble_mmw_fusion_centroid[0],
    #         ble_mmw_fusion_centroid[1],
    #         ble_mmw_fusion_centroid[2],
    #         c="orange",
    #         marker="^",
    #         s=50,
    #         label="BLE and MMW Fusion",
    #     )

    # plot BLE and mmw centroid fusion by adding the two centroids and dividing by 2
    if centroid:
        kalmanttf = [timestamp_data['X_fused'], timestamp_data['Y_fused'], 1.78]
        ax.scatter(
            kalmanttf[0],
            kalmanttf[1],
            kalmanttf[2],
            c="orange",
            marker="^",
            s=50,
            label="BLE and MMW Fusion",
        )

    # Plot the real_xyz points in green
    error = 0
    if real_points:
        ax.scatter(
            real_points[0],
            real_points[1],
            real_points[2],
            c="g",
            marker="s",
            s=100,
            label="Real Point",
        )
        real_3d = np.array([real_points[0], real_points[1], real_points[2]])
        centroid_3d = np.array([centroid[0], centroid[1], centroid[2]])
        error = np.sqrt((real_3d - centroid_3d) ** 2).sum()

    # Set axis limits
    ax.set_xlim([-5, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-1, 6])

    # Set labels and title
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title(
        f"Timestamp: {frame_timestamp}. MMWRMSE: {errors['rmse_centroid']:.2f} MMW_Error:{error:.2f}"
    )

    # Add legend
    ax.legend()

    # # Vary the angle of the plot every two seconds in the animation
    # if (frame_timestamp.second % 2 == 0):
    #     step = 6
    #     ax.view_init(elev=45, azim=ax.axis_view_counter * step)
    #     ax.axis_view_counter+=1

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

video_writer = FFMpegWriter(fps=10)
ani.save("plot_dataset_animation_output.mp4", writer=video_writer)

# Display the plot
plt.show()
