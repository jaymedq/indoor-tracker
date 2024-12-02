import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython import display 

# Load data
data = pd.read_csv("../Results/ble_mmwave_fusion_all.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y/%m/%d %H:%M:%S')

# Ensure stringified lists are parsed correctly
data['x'] = data['x'].apply(eval)
data['y'] = data['y'].apply(eval)
data['z'] = data['z'].apply(eval)

# Parse centroids and real_xyz if they are present in the dataset
if 'centroid_xyz' in data.columns:
    data['centroid_xyz'] = data['centroid_xyz'].apply(eval)
if 'real_xyz' in data.columns:
    data['real_xyz'] = data['real_xyz'].apply(eval)

# Initialize the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    
    # Filter data for the current timestamp
    timestamp_data = data[data['timestamp'] == frame]
    
    # Extract point cloud
    xs = timestamp_data['x'].values[0]
    ys = timestamp_data['y'].values[0]
    zs = timestamp_data['z'].values[0]
    
    # Extract centroid (if present)
    centroid = timestamp_data['centroid_xyz'].values[0] if 'centroid_xyz' in timestamp_data.columns else None
    
    # Extract real points (if present)
    real_points = timestamp_data['real_xyz'].values[0] if 'real_xyz' in timestamp_data.columns else None
    
    # Plot the point cloud in blue
    ax.scatter(xs, ys, zs, c='b', marker='o', label='Point Cloud')
    
    # Plot the centroid in red
    if centroid:
        ax.scatter(centroid[0], centroid[1], centroid[2], c='r', marker='^', s=50, label='Centroid')
    
    # Plot the real_xyz points in green
    if real_points:
        ax.scatter(real_points[0], real_points[1], real_points[2], c='g', marker='s', s=100, label='Real Point')
    
    # Set axis limits
    ax.set_xlim([0, 10])
    ax.set_ylim([-10, 0])
    ax.set_zlim([-1, 6])
    
    # Set labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(f"Point Cloud at Timestamp: {frame}")
    
    # Add legend
    ax.legend()
    
    return ax

# Create animation
ani = FuncAnimation(fig, update, frames=sorted(data['timestamp'].unique()), blit=False, repeat=True, interval=250, repeat_delay=3000)

writervideo = FFMpegWriter(fps=15)
ani.save('plot_dataset_animation_output.mp4', writer=writervideo) 

# Display the plot
plt.show()
