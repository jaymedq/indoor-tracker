import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = pd.read_csv("build/output.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d/%m/%Y %H:%M:%S')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    timestamp_data = data[data['timestamp'] == frame]
    
    xs = timestamp_data['x'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])
    ys = timestamp_data['y'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])
    zs = timestamp_data['z'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])
    
    # Plot the current point cloud for this timestamp
    ax.scatter(xs.values[0], ys.values[0], zs.values[0], c='b', marker='o')
    
    # Set axis limits
    ax.set_xlim([0, 6])
    ax.set_ylim([-6, 6])
    ax.set_zlim([-1, 6])
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(f"Point Cloud at Timestamp: {frame}")
    
    return ax,

ani = FuncAnimation(fig, update, frames=sorted(data['timestamp'].unique()), blit=False, repeat=True, interval = 500, repeat_delay = 3)
plt.show()
