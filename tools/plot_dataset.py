import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to read the dataset and plot the points in 3D
def plot_3d_points(file_path):
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Extract data
    ranges = df['range'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])
    xs = df['x'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])
    ys = df['y'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])
    zs = df['z'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])
    
    # Flatten the lists of coordinates
    all_xs = [item for sublist in xs for item in sublist]
    all_ys = [item for sublist in ys for item in sublist]
    all_zs = [item for sublist in zs for item in sublist]
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(all_xs, all_ys, all_zs)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()

# Path to your CSV file
file_path = '2d_xwr68xxconfig_output.csv'
plot_3d_points(file_path)
