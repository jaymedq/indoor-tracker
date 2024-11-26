import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# Annotated points
annotated_points = {
    "PA": [7.1, -6.865, 0],
    "P11": [1.102, -6.865, 0],
    "P12": [2.308, -6.865, 0],
    "P13": [3.503, -6.865, 0],
    "P14": [4.7, -6.865, 0], 
    "P15": [5.9, -6.865, 0],
}
# annotated_points = {
#     "P1": [1.15, -0.4, 0], "P2": [2.35, -0.4, 0], "P3": [3.55, -0.4, 0], 
#     "P4": [4.75, -0.4, 0], "P5": [5.95, -0.4, 0], "P6": [1.143, -4.462, 0],
#     "P7": [2.343, -4.462, 0], "P8": [3.543, -4.462, 0], "P9": [4.745, -4.462, 0],
#     "P10": [5.944, -4.462, 0], "P11": [1.102, -6.865, 0], "P12": [2.308, -6.865, 0],
#     "P13": [3.503, -6.865, 0], "P14": [4.7, -6.865, 0], "P15": [5.9, -6.865, 0],
#     "PA": [7.1, -6.865, 0], "PB": [7.14, -5.618, 0], "PC": [7.13, -4.462, 0],
#     "PD": [7.1, -3.215, 0], "PE": [7.143, -2.015, 0], "PF": [7.144, -0.863, 0],
#     "PIN": [8.61, -7.473, 0], "PS": [8.61, -1.457, 0], 
#     "A1": [0.995, -7.825, 1.70], "A2": [0.99, -1.206, 2.416], 
#     "A3": [5.717, -7.846, 2.41], "A4": [3.524, -4.629, 2.416]
# }

# Radar origin
radar_placement = np.array([0.995, -7.825, 1.70])

# Load dataset
data = pd.read_csv("tools/output_lab_tag_centroid.csv")
data['x'] = data['x'].apply(eval)
data['y'] = data['y'].apply(eval)
data['z'] = data['z'].apply(eval)

if 'real_xyz' in data.columns:
    data['real_xyz'] = data['real_xyz'].apply(eval)

# Function to transform coordinates
def transform_coordinates(row):
    points = np.array([row['x'], row['y'], row['z']]).T
    transformed = points + radar_placement
    return [transformed[:, 0].tolist(), transformed[:, 1].tolist(), transformed[:, 2].tolist()]

# Apply transformation
data[['x', 'y', 'z']] = data.apply(transform_coordinates, axis=1, result_type='expand')

# Function to map "real_xyz" to closest annotated point
def map_to_closest_real(real_points):
    real_points = np.atleast_2d(real_points)  # Ensure 2D
    annotated_array = np.array(list(annotated_points.values()))
    closest_indices = cdist(real_points, annotated_array).argmin(axis=1)
    mapped_points = list(annotated_points.values())[closest_indices[0]]
    return mapped_points

# Apply mapping for "real_xyz"
if 'real_xyz' in data.columns:
    data['real_xyz'] = data['real_xyz'].apply(map_to_closest_real)

# Save transformed dataset
data.to_csv("output_transformed.csv", index=False)
