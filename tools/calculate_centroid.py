import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

input_file = "output_lab_tag.csv"
output_file = "output_transformed_centroid.csv"

# Annotated points
annotated_points = {
    "P1": [1.15, -0.4, 0], "P2": [2.35, -0.4, 0], "P3": [3.55, -0.4, 0], 
    "P4": [4.75, -0.4, 0], "P5": [5.95, -0.4, 0], "P6": [1.143, -4.462, 0],
    "P7": [2.343, -4.462, 0], "P8": [3.543, -4.462, 0], "P9": [4.745, -4.462, 0],
    "P10": [5.944, -4.462, 0], "P11": [1.102, -6.865, 0], "P12": [2.308, -6.865, 0],
    "P13": [3.503, -6.865, 0], "P14": [4.7, -6.865, 0], "P15": [5.9, -6.865, 0],
    "PA": [7.1, -6.865, 0], "PB": [7.14, -5.618, 0], "PC": [7.13, -4.462, 0],
    "PD": [7.1, -3.215, 0], "PE": [7.143, -2.015, 0], "PF": [7.144, -0.863, 0],
    "PIN": [8.61, -7.473, 0], "PS": [8.61, -1.457, 0], 
    "A1": [0.995, -7.825, 1.70], "A2": [0.99, -1.206, 2.416], 
    "A3": [5.717, -7.846, 2.41], "A4": [3.524, -4.629, 2.416]
}

annotated_points = {
    "PA": [7.1, -6.865, 1.78],
    "P11": [1.102, -6.865, 1.78],
    "P12": [2.308, -6.865, 1.78],
    "P13": [3.503, -6.865, 1.78],
    "P14": [4.7, -6.865, 1.78],
    "P15": [5.9, -6.865, 1.78],
}

# Radar origin
radar_placement = np.array([0.995, -7.825, 1.70])

df = pd.read_csv(input_file)

df['x'] = df['x'].apply(eval)
df['y'] = df['y'].apply(eval)
df['z'] = df['z'].apply(eval)

# Function to transform coordinates
def transform_coordinates(row):
    points = np.array([row['x'], row['y'], row['z']]).T
    transformed = np.array([
        radar_placement[0] + points[:, 0],  # Add radar x
        radar_placement[1] - points[:, 1],  # Subtract radar y
        radar_placement[2] + points[:, 2]   # Add radar z
    ])
    return [transformed[0].tolist(), transformed[1].tolist(), transformed[2].tolist()]

# Apply transformation
df[['x', 'y', 'z']] = df.apply(transform_coordinates, axis=1, result_type='expand')

def calculate_centroid(row):
    centroid_x = round(np.mean(row['x']), 2)
    centroid_y = round(np.mean(row['y']), 2)
    centroid_z = round(np.mean(row['z']), 2)
    return [centroid_x, centroid_y, centroid_z]

df['centroid_xyz'] = df.apply(calculate_centroid, axis=1)

df.to_csv(output_file, index = False)

print(f"Centroid calculation added to data at {output_file}")