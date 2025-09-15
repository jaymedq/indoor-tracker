import numpy as np
import pandas as pd
# I'm assuming you have a constants.py file with this, if not, define it.
# For example: RADAR_PLACEMENT = np.array([0, 0, 1.5])
from constants import RADAR_PLACEMENT

# This "jitter" is added to the variance to prevent singular matrices
REGULARIZATION_FACTOR = 1e-9

def track_to_track_fusion(mean1, cov1, mean2, cov2):
    """
    Fuse two position estimates with covariance weighting.
    """
    cov_inv1 = np.linalg.inv(cov1)
    cov_inv2 = np.linalg.inv(cov2)
    fused_cov = np.linalg.inv(cov_inv1 + cov_inv2)
    fused_mean = np.dot(fused_cov, np.dot(cov_inv1, mean1) + np.dot(cov_inv2, mean2))
    return fused_mean, fused_cov



def safe_eval_list(s):
    """
    Safely evaluate a string representation of a list, correctly handling 'nan'.
    """
    try:
        # Provide a scope to eval where 'nan' is defined as numpy.nan
        return eval(s, {"nan": np.nan})
    except NameError:
        # If eval fails for any reason, return a list of NaNs
        return [np.nan, np.nan, np.nan]

df = pd.read_csv("FUSAO_PROCESSADA.csv", sep=";")
# Drop np.isnan values - Making this check more comprehensive
# df: pd.DataFrame = df.dropna(subset=['x_ble', 'y_ble']).reset_index(drop=True)
df["centroid_xyz"] = df["centroid_xyz"].apply(eval)
df["ble_xyz"] = df["ble_xyz"].apply(eval)
df["ble_xyz_filter"] = df["ble_xyz_filter"].apply(safe_eval_list)
df["real_xyz"] = df["real_xyz"].apply(eval)

# <<< MODIFIED AND FINAL VERSION OF THE FUSION FUNCTION >>>
def fuse_sensor_data(row, mmw_cov, ble_cov, mmwave_column="centroid_xyz", ble_column="ble_xyz_replace_filter"):
    """
    Robustly fuses sensor data, handling NaN inputs gracefully.
    """
    mm_meas = row[mmwave_column]
    ble_meas = row[ble_column]
    
    mean_mmwave = np.array(mm_meas[:2])
    mean_ble = np.array(ble_meas[:2])

    # Check validity of each sensor's 2D position data
    is_mmw_valid = not np.isnan(mean_mmwave).any()
    is_ble_valid = not np.isnan(mean_ble).any()

    if is_mmw_valid and is_ble_valid:
        # **Case 1: Both sensors are valid.** Attempt fusion.
        try:
            fused_position, fused_covariance = track_to_track_fusion(mean_mmwave, mmw_cov, mean_ble, ble_cov)
        except np.linalg.LinAlgError:
            # Fallback if fusion math fails: use the more certain sensor
            if np.trace(mmw_cov) < np.trace(ble_cov):
                fused_position, fused_covariance = mean_mmwave, mmw_cov
            else:
                fused_position, fused_covariance = mean_ble, ble_cov
    else:
        # **Case 4: Neither is valid.** Return a default value with high uncertainty.
        fused_position = np.array([np.nan, np.nan])
        fused_covariance = np.diag([1e6, 1e6]) # Large covariance = very uncertain

    fused_position = fused_position.tolist()
    if np.isnan(fused_position).all():
        fused_position.append(np.nan)  # Add Z coordinate
    else:
        fused_position.append(1.78)  # Add Z coordinate

    return fused_position, fused_covariance

def calculate_distance(row):
    return np.linalg.norm(np.array(row["real_xyz"]) - RADAR_PLACEMENT)

df['distance'] = df.apply(calculate_distance, axis=1)

# Default covariance for single-point groups or groups with all-NaN columns
DEFAULT_COV = np.diag([0.0, 0.0])

grouped_dict = {key: group for key, group in df.groupby("distance")}
for key, group in grouped_dict.items():
    if len(group) <= 1:
        mmw_cov = ble_cov = ble_cov_filter = DEFAULT_COV
    else:
        # Helper to safely calculate variance, falling back to default if result is NaN
        def safe_var(series, default_variance):
            v = series.dropna().var()
            return default_variance if np.isnan(v) else v

        mmw_x_var = safe_var(group['centroid_xyz'].apply(lambda x: x[0]), DEFAULT_COV[0, 0]) + REGULARIZATION_FACTOR
        mmw_y_var = safe_var(group['centroid_xyz'].apply(lambda y: y[1]), DEFAULT_COV[1, 1]) + REGULARIZATION_FACTOR
        mmw_cov = np.array([[mmw_x_var, 0], [0, mmw_y_var]])
        
        ble_x_var = safe_var(group['x_ble'], DEFAULT_COV[0, 0]) + REGULARIZATION_FACTOR
        ble_y_var = safe_var(group['y_ble'], DEFAULT_COV[1, 1]) + REGULARIZATION_FACTOR
        ble_cov = np.array([[ble_x_var, 0], [0, ble_y_var]])
        
        ble_x_filter_var = safe_var(group['x_ble_filter'], DEFAULT_COV[0, 0]) + REGULARIZATION_FACTOR
        ble_y_filter_var = safe_var(group['y_ble_filter'], DEFAULT_COV[1, 1]) + REGULARIZATION_FACTOR
        ble_cov_filter = np.array([[ble_x_filter_var, 0], [0, ble_y_filter_var]])

    # Apply the robust fusion function
    group[["sensor_fused_xyz", "sensor_fused_cov"]] = group.apply(
        fuse_sensor_data, mmw_cov=mmw_cov, ble_cov=ble_cov, mmwave_column="centroid_xyz", ble_column="ble_xyz", axis=1, result_type='expand'
    )
    group[["sensor_fused_xyz_filter", "sensor_fused_cov_filter"]] = group.apply(
        fuse_sensor_data, mmw_cov=mmw_cov, ble_cov=ble_cov_filter, mmwave_column="centroid_xyz", ble_column="ble_xyz_filter", axis=1, result_type='expand'
    )
    
    # Update main DataFrame
    df.loc[group.index, ['sensor_fused_xyz', 'sensor_fused_cov', 'sensor_fused_xyz_filter', 'sensor_fused_cov_filter']] = \
        group[['sensor_fused_xyz', 'sensor_fused_cov', 'sensor_fused_xyz_filter', 'sensor_fused_cov_filter']]
    
    print(f"Distance: {key}")
    print("Fused Position (first 5):", df.loc[group.index, 'sensor_fused_xyz'].head().values)
    print("Covariance Matrix (first row):\n", group['sensor_fused_cov'].iloc[0])
    print("===================================")

# Save the updated dataframe to a new CSV file.
df.to_csv("fused_dataset.csv", sep=';', index=False)
print("Fused dataset saved to 'fused_dataset.csv'")