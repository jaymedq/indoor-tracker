import os
import pandas as pd
import numpy as np
from pathlib import Path
import ast # Import the Abstract Syntax Tree module

# --- Constants are unchanged ---
MMW_DATASETS_SUFFIX = "_mmwave_data"
TEST_NAMES = [
    "T029_MMW_A1_BLE_C3P1", "T030_MMW_A1_BLE_C3P2", "T031_MMW_A1_BLE_C3P3",
    "T032_MMW_A1_BLE_C3P4", "T033_MMW_A1_BLE_C3P5", "T034_MMW_A1_BLE_C4PA",
    "T035_MMW_A1_BLE_CVP1", "T036_MMW_A1_BLE_CVP2", "T037_MMW_A1_BLE_CVP3",
    "T038_MMW_A1_BLE_CVP4", "T038_MMW_A1_BLE_CVP5", "T040_MMW_A1_BLE_C4PV",
    "T047_MMW_A1_BLE_C2P1", "T048_MMW_A1_BLE_C2P2", "T049_MMW_A1_BLE_C2P3",
    "T050_MMW_A1_BLE_C2P4", "T051_MMW_A1_BLE_C2P5", "T052_MMW_A1_BLE_C4P4",
    "T053_MMW_A1_BLE_C4P5", "T054_MMW_A1_BLE_C4P6", "T055_MMW_A1_BLE_C3P5",
    "T056_MMW_A1_BLE_C3P4", "T057_MMW_A1_BLE_C3P3", "T058_MMW_A1_BLE_C3P2",
    "T059_MMW_A1_BLE_C4P1", "T060_MMW_A1_BLE_C1P5"
]


def spherical_to_cartesian(row):
    """
    Converts spherical coordinates (stored in x, y, z columns) to Cartesian.
    Handles list-like data within each cell.
    """
    # --- FIX 1: Convert lists to NumPy arrays for vectorized math ---
    # This enables element-wise multiplication and calculations.
    r = np.array(row['x'])
    azimuth_radians = np.array(row['y'])
    elev_radians = np.array(row['z'])

    # The calculations are now correct because they operate on NumPy arrays
    x = r * np.cos(elev_radians) * np.cos(azimuth_radians)
    y = r * np.cos(elev_radians) * np.sin(azimuth_radians)
    z = r * np.sin(elev_radians)

    # Return as lists to store back into the DataFrame cells
    return [x.tolist(), y.tolist(), z.tolist()]

def copy_wrong_cordinates_to_spherical(row):
    """
    Copies the original spherical data from x,y,z before they are overwritten.
    """
    # This function was correct, no changes needed.
    r = row['x']
    azimuth_radians = row['y']
    elev_radians = row['z']
    return [r, azimuth_radians, elev_radians]

def fix_mmwave_spherical_to_cartesian(data_dir):
    """
    Processes all specified test files to correct the coordinate systems.
    """
    data_path = Path(data_dir)
    for test_name in TEST_NAMES:
        mmwave_file = data_path / f"{test_name}" / f"{test_name}{MMW_DATASETS_SUFFIX}.csv"
        if not mmwave_file.exists():
            print(f"File {mmwave_file} does not exist. Skipping.")
            continue

        # --- FIX 3: Detect separator to avoid errors on re-running the script ---
        # The script saves with ';', so if you run it twice, it would fail.
        # This peeks at the first line to see which separator to use.
        with open(mmwave_file, 'r') as f:
            header = f.readline()
            sep = ';' if ';' in header else ','

        df = pd.read_csv(mmwave_file, sep=sep)
        
        # --- FIX 2: Use ast.literal_eval for safe parsing of string data ---
        # This avoids the security risks of using eval().
        # It safely evaluates strings containing Python literals (lists, dicts, etc.).
        for col in ['x', 'y', 'z', 'velocity']:
            # Check if the column data is string type before applying
            if isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(ast.literal_eval)

        print(f"Processing file: {mmwave_file}")

        # Backup the original spherical coordinates into new columns
        # Your original logic for this was correct.
        df[['r','azimuth_radians','elev_radians']] = df.apply(copy_wrong_cordinates_to_spherical, axis=1, result_type='expand')
        
        # Apply the corrected conversion function to overwrite x, y, z with Cartesian coordinates
        df[['x','y','z']] = df.apply(spherical_to_cartesian, axis=1, result_type='expand')

        # Save the file back. Consider using the original separator if needed.
        df.to_csv(mmwave_file, sep=',', index=False)
        print(f"Updated file saved: {mmwave_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix MMWave spherical to Cartesian coordinates in CSV files.")
    parser.add_argument("data_dir", type=str, help="Directory containing the MMWave CSV files.")
    args = parser.parse_args()

    fix_mmwave_spherical_to_cartesian(args.data_dir)
