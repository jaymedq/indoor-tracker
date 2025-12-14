import os
import re
import subprocess
import sys

FOLDERS = [
    "Results/T121_MMW_A5_BLE_C2P2",
    "Results/T122_MMW_A5_BLE_C2P2",
    "Results/T123_MMW_A5_BLE_C2P2",
    "Results/T125_MMW_A5_BLE_C2P2",
    "Results/T126_MMW_A5_BLE_C2P3",
    "Results/T127_MMW_A5_BLE_C2P4",
    "Results/T128_MMW_A5_BLE_C2P5",
    "Results/T129_MMW_A5_BLE_C4P4",
    "Results/T130_MMW_A5_BLE_C3P5",
    "Results/T131_MMW_A5_BLE_C3P4",
    "Results/T132_MMW_A5_BLE_C3P3",
]

OLDER_BLE_FORMAT = r"exported_(T[0-9]{3})_MMW_(A[0-9])_BLE_(C[0-9]P[0-9]).txt"
NEW_BLE_FORMAT = r"\1_MMW_\2_BLE_\3_ble_data.csv"

# Check if the script is running inside a Git repository
try:
    # Check the current directory
    subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], check=True, capture_output=True)
except subprocess.CalledProcessError:
    print("FATAL ERROR: Not inside a Git repository.")
    print("Please run this script from the root of your Git repository.")
    sys.exit(1)

for folder in FOLDERS:
    if not os.path.isdir(folder):
        continue

    regex = re.compile(OLDER_BLE_FORMAT)
    
    for file_name in os.listdir(folder):
        match_object = regex.search(file_name)
        
        if match_object:
            old_file_path = os.path.join(folder, file_name)
            
            # Skip if the file is not tracked by Git (git mv only works on tracked files)
            try:
                subprocess.run(['git', 'ls-files', '--error-unmatch', old_file_path], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                # print(f"Skipping untracked file: {old_file_path}") # Uncomment for debugging
                continue

            new_file_name = re.sub(OLDER_BLE_FORMAT, NEW_BLE_FORMAT, file_name)
            new_file_path = os.path.join(folder, new_file_name)
            
            # Execute the 'git mv' command
            try:
                # The subprocess.run command replaces shutil.copyfile
                # It uses 'git mv <source> <destination>'
                result = subprocess.run(
                    ['git', 'mv', old_file_path, new_file_path], 
                    check=True,  # Raises an exception for non-zero return code
                    text=True,
                    capture_output=True
                )
                print(f"Renamed and staged: {old_file_path} -> {new_file_path}")
            except subprocess.CalledProcessError as e:
                # Handle cases where git mv fails (e.g., destination already exists)
                print(f"ERROR: git mv failed for {old_file_path}. Output: {e.stderr.strip()}")
            except Exception as e:
                print(f"ERROR: An unexpected error occurred for {old_file_path}: {e}")