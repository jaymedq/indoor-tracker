from pre_process_dataset import process_centroids, TEST_NAMES, MMW_DATASETS_SUFFIX
import pandas as pd

for test_name in TEST_NAMES:
    mmwave_file = f"{test_name}{MMW_DATASETS_SUFFIX}"
    mmwave_data = pd.read_csv(f"Results/{mmwave_file.replace(MMW_DATASETS_SUFFIX,'')}/{mmwave_file}.csv")
    mmwave_data["x"] = mmwave_data["x"].apply(eval)
    mmwave_data["y"] = mmwave_data["y"].apply(eval)
    mmwave_data['z'] = mmwave_data['z'].apply(eval)
    mmwave_data['velocity'] = mmwave_data['velocity'].apply(eval)
    process_centroids(mmwave_data, f"Results/{mmwave_file.replace(MMW_DATASETS_SUFFIX,'')}/centroid_{mmwave_file}.csv")

# To use this script, execute the following commands:
# git restore Results\*
# py tools\pre_process.mmw_centroid.py
# make sure apply_dataset_filter.bat filters mmwave data (e.g., py tools\filter_dataset_columns.py --input "Results\T125_MMW_A5_BLE_C2P2\centroid_T125_MMW_A5_BLE_C2P2_mmwave_data.csv" --columns centroid_x centroid_y centroid_z --threshold 0.20 --window 7 )
# py tools\pre_process_dataset.py
# py tools\fuse_sensor_data.py
# py tools\plot_mse_rmse_by_distance.py