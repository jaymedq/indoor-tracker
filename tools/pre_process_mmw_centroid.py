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
