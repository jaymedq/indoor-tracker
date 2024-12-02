from datetime import datetime
import pandas as pd

# Load the datasets
BLE_DATASET_FILES = [
    "resultado_RADAR_C3_PRD_P12_178_JM",
    "resultado_RADAR_C3_PRD_P13_178_JM",
    "resultado_RADAR_C3_PRD_P14_178_JM",
    "resultado_RADAR_C3_PRD_P15_178_JM",
    "resultado_RADAR_C3_PRD_PA_178_JM",
]
MMWAVE_DATASET_FILE = "ResultadoMMWave"

EXPERIMENT_POINTS = {
    "PA": [7.1, -6.865, 1.78],
    "P11": [1.102, -6.865, 1.78],
    "P12": [2.308, -6.865, 1.78],
    "P13": [3.503, -6.865, 1.78],
    "P14": [4.7, -6.865, 1.78], 
    "P15": [5.9, -6.865, 1.78],
}

def createTimeToDt(row):
    try:
        epoch_in_seconds = row["CreateTime"]
        return datetime.fromtimestamp(epoch_in_seconds).strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        print(row)
        raise e

for ble_file in BLE_DATASET_FILES:
    ble_data = pd.read_csv(f"{ble_file}.csv")
    ble_data["CreateTime"] = ble_data.apply(createTimeToDt, axis=1)
    mmwave_data = pd.read_csv(f"{MMWAVE_DATASET_FILE}.csv")

    ble_data["CreateTime"] = pd.to_datetime(
        ble_data["CreateTime"], format="%Y-%m-%d %H:%M:%S"
    )
    mmwave_data["timestamp"] = pd.to_datetime(
        mmwave_data["timestamp"], format="%d/%m/%Y %H:%M:%S"
    )
    fusion_data = pd.merge(
        ble_data, mmwave_data, left_on="CreateTime", right_on="timestamp", how="inner"
    )

    for point in EXPERIMENT_POINTS.keys():
        if f"_{point}_" in ble_file:
            fusion_data["real_xyz"] = f"{EXPERIMENT_POINTS[point]}"

    BLE_MMWAVE_FUSION_FILENAME = f"{ble_file}_mmwave_fusion.csv"
    fusion_data.to_csv(BLE_MMWAVE_FUSION_FILENAME, index=False)
    print(f"Fusion dataset saved asf {BLE_MMWAVE_FUSION_FILENAME}")
