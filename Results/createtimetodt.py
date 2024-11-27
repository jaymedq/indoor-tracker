import pandas as pd
from datetime import datetime

FILENAME = "resultado_RADAR_C3_PRD_PA_178_JM"
data = pd.read_csv(f"{FILENAME}.csv", sep=';')

def createTimeToDt(row):
    epoch_in_seconds = row['CreateTime'] / 1e6  
    return datetime.fromtimestamp(epoch_in_seconds).strftime('%Y-%m-%d %H:%M:%S')

data['CreateTime'] = data.apply(createTimeToDt, axis=1)

print(data.head())

data.to_csv(f"{FILENAME}_datetime.csv", index=False)
