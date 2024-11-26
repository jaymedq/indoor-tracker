import pandas as pd
from datetime import datetime
# Load data
data = pd.read_csv("Resultados/resultado_RADAR_C3_PRD_PA_178_JM.csv",sep=';')
dt_object = data['CreateTime'].apply(datetime.fromtimestamp)
formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')
data['CreateTime'] = formatted_time
