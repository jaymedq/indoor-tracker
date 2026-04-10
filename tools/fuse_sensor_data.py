import numpy as np
import pandas as pd
from constants import RADAR_PLACEMENT

REGULARIZATION_FACTOR = 0
DEFAULT_COV = np.diag([0.0, 0.0])

def safe_eval_list(s):
    try:
        return eval(s, {"nan": np.nan})
    except NameError:
        return [np.nan, np.nan, np.nan]

def track_to_track_fusion(mean1, cov1, mean2, cov2):
    cov_inv1 = np.linalg.inv(cov1)
    cov_inv2 = np.linalg.inv(cov2)
    fused_cov = np.linalg.inv(cov_inv1 + cov_inv2)
    fused_mean = np.dot(fused_cov, np.dot(cov_inv1, mean1) + np.dot(cov_inv2, mean2))
    return fused_mean, fused_cov

# ==========================================
# 1. Carregar dataset
# ==========================================
def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=";")
    df["centroid_xyz"] = df["centroid_xyz"].apply(eval)
    if "centroid_xyz_replace_filter" in df.columns:
        df["centroid_xyz_replace_filter"] = df["centroid_xyz_replace_filter"].apply(eval)
    df["ble_xyz"] = df["ble_xyz"].apply(eval)
    df["ble_xyz_filter"] = df["ble_xyz_filter"].apply(safe_eval_list)
    df["real_xyz"] = df["real_xyz"].apply(eval)
    return df

# ==========================================
# 2. Agrupar por distancia
# ==========================================
def calculate_group_distances(df: pd.DataFrame) -> pd.DataFrame:
    def calculate_distance(row):
        return np.linalg.norm(np.array(row["real_xyz"]) - RADAR_PLACEMENT)
    df['distance'] = df.apply(calculate_distance, axis=1)
    return df

# ==========================================
# 3. Computar matrizes de covariancia
# ==========================================
def compute_covariance_matrices(group: pd.DataFrame):
    if len(group) <= 1:
        return DEFAULT_COV, DEFAULT_COV, DEFAULT_COV
        
    def safe_var(series, default_variance):
        v = series.dropna().var()
        return default_variance if np.isnan(v) else v

    # MMWave
    mmw_x = group['centroid_xyz'].apply(lambda x: x[0])
    mmw_y = group['centroid_xyz'].apply(lambda x: x[1])
    mmw_x_var = safe_var(mmw_x, DEFAULT_COV[0, 0]) + REGULARIZATION_FACTOR
    mmw_y_var = safe_var(mmw_y, DEFAULT_COV[1, 1]) + REGULARIZATION_FACTOR
    mmw_cov = np.array([[mmw_x_var, mmw_x.cov(mmw_y)], [mmw_x.cov(mmw_y), mmw_y_var]])
    
    # BLE Raw
    ble_x = group['x_ble']
    ble_y = group['y_ble']
    ble_x_var = safe_var(ble_x, DEFAULT_COV[0, 0]) + REGULARIZATION_FACTOR
    ble_y_var = safe_var(ble_y, DEFAULT_COV[1, 1]) + REGULARIZATION_FACTOR
    ble_cov = np.array([[ble_x_var, ble_x.cov(ble_y)], [ble_x.cov(ble_y), ble_y_var]])
    
    # BLE Filtered (SWMF)
    ble_x_f = group['x_ble_filter']
    ble_y_f = group['y_ble_filter']
    ble_x_f_var = safe_var(ble_x_f, DEFAULT_COV[0, 0]) + REGULARIZATION_FACTOR
    ble_y_f_var = safe_var(ble_y_f, DEFAULT_COV[1, 1]) + REGULARIZATION_FACTOR
    ble_cov_filter = np.array([[ble_x_f_var, ble_x_f.cov(ble_y_f)], [ble_x_f.cov(ble_y_f), ble_y_f_var]])
    
    return mmw_cov, ble_cov, ble_cov_filter

# ==========================================
# 4. Aplicar T2TF para cada conjunto
# ==========================================
def apply_t2tf_to_row(row, mmw_cov, ble_cov, mmwave_column="centroid_xyz", ble_column="ble_xyz_filter"):
    mm_meas = row[mmwave_column]
    ble_meas = row[ble_column]
    
    mean_mmwave = np.array(mm_meas[:2])
    mean_ble = np.array(ble_meas[:2])

    is_mmw_valid = not np.isnan(mean_mmwave).any()
    is_ble_valid = not np.isnan(mean_ble).any()

    if is_mmw_valid and is_ble_valid:
        try:
            fused_position, fused_covariance = track_to_track_fusion(mean_mmwave, mmw_cov, mean_ble, ble_cov)
        except np.linalg.LinAlgError:
            # Fallback if fusion math fails (singular matrices)
            if np.trace(mmw_cov) < np.trace(ble_cov):
                fused_position, fused_covariance = mean_mmwave, mmw_cov
            else:
                fused_position, fused_covariance = mean_ble, ble_cov
    else:
        fused_position = np.array([np.nan, np.nan])
        fused_covariance = np.diag([1e6, 1e6])

    # Convert to 3D by appending fixed height Z
    fused_position = fused_position.tolist()
    if np.isnan(fused_position).all():
        fused_position.append(np.nan)
    else:
        fused_position.append(1.78)

    return fused_position, fused_covariance

def process_groups_and_fuse(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("distance")
    
    for key, group in grouped:
        mmw_cov, ble_cov, ble_cov_filter = compute_covariance_matrices(group)

        group[["sensor_fused_xyz", "sensor_fused_cov"]] = group.apply(
            apply_t2tf_to_row, mmw_cov=mmw_cov, ble_cov=ble_cov, 
            mmwave_column="centroid_xyz", ble_column="ble_xyz", axis=1, result_type='expand'
        )
        
        group[["sensor_fused_xyz_filter", "sensor_fused_cov_filter"]] = group.apply(
            apply_t2tf_to_row, mmw_cov=mmw_cov, ble_cov=ble_cov_filter, 
            mmwave_column="centroid_xyz", ble_column="ble_xyz_filter", axis=1, result_type='expand'
        )
        
        df.loc[group.index, ['sensor_fused_xyz', 'sensor_fused_cov', 'sensor_fused_xyz_filter', 'sensor_fused_cov_filter']] = \
            group[['sensor_fused_xyz', 'sensor_fused_cov', 'sensor_fused_xyz_filter', 'sensor_fused_cov_filter']]
            
    return df

# ==========================================
# 5. Salvar no csv / Main Pipeline
# ==========================================
def main():
    print("1. Carregando dataset...")
    df = load_dataset("FUSAO_PROCESSADA.csv")
    
    print("2. Agrupando amostras por distancia radial...")
    df = calculate_group_distances(df)
    
    print("3 & 4. Computando Matrizes de Covariancia por grupo e Aplicando Fuso T2TF...")
    df = process_groups_and_fuse(df)
    
    print("5. Salvando resultado final...")
    df.to_csv("fused_dataset.csv", sep=';', index=False)
    print("Pipeline de Fuso concludo! salvo em: 'fused_dataset.csv'")

if __name__ == "__main__":
    main()