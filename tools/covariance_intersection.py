import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv, det
from constants import RADAR_PLACEMENT

# ---------------------------
# Covariance Intersection (CI)
# ---------------------------
def covariance_intersection(mean1, cov1, mean2, cov2, criterion="det", eps=1e-6, grid_points=201):
    """
    Fuse two 2D estimates (means, covariances) with Covariance Intersection (CI).
    Returns fused_mean (2,), fused_cov (2x2), and chosen omega.

    criterion: "det" (minimize det of fused covariance) or "trace" (minimize trace).
    eps: small Tikhonov regularization to keep matrices SPD.
    grid_points: number of omega samples in (0,1).
    """
    # Regularize to avoid singularities
    I = np.eye(2)
    P1 = cov1 + eps * I
    P2 = cov2 + eps * I

    P1_inv = inv(P1)
    P2_inv = inv(P2)

    # Search omega in (0,1). Avoid endpoints.
    omegas = np.linspace(0.001, 0.999, grid_points)
    best_val = np.inf
    best = None

    for w in omegas:
        try:
            Pinv = w * P1_inv + (1.0 - w) * P2_inv
            P = inv(Pinv)
            m = P @ (w * (P1_inv @ mean1) + (1.0 - w) * (P2_inv @ mean2))

            val = det(P) if criterion == "det" else np.trace(P)
            if not np.isfinite(val):
                continue
            if val < best_val:
                best_val = val
                best = (m, P, w)
        except np.linalg.LinAlgError:
            continue

    if best is None:
        # Fallback to a neutral weight if the search failed numerically
        w = 0.5
        Pinv = w * P1_inv + (1.0 - w) * P2_inv
        P = inv(Pinv)
        m = P @ (w * (P1_inv @ mean1) + (1.0 - w) * (P2_inv @ mean2))
        return m, P, w

    return best  # (fused_mean, fused_cov, omega)

# ---------------------------
# Data loading & cleaning
# ---------------------------
df = pd.read_csv("FUSAO_PROCESSADA.csv", sep=";")
# Drop np.isnan values
df: pd.DataFrame = df[~np.isnan(df[["x_ble"]]).any(axis=1)]
df = df.reset_index(drop=True)
df["centroid_xyz"] = df["centroid_xyz"].apply(eval)
df["ble_xyz_filter"] = df["ble_xyz_filter"].apply(eval)
df["real_xyz"] = df["real_xyz"].apply(eval)


# Drop rows where BLE x is NaN or lists are malformed
def valid_vec(v, at_least=2):
    if not isinstance(v, (list, tuple)) or len(v) < at_least:
        return False
    return np.all(~pd.isna(v[:at_least]))

mask = df["ble_xyz_filter"].apply(valid_vec) & df["centroid_xyz"].apply(valid_vec) & df["real_xyz"].apply(valid_vec)
df = df.loc[mask].reset_index(drop=True)

# Explode helper columns for convenience
df["x_mmw"] = df["centroid_xyz"].apply(lambda v: float(v[0]))
df["y_mmw"] = df["centroid_xyz"].apply(lambda v: float(v[1]))
df["x_ble_filter"] = df["ble_xyz_filter"].apply(lambda v: float(v[0]))
df["y_ble_filter"] = df["ble_xyz_filter"].apply(lambda v: float(v[1]))

# ---------------------------
# Distance for grouping
# ---------------------------
def calculate_distance(row):
    return np.linalg.norm(np.array(row["real_xyz"]) - RADAR_PLACEMENT)

df["distance"] = df.apply(calculate_distance, axis=1)

# Optional: bin distances to reduce over-fragmentation (uncomment if desired)
# df["distance"] = df["distance"].round(2)

# ---------------------------
# Per-row fusion function (CI)
# ---------------------------
Z_FIXED = 1.78  # keep your Z convention

def fuse_sensor_data_ci(row, mmw_cov, ble_cov, criterion="det"):
    mean_mmwave = np.array([row["x_mmw"], row["y_mmw"]], dtype=float)
    mean_ble = np.array([row["x_ble_filter"], row["y_ble_filter"]], dtype=float)

    fused_mean, fused_cov, omega = covariance_intersection(mean_mmwave, mmw_cov, mean_ble, ble_cov, criterion=criterion)

    fused_xyz = [float(fused_mean[0]), float(fused_mean[1]), Z_FIXED]
    return [
        fused_xyz,
        fused_cov,
        omega
    ]

# ---------------------------
# Group, estimate covariances, fuse
# ---------------------------
df[["sensor_fused_xyz", "fused_cov","ci_omega"]] = np.nan

grouped = df.groupby("distance")
for key, group in grouped:
    # Build 2xN arrays for covariance estimation; require >=2 samples
    mm_stack = np.vstack([group["x_mmw"].values, group["y_mmw"].values]) if len(group) >= 2 else None
    ble_stack = np.vstack([group["x_ble_filter"].values, group["y_ble_filter"].values]) if len(group) >= 2 else None

    # Empirical covariances (unbiased); fallback to diagonal sample variances if necessary
    def safe_cov(stack, gx, gy):
        if stack is not None and stack.shape[1] >= 2:
            C = np.cov(stack, bias=False)  # 2x2
            # Regularize if ill-conditioned
            if not np.isfinite(det(C)) or det(C) <= 0:
                C = np.diag([np.var(gx, ddof=1), np.var(gy, ddof=1)]) + 1e-6 * np.eye(2)
            return C
        # Fallback for tiny groups
        return np.diag([np.var(gx, ddof=0), np.var(gy, ddof=0)]) + 1e-3 * np.eye(2)

    mmw_cov = safe_cov(mm_stack, group["x_mmw"].values, group["y_mmw"].values)
    ble_cov = safe_cov(ble_stack, group["x_ble_filter"].values, group["y_ble_filter"].values)

    # Apply CI row-wise
    group[
        ["sensor_fused_xyz", "fused_cov","ci_omega"]
    ] = group.apply(
        fuse_sensor_data_ci,
        mmw_cov=mmw_cov,
        ble_cov=ble_cov,
        criterion="trace",           # change to "trace" if you prefer A-optimal
        axis=1,
        result_type="expand"
    )
    df.loc[group.index] = group
    # Optional debug prints
    print(f"Distance (group): {key:.4f}")
    print("Sample mmWave cov:\n", mmw_cov)
    print("Sample BLE cov:\n", ble_cov)
    print("First few omegas:\n", df.loc[group.index, "ci_omega"].head().to_list())
    print("===================================")

# ---------------------------
# Save result
# ---------------------------
df.to_csv("fused_dataset.csv", sep=';', index=False)
print("Fused dataset saved to 'fused_dataset.csv'")

# ---------------------------
# Plotting
# ---------------------------
def plot_covariance_by_distance(df, arg1):
    plt.figure()
    plt.plot(df[arg1], df["fused_cov"].apply(lambda m: m[0,0]), marker='o', label='cov_xx', alpha=0.5)
    plt.plot(df[arg1], df["fused_cov"].apply(lambda m: m[0,1]), marker='*', label='cov_xy', alpha=0.5)
    plt.plot(df[arg1], df["fused_cov"].apply(lambda m: m[1,0]), marker='h', label='cov_yy', alpha=0.5)
    plt.title(f"Covariance over {arg1}")
    plt.xlabel(f"{arg1}")
    plt.ylabel("Covariance Component")
    plt.legend()
    plt.tight_layout()
    plt.show()

# If your CSV has 'timestamp' column, keep this; otherwise comment it out.
if "timestamp" in df.columns:
    plot_covariance_by_distance(df, "timestamp")
plot_covariance_by_distance(df, "distance")
