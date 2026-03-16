import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from plot_room_2d import plot_obstacles, plot_radar_fov, plot_experiment_points
from constants import EXPERIMENT_POINTS

def safe_eval_list(s):
    if isinstance(s, str):
        try:
            return eval(s, {"nan": np.nan, "np": np})
        except:
            pass
    return [np.nan, np.nan, np.nan]

def plot_covariance_ellipse(ax, mean, cov, color, label, n_std=1.0, alpha=0.3):
    """Plots an error ellipse for a given covariance matrix."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                      facecolor=color, edgecolor=color, alpha=alpha, linewidth=1, zorder=3)
    ax.add_patch(ellipse)
    
    import matplotlib.patches as mpatches
    return mpatches.Patch(color=color, alpha=alpha, label=label)

def get_covariances(df, point_name):
    df_point = df[df['experiment_point'] == point_name].copy()
    
    # BLE
    ble_pts = df_point[['x_ble', 'y_ble']].dropna().values
    if len(ble_pts) > 1:
        K_ble = np.cov(ble_pts, rowvar=False)
        mean_ble = np.mean(ble_pts, axis=0)
    else:
        K_ble = np.eye(2) * 0.001
        mean_ble = EXPERIMENT_POINTS[point_name][:2]

    # mmWave (centroid_xyz)
    mmw_pts = []
    for val in df_point['centroid_xyz']:
        val = safe_eval_list(val)
        if len(val) >= 2 and not np.isnan(val[0]):
            mmw_pts.append([val[0], val[1]])
    if len(mmw_pts) > 1:
        K_mmw = np.cov(np.array(mmw_pts), rowvar=False)
        mean_mmw = np.mean(mmw_pts, axis=0)
    else:
        K_mmw = np.eye(2) * 0.001
        mean_mmw = EXPERIMENT_POINTS[point_name][:2]

    # Sensor Fused
    fused_pts = []
    for val in df_point['sensor_fused_xyz_filter'].dropna():
        val = safe_eval_list(val)
        if len(val) >= 2 and not np.isnan(val[0]):
            fused_pts.append([val[0], val[1]])
            
    if len(fused_pts) > 1:
        K_fused = np.cov(np.array(fused_pts), rowvar=False)
        mean_fused = np.mean(fused_pts, axis=0)
    else:
        K_fused = np.eye(2) * 0.001
        mean_fused = EXPERIMENT_POINTS[point_name][:2]

    return K_ble, K_mmw, K_fused, mean_ble, mean_mmw, mean_fused

def main():
    try:
        df = pd.read_csv("fused_dataset.csv", sep=";")
    except FileNotFoundError:
        print("Note: fused_dataset.csv must be in the parent directory.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    plot_radar_fov(ax, True)
    plot_obstacles(ax)
    plot_experiment_points(ax, True)
    
    # We choose C2P2 and C2P5 as examples
    points_to_plot = {
        "C2P2": "Close Range\n(Trusting mmWave)",
        "C2P5": "Far Range\n(Fusion mitigates mmWave variance)"
    }

    n_std = 2.0 # Show 2-sigma ellipses (95% confidence)
    handles = []
    
    for pt, annotation in points_to_plot.items():
        if pt not in EXPERIMENT_POINTS: continue
        p_ideal = np.array(EXPERIMENT_POINTS[pt][:2])
        
        K_ble, K_mmw, K_fused, m_ble, m_mmw, m_fused = get_covariances(df, pt)
        
        l_ble = plot_covariance_ellipse(ax, m_ble, K_ble, 'dodgerblue', 'BLE Footprint', n_std, alpha=0.2)
        l_mm = plot_covariance_ellipse(ax, m_mmw, K_mmw, 'limegreen', 'mmWave Footprint', n_std, alpha=0.2)
        l_f = plot_covariance_ellipse(ax, m_fused, K_fused, 'red', 'Fused T2TF Footprint', n_std, alpha=0.5)
        handles = [l_ble, l_mm, l_f]
        
        # Draw a small marker for the ground truth
        ax.scatter([p_ideal[0]], [p_ideal[1]], color='black', marker='x', s=50, zorder=4)
        
        ax.text(p_ideal[0], p_ideal[1]+0.8, annotation, ha='center', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xlim([0, 10])
    ax.set_ylim([-10, 0])
    
    if handles:
        ax.legend(handles=handles, loc='upper left', fontsize=10)
    
    ax.set_xlabel("X [m]", fontsize=14)
    ax.set_ylabel("Y [m]", fontsize=14)
    
    output_eps = "CovarianceEllipses"
    fig.tight_layout()
    fig.savefig(f"{output_eps}.eps", format='eps')
    print(f"Saved {output_eps}.eps")

if __name__ == "__main__":
    main()
