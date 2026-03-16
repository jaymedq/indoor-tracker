import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull

# --- STEP 1: PGF CONFIGURATION ---
# This must happen BEFORE plt.subplots()
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",     # Matches LaTeX default
    "text.usetex": True,        # Let LaTeX handle the rendering
    "pgf.rcfonts": False,       # Ignore Matplotlib's internal fonts
})

from plot_room_2d import plot_obstacles, plot_radar_fov, plot_experiment_points
from constants import EXPERIMENT_POINTS

# --- UTILS ---
def get_size(width_pt, fraction=1, subplots=(1, 1), aspect_ratio=None):
    """Set figure dimensions to avoid scaling in LaTeX."""
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    fig_width_in = fig_width_pt * inches_per_pt
    if aspect_ratio is None:
        golden_ratio = (5**.5 - 1) / 2
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    else:
        fig_height_in = fig_width_in * aspect_ratio
    return (fig_width_in, fig_height_in)

def safe_eval_list(s):
    if isinstance(s, str):
        try:
            return eval(s, {"nan": np.nan, "np": np})
        except:
            pass
    return [np.nan, np.nan, np.nan]

def remove_outliers_distance(points, threshold=3.0):
    if len(points) < 3:
        return points 
    mean_pt = np.mean(points, axis=0)
    distances = np.linalg.norm(points - mean_pt, axis=1)
    mask = distances <= threshold
    return points[mask]

def get_points(df, point_name):
    df_point = df[df['experiment_point'] == point_name].copy()
    ble_pts = df_point[['x_ble', 'y_ble']].dropna().values

    mmw_pts = []
    for val in df_point['centroid_xyz']:
        val = safe_eval_list(val)
        if len(val) >= 2 and not np.isnan(val[0]):
            mmw_pts.append([val[0], val[1]])
    mmw_pts = np.array(mmw_pts) if len(mmw_pts) > 0 else np.array([])

    fused_pts = []
    for val in df_point['sensor_fused_xyz_filter'].dropna():
        val = safe_eval_list(val)
        if len(val) >= 2 and not np.isnan(val[0]):
            fused_pts.append([val[0], val[1]])
    fused_pts = np.array(fused_pts) if len(fused_pts) > 0 else np.array([])
            
    ble_pts = remove_outliers_distance(ble_pts, threshold=7.0)
    mmw_pts = remove_outliers_distance(mmw_pts, threshold=7.0)
    fused_pts = remove_outliers_distance(fused_pts, threshold=7.0)
            
    return ble_pts, mmw_pts, fused_pts

def plot_footprint_hull(ax, points, color, label, alpha=0.3):
    if len(points) < 3:
        return None
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    # In PGF, alpha is handled natively
    poly = Polygon(hull_points, facecolor=color, edgecolor=color, alpha=alpha, linewidth=1.0, zorder=3)
    ax.add_patch(poly)
    return mpatches.Patch(color=color, alpha=alpha, label=label)

def main():
    try:
        df = pd.read_csv("fused_dataset.csv", sep=";")
    except FileNotFoundError:
        print("Error: fused_dataset.csv not found.")
        return

    # --- STEP 2: SIZE CALCULATION ---
    # Give the square map explicit 1:1 aspect height so it doesn't artificially shrink to the golden ratio edge
    fig, ax = plt.subplots(figsize=get_size(width_pt=345, aspect_ratio=1.0))
    
    plot_radar_fov(ax, True)
    plot_obstacles(ax)
    plot_experiment_points(ax, True)
    
    points_to_plot = {"C2P2": "Close Range", "C3P5": "Far Range"}
    handles = []
    
    for pt, annotation in points_to_plot.items():
        if pt not in EXPERIMENT_POINTS: continue
        p_ideal = np.array(EXPERIMENT_POINTS[pt][:2])
        ble_pts, mmw_pts, fused_pts = get_points(df, pt)
        
        h_ble = plot_footprint_hull(ax, ble_pts, 'dodgerblue', 'BLE', alpha=0.3)
        h_mm = plot_footprint_hull(ax, mmw_pts, 'limegreen', 'mmWave', alpha=0.4)
        h_f = plot_footprint_hull(ax, fused_pts, 'red', 'Fused T2TF', alpha=0.6)
            
        if not handles:
            if h_ble: handles.append(h_ble)
            if h_mm: handles.append(h_mm)
            if h_f: handles.append(h_f)
        
        ax.scatter([p_ideal[0]], [p_ideal[1]], color='black', marker='x', s=20, zorder=5)
        # Using LaTeX math mode in the label
        ax.text(p_ideal[0], p_ideal[1]+0.6, rf"\textsf{{{annotation}}}", 
                ha='center', va='bottom', fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round'), zorder=6)

    ax.set_xlim([0, 10])
    ax.set_ylim([-10, 0])
    ax.set_aspect('equal')
    
    if handles:
        ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=9)
    
    ax.set_xlabel(r"X position [m]")
    ax.set_ylabel(r"Y position [m]")
    
    # --- STEP 3: SAVE AS PGF ---
    output = "FootprintAreas"
    fig.tight_layout()
    # Save the PGF for LaTeX and a PDF for quick viewing (PGFs are hard to preview)
    fig.savefig(f"{output}.pgf", backend='pgf')
    fig.savefig(f"{output}.pdf") 
    print(f"Saved {output}.pgf and {output}.pdf")

if __name__ == "__main__":
    main()