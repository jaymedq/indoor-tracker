import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats

from constants import RADAR_PLACEMENT

# -----------------------------------------------------------------------
# NOTE on X/Y inversion (kept for audit trail):
#   The original script plotted ble_x → axX and ble_y → axY, which was
#   correct (no inversion). This version replaces that decomposition with
#   polar coordinates (range, azimuth) relative to RADAR_PLACEMENT.
# -----------------------------------------------------------------------

def get_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX."""
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    fig_width_in = fig_width_pt * inches_per_pt
    golden_ratio = (5**.5 - 1) / 2
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)

def safe_eval_list(s):
    if isinstance(s, str):
        try:
            return eval(s, {"nan": np.nan, "np": np})
        except:
            pass
    return [np.nan, np.nan, np.nan]

def filter_errors(err_list, limit=5.0):
    err = np.array(err_list)
    return err[np.abs(err) <= limit]

def cartesian_to_polar(x, y, origin_xy):
    """
    Convert 2-D Cartesian coordinates to polar (range, azimuth) relative to
    an origin. Azimuth is measured from the +X axis, in degrees.

    Parameters
    ----------
    x, y      : scalar - point coordinates [m]
    origin_xy : array-like [ox, oy] - radar placement in the same frame

    Returns
    -------
    r   : range [m]
    az  : azimuth [deg]  (-180, +180]
    """
    dx = x - origin_xy[0]
    dy = y - origin_xy[1]
    r  = np.sqrt(dx**2 + dy**2)
    az = np.degrees(np.arctan2(dy, dx))
    return r, az

MIN_SAMPLES_PER_GROUP = 10  # skip experiment points with too few rows

def get_errors_by_point(df):
    """
    Same polar error decomposition as get_all_errors(), but grouped by
    experiment_point (each point has a unique distance from the radar).

    Returns
    -------
    dict keyed by experiment_point label, each value is:
        { 'distance': float,
          'ble_r': ndarray, 'ble_az': ndarray,
          'mmw_r': ndarray, 'mmw_az': ndarray }
    Only groups with >= MIN_SAMPLES_PER_GROUP valid rows are included.
    """
    origin_xy = RADAR_PLACEMENT[:2]
    result = {}

    for point, group in df.groupby('experiment_point'):
        ble_r, ble_az = [], []
        mmw_r, mmw_az = [], []
        fus_r, fus_az = [], []

        # Distance is pre-computed in the CSV (Euclidean to RADAR_PLACEMENT)
        dist_vals = group['distance'].dropna()
        if len(dist_vals) == 0:
            continue
        distance = float(dist_vals.iloc[0])

        for _, row in group.iterrows():
            try:
                real = safe_eval_list(row['real_xyz'])
                if np.isnan(real[0]):
                    continue
                real_r, real_az = cartesian_to_polar(real[0], real[1], origin_xy)

                # BLE
                if not np.isnan(row['x_ble']):
                    est_r, est_az = cartesian_to_polar(row['x_ble'], row['y_ble'], origin_xy)
                    ble_r.append(est_r - real_r)
                    d = ((est_az - real_az) + 180) % 360 - 180
                    ble_az.append(d)

                # mmWave
                mmw = safe_eval_list(row['centroid_xyz'])
                if not np.isnan(mmw[0]):
                    est_r, est_az = cartesian_to_polar(mmw[0], mmw[1], origin_xy)
                    mmw_r.append(est_r - real_r)
                    d = ((est_az - real_az) + 180) % 360 - 180
                    mmw_az.append(d)

                # Fused
                fus = safe_eval_list(row['sensor_fused_xyz_filter'])
                if not np.isnan(fus[0]):
                    est_r, est_az = cartesian_to_polar(fus[0], fus[1], origin_xy)
                    fus_r.append(est_r - real_r)
                    d = ((est_az - real_az) + 180) % 360 - 180
                    fus_az.append(d)
            except Exception:
                pass

        ble_r_f  = filter_errors(ble_r)
        ble_az_f = filter_errors(ble_az, limit=90.0)
        mmw_r_f  = filter_errors(mmw_r)
        mmw_az_f = filter_errors(mmw_az, limit=90.0)
        fus_r_f  = filter_errors(fus_r)
        fus_az_f = filter_errors(fus_az, limit=90.0)

        if len(ble_r_f) < MIN_SAMPLES_PER_GROUP and len(mmw_r_f) < MIN_SAMPLES_PER_GROUP:
            continue

        result[point] = {
            'distance': distance,
            'ble_r':  ble_r_f,
            'ble_az': ble_az_f,
            'mmw_r':  mmw_r_f,
            'mmw_az': mmw_az_f,
            'fus_r':  fus_r_f,
            'fus_az': fus_az_f,
        }

    return result


def plot_polar_by_distance(by_point, labels=None, pgf_backend=True):
    """
    Produce three figures - BLE, mmWave, Fused - each with
    (range error | azimuth error) subplots.  Every experiment point is one
    PDF curve coloured by its distance from the radar.

    Parameters
    ----------
    by_point : dict  - output of get_errors_by_point()
    labels   : dict  - optional {point_name: annotation} to override legend text
                       (same format as plot_fusion_footprint.py's points_to_plot).
                       When provided a legend is used instead of a colorbar.
    """
    from matplotlib.colors import Normalize
    from matplotlib.colorbar import ColorbarBase

    # Sort groups by distance so the colormap is monotone
    points_sorted = sorted(by_point.items(), key=lambda kv: kv[1]['distance'])
    distances = [v['distance'] for _, v in points_sorted]

    d_min, d_max = min(distances), max(distances)
    norm  = Normalize(vmin=d_min, vmax=d_max)
    cmap  = mpl.colormaps['plasma']      # warm = far, cool = close

    x_r  = np.linspace(-2.0,  2.0, 800)   # range error axis  [m]
    x_az = np.linspace(-30.0, 30.0, 800)  # azimuth error axis [°]

    sensors = [
        ('BLE',    'ble_r',  'ble_az',  'ErrorByDist_BLE'),
        ('mmWave', 'mmw_r',  'mmw_az',  'ErrorByDist_MMW'),
        ('Fused T2TF', 'fus_r', 'fus_az', 'ErrorByDist_Fused'),
    ]

    figs = []
    for sensor_label, key_r, key_az, out_name in sensors:
        fw, fh = get_size(width_pt=345, subplots=(1, 2))
        fig, (axR, axAz) = plt.subplots(
            1, 2, figsize=(fw + 0.8, fh + 0.7)
        )

        for point, data in points_sorted:
            colour = cmap(norm(data['distance']))
            # Use annotation label if provided, otherwise fall back to point + distance
            if labels and point in labels:
                lbl = f"{labels[point]}  ({data['distance']:.1f} m)"
            else:
                lbl = f"{point}  ({data['distance']:.1f} m)"

            arr_r  = data[key_r]
            arr_az = data[key_az]

            # ---- range PDF ----
            if len(arr_r) >= MIN_SAMPLES_PER_GROUP:
                mu, sd = np.mean(arr_r), np.std(arr_r)
                if sd > 1e-6:
                    pdf = stats.norm.pdf(x_r, mu, sd)
                    axR.plot(x_r, pdf / pdf.max(), color=colour, linewidth=1.3, label=lbl)

            # ---- azimuth PDF ----
            if len(arr_az) >= MIN_SAMPLES_PER_GROUP:
                mu, sd = np.mean(arr_az), np.std(arr_az)
                if sd > 1e-6:
                    pdf = stats.norm.pdf(x_az, mu, sd)
                    axAz.plot(x_az, pdf / pdf.max(), color=colour, linewidth=1.3, label=lbl)

        for ax, xlabel in [(axR, r"Range Error [m]"), (axAz, r"Azimuth Error [$^\circ$]")]:
            ax.set_xlabel(xlabel)
            ax.set_ylim([0, 1.15])
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.axvline(0, color='k', linewidth=0.8, linestyle=':')

        axR.set_xlim([-2.0,  2.0])
        axAz.set_xlim([-30.0, 30.0])
        axR.set_ylabel(r"Normalized PDF")

        # Use a legend when a small curated set of points is shown; colorbar otherwise
        if labels:
            axAz.legend(loc='upper right', fontsize=7)
            fig.suptitle(f"{sensor_label} - Polar Error PDF by Distance", fontsize=9, y=1.02)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fig.tight_layout()
        else:
            fig.subplots_adjust(right=0.82)
            cax = fig.add_axes([0.85, 0.15, 0.03, 0.70])
            cb  = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
            cb.set_label(r"Distance to Radar [m]")
            fig.suptitle(f"{sensor_label} - Polar Error PDF by Distance", fontsize=9, y=1.02)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fig.tight_layout(rect=[0, 0, 0.83, 1])


        fig.savefig(f"{out_name}.png", transparent=True, dpi=300, bbox_inches='tight')
        print(f"Saved {out_name}.png")

        figs.append((fig, out_name))

    if pgf_backend:
        mpl.use("pgf")
        mpl.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        })
        for fig, out_name in figs:
            fig.savefig(f"{out_name}.pdf", bbox_inches='tight')
            print(f"Saved {out_name}.pdf")

    return figs


def plot_all_sensors_by_distance(by_point, labels=None):
    """
    Single combined figure: 2 subplots (range error | azimuth error).
    Color  → sensor  (BLE=dodgerblue, mmWave=limegreen, Fused=red)
    Style  → distance (solid=Close Range, dashed=Far Range …)

    Parameters
    ----------
    by_point : dict  - output of get_errors_by_point(), already filtered
    labels   : dict  - {point_name: annotation} (same as points_to_plot)
    """
    import warnings

    # Sort by distance so linestyles are assigned from nearest to farthest
    points_sorted = sorted(by_point.items(), key=lambda kv: kv[1]['distance'])
    linestyles  = ['-', '--', ':', '-.']          # one per distance group
    linewidths  = [1.8, 1.8, 1.5, 1.5]

    sensor_styles = [
        ('BLE',         'ble_r',  'ble_az',  'dodgerblue'),
        ('mmWave',      'mmw_r',  'mmw_az',  'limegreen'),
        ('Fused T2TF',  'fus_r',  'fus_az',  'red'),
    ]

    x_r  = np.linspace(-2.0,  2.0, 800)
    x_az = np.linspace(-30.0, 30.0, 800)

    fw, fh = get_size(width_pt=345, subplots=(1, 2))
    fig, (axR, axAz) = plt.subplots(1, 2, figsize=(fw + 0.4, fh + 0.7))

    for i, (point, data) in enumerate(points_sorted):
        ls  = linestyles[i % len(linestyles)]
        lw  = linewidths[i % len(linewidths)]
        dist_label = labels.get(point, point) if labels else point

        for sensor_label, key_r, key_az, colour in sensor_styles:
            # ---- range PDF ----
            arr_r = data[key_r]
            if len(arr_r) >= MIN_SAMPLES_PER_GROUP:
                mu, sd = np.mean(arr_r), np.std(arr_r)
                if sd > 1e-6:
                    pdf = stats.norm.pdf(x_r, mu, sd)
                    lbl = rf"{sensor_label} - {dist_label} ({data['distance']:.1f} m)"
                    axR.plot(x_r, pdf / pdf.max(), color=colour, linewidth=lw,
                             linestyle=ls, label=lbl)

            # ---- azimuth PDF ----
            arr_az = data[key_az]
            if len(arr_az) >= MIN_SAMPLES_PER_GROUP:
                mu, sd = np.mean(arr_az), np.std(arr_az)
                if sd > 1e-6:
                    pdf = stats.norm.pdf(x_az, mu, sd)
                    axAz.plot(x_az, pdf / pdf.max(), color=colour, linewidth=lw,
                              linestyle=ls)

    for ax, xlabel in [(axR, r"Range Error [m]"), (axAz, r"Azimuth Error [$^\circ$]")]:
        ax.set_xlabel(xlabel)
        ax.set_ylim([0, 1.15])
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axvline(0, color='k', linewidth=0.8, linestyle=':')

    axR.set_xlim([-2.0,  2.0])
    axAz.set_xlim([-30.0, 30.0])
    axR.set_ylabel(r"Normalized PDF")

    # Reserve top band for the legend before tight_layout so y-axis label is not displaced
    handles, leg_labels = axR.get_legend_handles_labels()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # rect leaves room at the top for a 2-row legend (≈18 % of figure height)
        fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig.legend(handles, leg_labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=3, fontsize=7, framealpha=0.9,
               bbox_transform=fig.transFigure,
               borderaxespad=0.3)

    out_name = "ErrorByDist_All"
    fig.savefig(f"{out_name}.png", transparent=True, dpi=300, bbox_inches='tight')
    print(f"Saved {out_name}.png")
    return [(fig, out_name)]


def get_all_errors(df):
    """
    Extract Euclidean distance errors and polar component errors
    (range error, azimuth error) for BLE, mmWave, and fused estimates.

    Polar errors are computed as:
        range_err  = range(estimate) - range(ground-truth)   [m]
        azimuth_err = azimuth(estimate) - azimuth(ground-truth) [deg]
    Both relative to RADAR_PLACEMENT (x, y only; z is ignored).
    """
    origin_xy = RADAR_PLACEMENT[:2]  # [x, y] of the radar

    ble_e,  mmw_e,  fus_e  = [], [], []
    ble_r,  mmw_r,  fus_r  = [], [], []   # range error
    ble_az, mmw_az, fus_az = [], [], []   # azimuth error [deg]

    for _, row in df.iterrows():
        try:
            real = safe_eval_list(row['real_xyz'])
            if np.isnan(real[0]):
                continue

            real_r, real_az = cartesian_to_polar(real[0], real[1], origin_xy)

            # ---- BLE ----
            if not np.isnan(row['x_ble']):
                ex = row['x_ble'] - real[0]
                ey = row['y_ble'] - real[1]
                ble_e.append(np.sqrt(ex**2 + ey**2))

                est_r, est_az = cartesian_to_polar(row['x_ble'], row['y_ble'], origin_xy)
                ble_r.append(est_r - real_r)
                # Wrap azimuth difference to (-180, +180]
                d_az = est_az - real_az
                d_az = (d_az + 180) % 360 - 180
                ble_az.append(d_az)

            # ---- mmWave ----
            mmw = safe_eval_list(row['centroid_xyz'])
            if not np.isnan(mmw[0]):
                ex = mmw[0] - real[0]
                ey = mmw[1] - real[1]
                mmw_e.append(np.sqrt(ex**2 + ey**2))

                est_r, est_az = cartesian_to_polar(mmw[0], mmw[1], origin_xy)
                mmw_r.append(est_r - real_r)
                d_az = est_az - real_az
                d_az = (d_az + 180) % 360 - 180
                mmw_az.append(d_az)

            # ---- Fused ----
            fus = safe_eval_list(row['sensor_fused_xyz_filter'])
            if not np.isnan(fus[0]):
                ex = fus[0] - real[0]
                ey = fus[1] - real[1]
                fus_e.append(np.sqrt(ex**2 + ey**2))

                est_r, est_az = cartesian_to_polar(fus[0], fus[1], origin_xy)
                fus_r.append(est_r - real_r)
                d_az = est_az - real_az
                d_az = (d_az + 180) % 360 - 180
                fus_az.append(d_az)

        except Exception:
            pass

    return {
        'ble_e':  filter_errors(ble_e),
        'mmw_e':  filter_errors(mmw_e),
        'fus_e':  filter_errors(fus_e),
        'ble_r':  filter_errors(ble_r),
        'mmw_r':  filter_errors(mmw_r),
        'fus_r':  filter_errors(fus_r),
        'ble_az': filter_errors(ble_az,  limit=90.0),
        'mmw_az': filter_errors(mmw_az,  limit=90.0),
        'fus_az': filter_errors(fus_az,  limit=90.0),
    }

def main():
    try:
        df = pd.read_csv("fused_dataset.csv", sep=";")
    except FileNotFoundError:
        print("Error: fused_dataset.csv not found.")
        return

    errs = get_all_errors(df)

    # ── Diagnostic summary (printed every run for audit purposes) ──────────
    def _stats(arr):
        return np.mean(arr), np.std(arr), len(arr)

    mu_ble_r_,  std_ble_r_,  n = _stats(errs['ble_r'])
    mu_mmw_r_,  std_mmw_r_,  n = _stats(errs['mmw_r'])
    mu_fus_r_,  std_fus_r_,  n = _stats(errs['fus_r'])
    mu_ble_az_, std_ble_az_, n = _stats(errs['ble_az'])
    mu_mmw_az_, std_mmw_az_, n = _stats(errs['mmw_az'])
    mu_fus_az_, std_fus_az_, n = _stats(errs['fus_az'])
    print("=== Polar Error Summary (relative to RADAR_PLACEMENT) ===")
    print(f"  RADAR origin  : {RADAR_PLACEMENT[:2]}")
    print(f"  {'Sensor':<8}  {'range μ':>8}  {'range σ':>8}  {'az μ[°]':>9}  {'az σ[°]':>9}  n")
    print(f"  {'BLE':<8}  {mu_ble_r_:>8.4f}  {std_ble_r_:>8.4f}  {mu_ble_az_:>9.4f}  {std_ble_az_:>9.4f}  {len(errs['ble_r'])}")
    print(f"  {'mmWave':<8}  {mu_mmw_r_:>8.4f}  {std_mmw_r_:>8.4f}  {mu_mmw_az_:>9.4f}  {std_mmw_az_:>9.4f}  {len(errs['mmw_r'])}")
    print(f"  {'Fused':<8}  {mu_fus_r_:>8.4f}  {std_fus_r_:>8.4f}  {mu_fus_az_:>9.4f}  {std_fus_az_:>9.4f}  {len(errs['fus_r'])}")
    print()
    # ───────────────────────────────────────────────────────────────────────


    # ----------------------------------------------------
    # FIG 1: Euclidean Error (CDF)
    # ----------------------------------------------------
    fw, fh = get_size(width_pt=345)
    fig1, ax1 = plt.subplots(figsize=(fw, fh + 0.6))

    def plot_cdf(ax, data, color, label, linewidth):
        if len(data) == 0:
            return
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=color, linewidth=linewidth, label=label)

    var_ble_e = np.var(errs['ble_e']) if len(errs['ble_e']) > 0 else 0
    var_mmw_e = np.var(errs['mmw_e']) if len(errs['mmw_e']) > 0 else 0
    var_fus_e = np.var(errs['fus_e']) if len(errs['fus_e']) > 0 else 0

    plot_cdf(ax1, errs['ble_e'], 'dodgerblue', rf'BLE ($\sigma^2={var_ble_e:.2f}$)', 1.5)
    plot_cdf(ax1, errs['mmw_e'], 'limegreen',  rf'mmWave ($\sigma^2={var_mmw_e:.2f}$)', 1.5)
    plot_cdf(ax1, errs['fus_e'], 'red',        rf'Fused T2TF ($\sigma^2={var_fus_e:.2f}$)', 2.0)

    ax1.set_xlabel(r"Euclidean Distance to Ground Truth [m]")
    ax1.set_ylabel(r"Cumulative Probability")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlim([0, 3.0])
    ax1.set_ylim([0, 1.05])

    handles, labels = ax1.get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=8)

    out1 = "ErrorCDF"

    # ----------------------------------------------------
    # FIG 2: Range and Azimuth Error Subplots (Gaussian)
    # Polar decomposition relative to RADAR_PLACEMENT.
    # PDFs are normalized to [0, 1].
    # ----------------------------------------------------
    fw2, fh2 = get_size(width_pt=345, subplots=(1, 2))
    fig2, (axR, axAz) = plt.subplots(1, 2, figsize=(fw2, fh2 + 0.5))

    x_range_r  = np.linspace(-3, 3, 1000)       # range error  [m]
    x_range_az = np.linspace(-45, 45, 1000)     # azimuth error [deg]

    # ---- Range subplot ----
    mu_ble_r,  std_ble_r  = np.mean(errs['ble_r']),  np.std(errs['ble_r'])
    mu_mmw_r,  std_mmw_r  = np.mean(errs['mmw_r']),  np.std(errs['mmw_r'])
    mu_fus_r,  std_fus_r  = np.mean(errs['fus_r']),  np.std(errs['fus_r'])

    pdf_ble_r  = stats.norm.pdf(x_range_r, mu_ble_r,  std_ble_r)
    pdf_mmw_r  = stats.norm.pdf(x_range_r, mu_mmw_r,  std_mmw_r)
    pdf_fus_r  = stats.norm.pdf(x_range_r, mu_fus_r,  std_fus_r)
    max_pdf_r  = max(np.max(pdf_ble_r), np.max(pdf_mmw_r), np.max(pdf_fus_r))

    axR.plot(x_range_r, pdf_ble_r / max_pdf_r, color='dodgerblue', linewidth=1.5)
    axR.plot(x_range_r, pdf_mmw_r / max_pdf_r, color='limegreen',  linewidth=1.5)
    axR.plot(x_range_r, pdf_fus_r / max_pdf_r, color='red',        linewidth=2.0)
    axR.fill_between(x_range_r, pdf_fus_r / max_pdf_r, color='red', alpha=0.15)
    axR.set_xlabel(r"Range Error [m]")
    axR.set_ylabel(r"Normalized PDF", fontsize=7)
    axR.set_xlim([-1.8, 1.8])
    axR.set_ylim([0, 1.1])
    axR.grid(True, linestyle='--', alpha=0.5)

    # ---- Azimuth subplot ----
    mu_ble_az,  std_ble_az  = np.mean(errs['ble_az']),  np.std(errs['ble_az'])
    mu_mmw_az,  std_mmw_az  = np.mean(errs['mmw_az']),  np.std(errs['mmw_az'])
    mu_fus_az,  std_fus_az  = np.mean(errs['fus_az']),  np.std(errs['fus_az'])

    pdf_ble_az  = stats.norm.pdf(x_range_az, mu_ble_az,  std_ble_az)
    pdf_mmw_az  = stats.norm.pdf(x_range_az, mu_mmw_az,  std_mmw_az)
    pdf_fus_az  = stats.norm.pdf(x_range_az, mu_fus_az,  std_fus_az)
    max_pdf_az  = max(np.max(pdf_ble_az), np.max(pdf_mmw_az), np.max(pdf_fus_az))

    var_ble_r,  var_ble_az  = std_ble_r**2,  std_ble_az**2
    var_mmw_r,  var_mmw_az  = std_mmw_r**2,  std_mmw_az**2
    var_fus_r,  var_fus_az  = std_fus_r**2,  std_fus_az**2

    axAz.plot(x_range_az, pdf_ble_az / max_pdf_az, color='dodgerblue', linewidth=1.5,
              label=rf'BLE ($\sigma_r^2={var_ble_r:.2f}$, $\sigma_{{az}}^2={var_ble_az:.2f}$)')
    axAz.plot(x_range_az, pdf_mmw_az / max_pdf_az, color='limegreen',  linewidth=1.5,
              label=rf'mmWave ($\sigma_r^2={var_mmw_r:.2f}$, $\sigma_{{az}}^2={var_mmw_az:.2f}$)')
    axAz.plot(x_range_az, pdf_fus_az / max_pdf_az, color='red',        linewidth=2.0,
              label=rf'Fused ($\sigma_r^2={var_fus_r:.2f}$, $\sigma_{{az}}^2={var_fus_az:.2f}$)')
    axAz.fill_between(x_range_az, pdf_fus_az / max_pdf_az, color='red', alpha=0.15)
    axAz.set_xlabel(r"Azimuth Error [$^\circ$]")
    axAz.set_xlim([-45, 45])
    axAz.set_ylim([0, 1.1])
    axAz.grid(True, linestyle='--', alpha=0.5)

    # Single legend for both subplots
    handles, labels = axAz.get_legend_handles_labels()
    fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=8)

    out2 = "ErrorGaussianPolar"
    fig2.tight_layout()
    fig1.tight_layout()

    fig2.savefig(f"{out2}.png", transparent=True, dpi=300, bbox_inches='tight')
    fig1.savefig(f"{out1}.png", transparent=True, dpi=300, bbox_inches='tight')
    print(f"Saved {out2}.png and {out1}.png")

    # Return figures for deferred PDF/PGF export (backend switch done in __main__)
    return [(fig2, out2), (fig1, out1)]

def main_by_distance():
    """Generate the per-distance polar error figures. Returns list of (fig, name)."""
    # Same pattern as plot_fusion_footprint.py - only these two points are plotted.
    points_to_plot = {"C2P3": "Close Range", "C3P5": "Far Range"}

    try:
        df = pd.read_csv("fused_dataset.csv", sep=";")
    except FileNotFoundError:
        print("Error: fused_dataset.csv not found.")
        return []

    if 'experiment_point' not in df.columns or 'distance' not in df.columns:
        print("Error: fused_dataset.csv must contain 'experiment_point' and 'distance' columns.")
        return []

    by_point = get_errors_by_point(df)

    # Filter to the selected points only
    by_point = {k: v for k, v in by_point.items() if k in points_to_plot}
    if not by_point:
        print(f"Warning: none of {list(points_to_plot.keys())} found in dataset.")
        return []

    print(f"Plotting {len(by_point)} experiment points:")
    for pt, d in sorted(by_point.items(), key=lambda kv: kv[1]['distance']):
        print(f"  {pt:8s} ({points_to_plot[pt]})  dist={d['distance']:.2f}m  "
              f"ble_r n={len(d['ble_r'])}  mmw_r n={len(d['mmw_r'])}  fus_r n={len(d['fus_r'])}")

    # PNG only here - PDF written after pgf backend switch in __main__
    figs = plot_polar_by_distance(by_point, labels=points_to_plot, pgf_backend=False)
    figs += plot_all_sensors_by_distance(by_point, labels=points_to_plot)
    return figs


if __name__ == "__main__":
    # ── Phase 1: build all figures and save PNGs (Agg backend) ────────────
    figs_main = main()
    figs_dist = main_by_distance()
    all_figs  = (figs_main or []) + (figs_dist or [])

    # ── Phase 2: switch to PGF once and export PDFs/PGFs ──────────────────
    plt.close('all')   # suppress deprecation warning
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    })
    for fig, name in all_figs:
        fig.savefig(f"{name}.pgf", backend='pgf', bbox_inches='tight')
        fig.savefig(f"{name}.pdf", bbox_inches='tight')
        print(f"Saved {name}.pdf")
