import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats

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

def get_all_errors(df):
    ble_e, mmw_e, fus_e = [], [], []
    ble_x, mmw_x, fus_x = [], [], []
    ble_y, mmw_y, fus_y = [], [], []

    for _, row in df.iterrows():
        try:
            real = safe_eval_list(row['real_xyz'])
            if np.isnan(real[0]): continue
            
            # BLE
            if not np.isnan(row['x_ble']):
                ex = row['x_ble'] - real[0]
                ey = row['y_ble'] - real[1]
                ble_x.append(ex)
                ble_y.append(ey)
                ble_e.append(np.sqrt(ex**2 + ey**2))
                
            # mmWave
            mmw = safe_eval_list(row['centroid_xyz'])
            if not np.isnan(mmw[0]):
                ex = mmw[0] - real[0]
                ey = mmw[1] - real[1]
                mmw_x.append(ex)
                mmw_y.append(ey)
                mmw_e.append(np.sqrt(ex**2 + ey**2))
                
            # Fused
            fus = safe_eval_list(row['sensor_fused_xyz_filter'])
            if not np.isnan(fus[0]):
                ex = fus[0] - real[0]
                ey = fus[1] - real[1]
                fus_x.append(ex)
                fus_y.append(ey)
                fus_e.append(np.sqrt(ex**2 + ey**2))
        except Exception:
            pass

    return {
        'ble_e': filter_errors(ble_e), 'mmw_e': filter_errors(mmw_e), 'fus_e': filter_errors(fus_e),
        'ble_x': filter_errors(ble_x), 'mmw_x': filter_errors(mmw_x), 'fus_x': filter_errors(fus_x),
        'ble_y': filter_errors(ble_y), 'mmw_y': filter_errors(mmw_y), 'fus_y': filter_errors(fus_y)
    }

def main():
    try:
        df = pd.read_csv("fused_dataset.csv", sep=";")
    except FileNotFoundError:
        print("Error: fused_dataset.csv not found.")
        return

    errs = get_all_errors(df)

    # ----------------------------------------------------
    # FIG 1: Euclidean Error (CDF)
    # ----------------------------------------------------
    fw, fh = get_size(width_pt=345)
    fig1, ax1 = plt.subplots(figsize=(fw, fh + 0.6))
    
    def plot_cdf(ax, data, color, label, linewidth):
        if len(data) == 0: return
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=color, linewidth=linewidth, label=label)

    var_ble_e = np.var(errs['ble_e']) if len(errs['ble_e']) > 0 else 0
    var_mmw_e = np.var(errs['mmw_e']) if len(errs['mmw_e']) > 0 else 0
    var_fus_e = np.var(errs['fus_e']) if len(errs['fus_e']) > 0 else 0

    plot_cdf(ax1, errs['ble_e'], 'dodgerblue', rf'BLE ($\sigma^2={var_ble_e:.2f}$)', 1.5)
    plot_cdf(ax1, errs['mmw_e'], 'limegreen', rf'mmWave ($\sigma^2={var_mmw_e:.2f}$)', 1.5)
    plot_cdf(ax1, errs['fus_e'], 'red', rf'Fused T2TF ($\sigma^2={var_fus_e:.2f}$)', 2.0)
    
    ax1.set_xlabel(r"Euclidean Distance to Ground Truth [m]")
    ax1.set_ylabel(r"Cumulative Probability")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlim([0, 3.0])
    ax1.set_ylim([0, 1.05])

    # Place legend at the top
    handles, labels = ax1.get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=8)

    out1 = "ErrorCDF"

    # ----------------------------------------------------
    # FIG 2: X and Y Error Subplots (Gaussian)
    # ----------------------------------------------------
    fw2, fh2 = get_size(width_pt=345, subplots=(1, 2))
    fig2, (axX, axY) = plt.subplots(1, 2, figsize=(fw2, fh2 + 0.5))
    x_range2 = np.linspace(-3, 3, 1000)
    
    # X subplot
    mu_ble_x, std_ble_x = np.mean(errs['ble_x']), np.std(errs['ble_x'])
    mu_mmw_x, std_mmw_x = np.mean(errs['mmw_x']), np.std(errs['mmw_x'])
    mu_fus_x, std_fus_x = np.mean(errs['fus_x']), np.std(errs['fus_x'])
    
    pdf_ble_x = stats.norm.pdf(x_range2, mu_ble_x, std_ble_x)
    pdf_mmw_x = stats.norm.pdf(x_range2, mu_mmw_x, std_mmw_x)
    pdf_fus_x = stats.norm.pdf(x_range2, mu_fus_x, std_fus_x)
    max_pdf_x = max(np.max(pdf_ble_x), np.max(pdf_mmw_x), np.max(pdf_fus_x))
    
    axX.plot(x_range2, pdf_ble_x / max_pdf_x, color='dodgerblue', linewidth=1.5)
    axX.plot(x_range2, pdf_mmw_x / max_pdf_x, color='limegreen', linewidth=1.5)
    axX.plot(x_range2, pdf_fus_x / max_pdf_x, color='red', linewidth=2.0)
    axX.fill_between(x_range2, pdf_fus_x / max_pdf_x, color='red', alpha=0.15)
    axX.set_xlabel(r"Erro no eixo X[m]")
    axX.set_ylabel(r"Densidade Relativa", fontsize=7)
    axX.grid(True, linestyle='--', alpha=0.5)
    axX.set_xlim([-1.8, 1.8])
    
    # Y subplot
    mu_ble_y, std_ble_y = np.mean(errs['ble_y']), np.std(errs['ble_y'])
    mu_mmw_y, std_mmw_y = np.mean(errs['mmw_y']), np.std(errs['mmw_y'])
    mu_fus_y, std_fus_y = np.mean(errs['fus_y']), np.std(errs['fus_y'])
    
    pdf_ble_y = stats.norm.pdf(x_range2, mu_ble_y, std_ble_y)
    pdf_mmw_y = stats.norm.pdf(x_range2, mu_mmw_y, std_mmw_y)
    pdf_fus_y = stats.norm.pdf(x_range2, mu_fus_y, std_fus_y)
    max_pdf_y = max(np.max(pdf_ble_y), np.max(pdf_mmw_y), np.max(pdf_fus_y))
    
    var_ble_x, var_ble_y = std_ble_x**2, std_ble_y**2
    var_mmw_x, var_mmw_y = std_mmw_x**2, std_mmw_y**2
    var_fus_x, var_fus_y = std_fus_x**2, std_fus_y**2

    axY.plot(x_range2, pdf_ble_y / max_pdf_y, color='dodgerblue', linewidth=1.5, label=rf'BLE ($\sigma_x^2={var_ble_x:.2f}$, $\sigma_y^2={var_ble_y:.2f}$)')
    axY.plot(x_range2, pdf_mmw_y / max_pdf_y, color='limegreen', linewidth=1.5, label=rf'mmWave ($\sigma_x^2={var_mmw_x:.2f}$, $\sigma_y^2={var_mmw_y:.2f}$)')
    axY.plot(x_range2, pdf_fus_y / max_pdf_y, color='red', linewidth=2.0, label=rf'Fused ($\sigma_x^2={var_fus_x:.2f}$, $\sigma_y^2={var_fus_y:.2f}$)')
    axY.fill_between(x_range2, pdf_fus_y / max_pdf_y, color='red', alpha=0.15)
    axY.set_xlabel(r"Erro no eixo Y[m]")
    
    ymax = max(axX.get_ylim()[1], axY.get_ylim()[1]) * 1.05
    axX.set_ylim(0, ymax)
    axY.set_ylim(0, ymax)
    
    axY.grid(True, linestyle='--', alpha=0.5)
    axY.set_xlim([-1.8, 1.8])

    # Place a single legend for both subplots at the top
    handles, labels = axY.get_legend_handles_labels()
    fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=8)

    out2 = "ErrorGaussianXY"
    fig2.tight_layout()
    fig1.tight_layout()
    
    fig2.savefig(f"{out2}.png", transparent = True, dpi=300, bbox_inches='tight')
    fig1.savefig(f"{out1}.png", transparent = True, dpi=300, bbox_inches='tight')

    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",     # Matches LaTeX default
        "text.usetex": True,        # Let LaTeX handle the rendering
        "pgf.rcfonts": False,       # Ignore Matplotlib's internal fonts
    })
    fig2.savefig(f"{out2}.pgf", backend='pgf', bbox_inches='tight')
    fig2.savefig(f"{out2}.pdf", bbox_inches='tight')
    print(f"Saved {out2}")

    fig1.savefig(f"{out1}.pgf", backend='pgf', bbox_inches='tight')
    fig1.savefig(f"{out1}.pdf", bbox_inches='tight')
    print(f"Saved {out1}")

if __name__ == "__main__":
    main()
