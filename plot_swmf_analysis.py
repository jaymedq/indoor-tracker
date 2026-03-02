"""
Sliding Window Median Filter (SWMF) Analysis Figure Generator
Produces individual thesis-ready EPS figures from threshold_window_experiment_results.csv.

Output files
------------
SWMF_heatmaps.eps / .png   – two heatmaps (RMSE_Fusion + replace rate) stacked
SWMF_lineplots.eps / .png  – RMSE vs threshold (per window) + RMSE vs window size
SWMF_analysis.eps / .png   – combined 4-panel overview (preview)
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    "font.family":  "serif",
    "font.size":    9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi":   150,
})

CSV_PATH = "threshold_window_experiment_results.csv"

BLUE_CMAP  = LinearSegmentedColormap.from_list("blu", ["#cce5ff", "#084594"])
GREEN_CMAP = LinearSegmentedColormap.from_list("grn", ["#edf8e9", "#005a32"])

CHOSEN_T = 0.20   # selected threshold
CHOSEN_W = 7      # selected window size


def _load(path):
    df = pd.read_csv(path)
    windows    = sorted(df["window"].unique())
    thresholds = sorted(df["threshold"].unique())
    rmse_pivot = df.groupby(["window","threshold"])["RMSE_Fusion"].mean().unstack("threshold")
    rr_pivot   = df.groupby(["window","threshold"])["filter_replace_rate"].mean().unstack("threshold")
    ref_ble    = df["RMSE_BLE"].mean()
    ref_mmw    = df["RMSE_MMW"].mean()
    return df, windows, thresholds, rmse_pivot, rr_pivot, ref_ble, ref_mmw


def _mark_cell(ax, windows, thresholds, w, t, color="red", lw=2):
    ri = windows.index(w)
    ci = thresholds.index(t)
    ax.add_patch(plt.Rectangle(
        (ci - 0.5, ri - 0.5), 1, 1,
        linewidth=lw, edgecolor=color, facecolor="none", zorder=5
    ))


def _annotate_heatmap(ax, pivot, windows, thresholds, fmt, threshold_white):
    for ri, w in enumerate(windows):
        for ci, t in enumerate(thresholds):
            val = pivot.loc[w, t]
            clr = "white" if val > threshold_white else "black"
            ax.text(ci, ri, fmt.format(val), ha="center", va="center",
                    fontsize=5.5, color=clr)


def fig_heatmaps(df, windows, thresholds, rmse_pivot, rr_pivot):
    """Figure 1 – stacked heatmaps."""
    th_labels = [f"{t:.2f}" for t in thresholds]
    w_labels  = [str(w) for w in windows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.5),
                                   gridspec_kw={"hspace": 0.55})

    # ── RMSE heatmap ────────────────────────────────────────────────────────
    im1 = ax1.imshow(rmse_pivot.values, aspect="auto",
                     cmap=BLUE_CMAP, vmin=0.19, vmax=0.25)
    ax1.set_xticks(range(len(thresholds))); ax1.set_xticklabels(th_labels)
    ax1.set_yticks(range(len(windows)));   ax1.set_yticklabels(w_labels)
    ax1.set_xlabel("Threshold (m)"); ax1.set_ylabel("Window size $W$")
    ax1.set_title(r"(a) Mean $\mathrm{RMSE}_\mathrm{Fusion}$ (m)", fontweight="bold")
    cb1 = fig.colorbar(im1, ax=ax1, shrink=0.9, pad=0.01)
    cb1.set_label("RMSE (m)")
    _annotate_heatmap(ax1, rmse_pivot, windows, thresholds, "{:.3f}", 0.225)
    _mark_cell(ax1, windows, thresholds, CHOSEN_W, CHOSEN_T)

    # ── Replace-rate heatmap ─────────────────────────────────────────────────
    im2 = ax2.imshow(rr_pivot.values, aspect="auto",
                     cmap=GREEN_CMAP, vmin=0, vmax=90)
    ax2.set_xticks(range(len(thresholds))); ax2.set_xticklabels(th_labels)
    ax2.set_yticks(range(len(windows)));   ax2.set_yticklabels(w_labels)
    ax2.set_xlabel("Threshold (m)"); ax2.set_ylabel("Window size $W$")
    ax2.set_title("(b) Mean Filter Replace Rate (%)", fontweight="bold")
    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.9, pad=0.01)
    cb2.set_label("Replace rate (%)")
    _annotate_heatmap(ax2, rr_pivot, windows, thresholds, "{:.0f}", 50)
    _mark_cell(ax2, windows, thresholds, CHOSEN_W, CHOSEN_T)

    return fig


def fig_lineplots(df, windows, thresholds, ref_ble, ref_mmw):
    """Figure 2 – line plots."""
    cmap_lines = plt.get_cmap("tab10")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2),
                                   gridspec_kw={"wspace": 0.42})

    # ── Panel C: RMSE vs threshold ───────────────────────────────────────────
    for idx, w in enumerate(windows):
        sub  = df[df["window"] == w].groupby("threshold")["RMSE_Fusion"].mean()
        lw   = 2.2 if w == CHOSEN_W else 1.0
        zo   = 3   if w == CHOSEN_W else 1
        lbl  = f"$W$={w}" + (" (selected)" if w == CHOSEN_W else "")
        ax1.plot(sub.index * 100, sub.values,
                 marker="o", markersize=2.5, linewidth=lw, zorder=zo,
                 color=cmap_lines(idx), label=lbl)

    ax1.axhline(ref_ble, color="dimgray", linestyle="--", linewidth=1,
                label=f"BLE only ({ref_ble:.3f} m)")
    ax1.axhline(ref_mmw, color="black",   linestyle=":",  linewidth=1,
                label=f"mmWave only ({ref_mmw:.3f} m)")
    ax1.axvline(CHOSEN_T * 100, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Threshold (cm)")
    ax1.set_ylabel(r"Mean $\mathrm{RMSE}_\mathrm{Fusion}$ (m)")
    ax1.set_title(r"(a) RMSE$_\mathrm{Fusion}$ vs.\ Threshold", fontweight="bold")
    ax1.legend(ncol=2, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # ── Panel D: RMSE + replace-rate vs window ───────────────────────────────
    sub_w  = df[df["threshold"] == CHOSEN_T].groupby("window")["RMSE_Fusion"].mean()
    sub_rr = df[df["threshold"] == CHOSEN_T].groupby("window")["filter_replace_rate"].mean()

    ax2b = ax2.twinx()
    ax2b.bar(sub_rr.index, sub_rr.values, width=1.4,
             color="salmon", alpha=0.45, label="Replace rate", zorder=1)
    ax2b.set_ylabel("Mean replace rate (%)", color="salmon")
    ax2b.tick_params(axis="y", colors="salmon")
    ax2b.set_ylim(0, 100)

    ax2.plot(sub_w.index, sub_w.values, "b-o", linewidth=2,
             markersize=5, zorder=3, label=r"RMSE$_\mathrm{Fusion}$")
    ax2.axhline(ref_ble, color="dimgray", linestyle="--", linewidth=1, label="BLE only")
    ax2.axhline(ref_mmw, color="black",   linestyle=":",  linewidth=1, label="mmWave only")
    ax2.axvline(CHOSEN_W, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Window size $W$")
    ax2.set_ylabel(r"Mean $\mathrm{RMSE}_\mathrm{Fusion}$ (m)", color="blue")
    ax2.tick_params(axis="y", colors="blue")
    ax2.set_title(
        r"(b) RMSE$_\mathrm{Fusion}$ vs.\ Window Size"
        f"\n(threshold = {CHOSEN_T} m)",
        fontweight="bold"
    )
    ax2.set_xticks(windows)
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    return fig


def _save(fig, stem):
    fig.savefig(f"{stem}.eps", format="eps", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight")
    print(f"Saved: {stem}.eps  /  {stem}.png")
    plt.close(fig)


def main():
    df, windows, thresholds, rmse_pivot, rr_pivot, ref_ble, ref_mmw = _load(CSV_PATH)

    _save(fig_heatmaps(df, windows, thresholds, rmse_pivot, rr_pivot),
          "SWMF_heatmaps")

    _save(fig_lineplots(df, windows, thresholds, ref_ble, ref_mmw),
          "SWMF_lineplots")

    # ── combined overview (preview only) ────────────────────────────────────
    fig_all = plt.figure(figsize=(13, 11))
    gs = fig_all.add_gridspec(3, 2, hspace=0.52, wspace=0.38,
                              height_ratios=[1, 1, 1.1])
    th_labels = [f"{t:.2f}" for t in thresholds]
    w_labels  = [str(w) for w in windows]
    ax_a = fig_all.add_subplot(gs[0, :])
    ax_b = fig_all.add_subplot(gs[1, :])
    ax_c = fig_all.add_subplot(gs[2, 0])
    ax_d = fig_all.add_subplot(gs[2, 1])

    for ax, pivot, cmap, vmax, fmt, tw, title, cb_lbl in [
        (ax_a, rmse_pivot, BLUE_CMAP,  0.25, "{:.3f}", 0.225,
         r"(a) Mean $\mathrm{RMSE}_\mathrm{Fusion}$ (m)", "RMSE (m)"),
        (ax_b, rr_pivot,   GREEN_CMAP, 90,   "{:.0f}", 50,
         "(b) Mean Filter Replace Rate (%)", "Replace rate (%)"),
    ]:
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0 if cmap==GREEN_CMAP else 0.19, vmax=vmax)
        ax.set_xticks(range(len(thresholds))); ax.set_xticklabels(th_labels, fontsize=8)
        ax.set_yticks(range(len(windows)));   ax.set_yticklabels(w_labels, fontsize=8)
        ax.set_xlabel("Threshold (m)", fontsize=9); ax.set_ylabel("Window size $W$", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        cb = fig_all.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
        cb.ax.tick_params(labelsize=7); cb.set_label(cb_lbl, fontsize=8)
        _annotate_heatmap(ax, pivot, windows, thresholds, fmt, tw)
        _mark_cell(ax, windows, thresholds, CHOSEN_W, CHOSEN_T)

    cmap_lines = plt.get_cmap("tab10")
    for idx, w in enumerate(windows):
        sub = df[df["window"] == w].groupby("threshold")["RMSE_Fusion"].mean()
        lw = 2.2 if w == CHOSEN_W else 1.0
        ax_c.plot(sub.index * 100, sub.values, marker="o", markersize=2.5,
                  linewidth=lw, color=cmap_lines(idx),
                  label=f"$W$={w}" + (" *" if w == CHOSEN_W else ""))
    ax_c.axhline(ref_ble, color="dimgray", linestyle="--", linewidth=1, label=f"BLE ({ref_ble:.3f} m)")
    ax_c.axhline(ref_mmw, color="black",   linestyle=":",  linewidth=1, label=f"mmWave ({ref_mmw:.3f} m)")
    ax_c.axvline(CHOSEN_T * 100, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax_c.set_xlabel("Threshold (cm)", fontsize=9); ax_c.set_ylabel(r"Mean RMSE$_F$ (m)", fontsize=9)
    ax_c.set_title(r"(c) RMSE$_F$ vs. Threshold", fontsize=10, fontweight="bold")
    ax_c.legend(fontsize=6, ncol=2); ax_c.grid(True, alpha=0.3)

    sub_w  = df[df["threshold"] == CHOSEN_T].groupby("window")["RMSE_Fusion"].mean()
    sub_rr = df[df["threshold"] == CHOSEN_T].groupby("window")["filter_replace_rate"].mean()
    ax_db = ax_d.twinx()
    ax_db.bar(sub_rr.index, sub_rr.values, width=1.4, color="salmon", alpha=0.4)
    ax_db.set_ylabel("Replace rate (%)", color="salmon", fontsize=8)
    ax_db.tick_params(axis="y", colors="salmon", labelsize=7)
    ax_d.plot(sub_w.index, sub_w.values, "b-o", linewidth=2, markersize=4, zorder=3)
    ax_d.axhline(ref_ble, color="dimgray", linestyle="--", linewidth=1)
    ax_d.axhline(ref_mmw, color="black",   linestyle=":",  linewidth=1)
    ax_d.axvline(CHOSEN_W, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax_d.set_xlabel("Window size $W$", fontsize=9)
    ax_d.set_ylabel(r"Mean RMSE$_F$ (m)", color="blue", fontsize=9)
    ax_d.tick_params(axis="y", colors="blue", labelsize=7)
    ax_d.set_title(f"(d) RMSE$_F$ vs. Window (τ={CHOSEN_T} m)", fontsize=10, fontweight="bold")
    ax_d.set_xticks(windows); ax_d.grid(True, alpha=0.3)

    fig_all.suptitle("SWMF Parameter Study – BLE AoA + mmWave Fusion",
                     fontsize=12, fontweight="bold")
    _save(fig_all, "SWMF_analysis")


if __name__ == "__main__":
    main()
