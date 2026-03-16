import matplotlib as mpl
import matplotlib.pyplot as plt

# --- PGF CONFIGURATION ---
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",     # Matches LaTeX default
    "text.usetex": True,        # Let LaTeX handle the rendering
    "pgf.rcfonts": False,       # Ignore Matplotlib's internal fonts
})

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

from matplotlib.patches import Wedge
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from constants import RADAR_PLACEMENT, EXPERIMENT_POINTS, RADAR_FACING_ANGLE, RADAR_FOV, FOV_RADIUS


# Lista de obstáculos (exemplo aproximado das bancadas/mesas do laboratório)
OBSTACLES = [
    {"name": "Bench", "x": 0.3, "y": -2.0, "w": 6.5, "h": -0.8},  # bancada da frente
    {"name": "Bench", "x": 0.3, "y": -5.25, "w": 6.5, "h": -0.8},  # bancada do meio
    {"name": "Cabinets", "x": 2.0, "y": -9.5, "w": 5.0, "h": -0.5},  # armarios
    {"name": "Desk", "x": 1.0, "y": -8.8, "w": 1.0, "h": -1.2},  # bancada glauber
    {
        "name": "Meeting Room",
        "x": 8.0,
        "y": -2.0,
        "w": 2.0,
        "h": -4.0,
    },  # sala de reunioes
]

POINTS_TO_CONSIDER = ["C3P2", "C3P3", "C3P4", "C3P5", "C2P2", "C2P3", "C2P4", "C2P5"]
ANCHORS_TO_CONSIDER = ["ANCHOR1", "ANCHOR2", "ANCHOR3", "ANCHOR4"]

def plot_obstacles(ax):
    """Desenha os obstáculos fixos (mesas e bancadas)."""
    for obs in OBSTACLES:
        rect = Rectangle(
            (obs["x"], obs["y"]),  # canto superior esquerdo
            obs["w"],  # largura em x
            obs["h"],  # altura em y (negativa para baixo)
            linewidth=1,
            edgecolor="brown",
            facecolor="sandybrown",
            alpha=0.3,
        )
        ax.add_patch(rect)
        ax.text(
            obs["x"] + obs["w"] / 2,
            obs["y"] + obs["h"] / 2,
            obs["name"],
            color="brown",
            fontsize=9,
            ha="center",
            va="center",
        )

def plot_colored_points(ax, x, y, times, cmap, label, size=10, marker="."):
    """Plot points with fading colors (no connecting lines)."""
    if len(x) < 1:
        return
    if label == "Real":
        ax.scatter(x, y, marker=marker, c="black", s=size, label=label)
    else:
        norm = Normalize(vmin=times.min() - 1200, vmax=times.max())
        ax.scatter(
            x,
            y,
            marker=marker,
            c=times,
            cmap=cmap,
            norm=norm,
            s=size,
            alpha=0.8,
            label=label,
        )


def plot_radar_fov(ax, plot_radar_point_value:bool = False):
    """Plot radar FOV sector and radar position."""
    theta1 = RADAR_FACING_ANGLE - RADAR_FOV / 2
    theta2 = RADAR_FACING_ANGLE + RADAR_FOV / 2

    wedge = Wedge(
        center=(RADAR_PLACEMENT[0], RADAR_PLACEMENT[1]),
        r=FOV_RADIUS,
        theta1=theta1,
        theta2=theta2,
        facecolor="cyan",
        alpha=0.1,
        edgecolor="cyan",
        linestyle="--",
    )
    ax.add_patch(wedge)
    ax.scatter(
        RADAR_PLACEMENT[0],
        RADAR_PLACEMENT[1],
        c="k",
        marker="x",
        s=80,
        label="Radar Position",
    )
    if plot_radar_point_value:
        ha_val = "right"
        va_val = "bottom"
        ax.text(RADAR_PLACEMENT[0]+0.1, RADAR_PLACEMENT[1]-0.6, "R", weight="bold", fontsize=10, ha=ha_val, va=va_val)
        ax.text(RADAR_PLACEMENT[0]+1.85, RADAR_PLACEMENT[1]-0.6, f"{RADAR_PLACEMENT[:2]}", fontsize=10, ha=ha_val, va=va_val)


def plot_experiment_points(ax, plot_anchor_position=False):
    anchor_x = []
    anchor_y = []
    anchopr_labels = []
    for label, coords in EXPERIMENT_POINTS.items():
        if not (label in POINTS_TO_CONSIDER or label in ANCHORS_TO_CONSIDER) or "V" in label:
            continue
        elif "ANCHOR" in label and label in ANCHORS_TO_CONSIDER:
            offset_x = 0.3
            offset_y = -0.5
            anchor_x.append(coords[0])
            anchor_y.append(coords[1])
            anchopr_labels.append(label)
            ax.text(
                coords[0] + offset_x,
                coords[1] + offset_y,
                label.replace("ANCHOR", "A"),
                fontsize=10,
                weight="bold",
                ha="right",
                va="bottom",
                color="r"
            )
            if plot_anchor_position:
                ax.text(coords[0]+offset_x, coords[1]+offset_y, f"{coords[:2]}", fontsize=10, color="r")
        else:
            # Adjust label position to avoid overlap
            offset_x = 0
            offset_y = 0.1
            ha_val = "center"
            va_val = "bottom"
            
            if "C2" in label or "C3" in label:
                va_val = "top"
                offset_y = 0.40
            else:
                ha_val = "right"
                va_val = "bottom"
                offset_x = -0.1
                offset_y = 0.1

            ax.text(
                coords[0] + offset_x,
                coords[1] + offset_y,
                label.replace("C","H") if label.startswith("C") else label,
                fontsize=8,
                weight="bold",
                ha=ha_val,
                va=va_val,
                color="dimgray"
            )
            ax.scatter(coords[0], coords[1], c="gray", marker=".", s=20)
    ax.scatter(anchor_x, anchor_y, c="r", marker="v", s=20, label="BLE Anchor Position")

if __name__ == "__main__":
    # Plot static routes
    fig, ax = plt.subplots(figsize=get_size(345, aspect_ratio=1.0))

    plot_radar_fov(ax, True)
    plot_obstacles(ax)
    plot_experiment_points(ax, True)

    # Axis setup
    ax.set_xlim([0, 10])
    ax.set_ylim([-10, 0])
    ax.set_aspect('equal')
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=".",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="Annotated Points Position",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="<",
            color="w",
            markerfacecolor="cyan",
            markersize=8,
            label="Radar FOV",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="black",
            markersize=8,
            label="Radar Position",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="r",
            markersize=8,
            label="BLE Anchor Position",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9)

    ax.set_xlabel(r"X [m]")
    ax.set_ylabel(r"Y [m]")
    fig.tight_layout()
    
    fig.savefig("labsc_2d_map.pgf", backend='pgf', bbox_inches='tight')
    fig.savefig("labsc_2d_map.pdf", bbox_inches='tight')
    print("Saved labsc_2d_map.pgf and .pdf")
