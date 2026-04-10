import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from constants import RADAR_PLACEMENT
from plot_room_2d import plot_obstacles, plot_radar_fov

def draw_room_base(ax):
    """Desenha a estrutura básica da sala baseada no cenário real."""
    plot_obstacles(ax)
    
    # Draw room boundary
    room = patches.Rectangle((0, -10), 10, 10, linewidth=2, edgecolor='black', facecolor='none', zorder=1)
    ax.add_patch(room)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(-10, 0)
    ax.set_aspect('equal')
    ax.axis('off')

def draw_ble_scenario():
    """Desenha o cenário apenas com sensor BLE (AoA)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_room_base(ax)
    ax.set_title("AMBIENTE APENAS COM SENSOR BLE (AoA)", fontsize=12, fontweight='bold', pad=10)
    
    # Plot single anchor
    anchor_pos = [3.524, -4.629] # A4 position
    ax.scatter(anchor_pos[0], anchor_pos[1], c="r", marker="v", s=50, zorder=4)
    ax.text(anchor_pos[0] + 0.2, anchor_pos[1] - 0.2, "ANCORA", fontsize=9, fontweight='bold', color='red')
    
    # Target point 
    target_pos = (7.1, -3.215)
    
    # Direct path from A4
    ax.annotate("", xy=target_pos, xytext=anchor_pos, arrowprops=dict(arrowstyle="->", color='blue', alpha=0.5, connectionstyle="arc3,rad=-0.1", linewidth=1.5))
    
    # Multipath 1 -> Bounces on the Meeting Room walls 
    bounce1 = (8, -5)
    ax.annotate("", xy=bounce1, xytext=anchor_pos, arrowprops=dict(arrowstyle="-", color='gray', alpha=0.5, linestyle='--'))
    ax.annotate("", xy=target_pos, xytext=bounce1, arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5, linestyle='--'))

    # Multipath 2 -> Bounces on the other Bench
    bounce2 = (4.0, -2.5)
    ax.annotate("", xy=bounce2, xytext=anchor_pos, arrowprops=dict(arrowstyle="-", color='gray', alpha=0.5, linestyle='--'))
    ax.annotate("", xy=target_pos, xytext=bounce2, arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5, linestyle='--'))

    ax.text(5, -10, "Sensível a propagação multipercurso / interferência", fontsize=11, fontweight='bold', color='dimgray', ha='center', va='top', clip_on=False)
    
    # Error cloud
    np.random.seed(42)
    noise_x = np.random.normal(target_pos[0], 0.7, 100)
    noise_y = np.random.normal(target_pos[1], 0.7, 100)
    ax.scatter(noise_x, noise_y, color='gray', alpha=0.3, s=10, zorder=3)
    ax.scatter(target_pos[0], target_pos[1], color='black', s=50, zorder=4) 
    
    # ax.text(7, -1, "Alta Variância /\nErro Estocástico", fontsize=9, color='gray', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    fig.savefig('sensors_weakness_ble.png', transparent=True, bbox_inches='tight')
    plt.close(fig)
    print("BLE Figure saved to sensors_weakness_ble.png")


def draw_radar_scenario():
    """Desenha o cenário apenas com sensor Radar (mmWave)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_room_base(ax)
    ax.set_title("AMBIENTE APENAS COM SENSOR RADAR (mmWave)", fontsize=12, fontweight='bold', pad=10)
    
    # Radar position
    rx, ry = RADAR_PLACEMENT[0], RADAR_PLACEMENT[1]
    
    plot_radar_fov(ax) # Plots the wedge and radar marker
    ax.text(rx - 0.2, ry + 0.2, "Radar", fontsize=10, fontweight='bold', ha='right')
    
    # Draw diverging ellipses inside FOV
    distances = np.linspace(2, 8, 4)
    angle_rad = 0  # straight ahead (along positive x axis)
    
    for d in distances:
        tx = rx + d * np.cos(angle_rad)
        ty = ry + d * np.sin(angle_rad)
        
        # Ellipse dimensions
        radial_error = 0.5  # Fixed/slow increasing error along beam
        transversal_error = 0.2 * d**1.3  # Angular resolution dominates at distance
        
        ellipse = patches.Ellipse((tx, ty), width=radial_error, height=transversal_error, angle=0, linewidth=1.5, edgecolor='gray', facecolor='none', alpha=0.8, zorder=3)
        ax.add_patch(ellipse)
        
        # Generate random gray points inside the ellipse
        np.random.seed(int(d * 100))
        num_points = 50
        noise_x = np.random.normal(tx, radial_error / 4, num_points)
        noise_y = np.random.normal(ty, transversal_error / 4, num_points)
        ax.scatter(noise_x, noise_y, color='gray', alpha=0.4, s=10, zorder=3)
        
        ax.scatter(tx, ty, color='black', s=30, zorder=4)
        
    ax.text(5, -10, "Sensível à distância (resolução angular)", fontsize=11, fontweight='bold', color='dimgray', ha='center', va='top', clip_on=False)
    
    plt.tight_layout()
    fig.savefig('sensors_weakness_radar.png', transparent=True, bbox_inches='tight')
    plt.close(fig)
    print("Radar Figure saved to sensors_weakness_radar.png")

if __name__ == "__main__":
    draw_ble_scenario()
    draw_radar_scenario()
    
    print("\n" + "="*50)
    print("TEXTO PARA O SLIDE - BLE (AoA)")
    print("Cobre grande área geométrica.")
    print("Sensível a ricocheteamento do sinal (multipercurso),")
    print("gerando estimativas de posição ruidosas e instáveis.")
    print("(Var(x) alta; Cov(x,y) alta)")
    print("="*50)
    print("TEXTO PARA O SLIDE - RADAR (mmWave)")
    print("Alta precisão local (curta distância, banda limpa).")
    print("A precisão transversal diminui com a distância (r)")
    print("devido à resolução angular do radar, resultando em")
    print("maior incerteza geométrica ao longo do feixe.")
    print("="*50)