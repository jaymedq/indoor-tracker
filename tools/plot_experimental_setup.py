import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- DEFINIÇÃO DAS COORDENADAS E PARÂMETROS ---
# [x, y, z]
A1 = [0.995, -7.825, 2.417]
A2 = [0.99, -1.206, 2.416]
A3 = [5.717, -7.846, 2.41]
A4 = [3.524,  -4.629, 2.416]
R1 = [0.98, -4.5, 1.78]  # Radar mmWave
TAG = [5.9, -6.865, 1.8] # Tag do usuário (Alvo)

# Lista de obstáculos (exemplo aproximado das bancadas/mesas do laboratório)
OBSTACLES = [
    {"name": "Bancada", "x": 0.3, "y": -2.0, "w": 6.5, "h": -0.8},  # bancada da frente
    {"name": "Bancada", "x": 0.3, "y": -5.25, "w": 6.5, "h": -0.8},  # bancada do meio
    {"name": "Armários", "x": 2.0, "y": -9.5, "w": 5.0, "h": -0.5},  # armarios
    {"name": "Mesa", "x": 1.0, "y": -8.8, "w": 1.0, "h": -1.2},  # bancada glauber
    {"name": "Sala de Reuniões", "x": 8.0, "y": -2.0, "w": 2.0, "h": -4.0},  # sala de reunioes
]

# Parâmetros das Antenas Planares (Grids)
ble_anchors = np.array([A1, A2, A3, A4])
grid_size = 0.5  # Tamanho total do grid da antena (1m x 1m)
grid_elements = 5 # Número de elementos em cada direção

# Configurações do Radar
radar_fov_deg = 120
radar_radius = 12.0

# --- CONFIGURAÇÃO DA FIGURA ---
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 1. Chão (Área de cobertura) - plano XY translúcido
x_plane = np.linspace(0, 10)
y_plane = np.linspace(-10, 0)
X, Y = np.meshgrid(x_plane, y_plane)
Z = np.zeros_like(X)
ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')
# ax.text(2, 2, 0, 'Área de Cobertura de Teste', color='gray', fontsize=10, style='italic')

# 2. Desenhar as 4 Âncoras BLE como Arrays Planares (Grids)
# Dummy plot para forçar o símbolo representativo (grid azul) na legenda
ax.plot([], [], marker='s', linestyle='None', markerfacecolor='none', markeredgecolor='blue', markersize=10, markeredgewidth=1.5, label='Âncora BLE')

for i, pos in enumerate(ble_anchors):
    ax.scatter([pos[0]], [pos[1]], [pos[2]], color='black', marker='o', s=10)
    # Criar o Grid (Patch Plano)
    elements_x = np.linspace(pos[0] - grid_size/2, pos[0] + grid_size/2, grid_elements)
    elements_y = np.linspace(pos[1] - grid_size/2, pos[1] + grid_size/2, grid_elements)
    grid_coords = []
    
    # Desenhar linhas do grid
    for x in elements_x:
        ax.plot([x, x], [elements_y[0], elements_y[-1]], [pos[2], pos[2]], color='blue', linewidth=0.8)
    for y in elements_y:
        ax.plot([elements_x[0], elements_x[-1]], [y, y], [pos[2], pos[2]], color='blue', linewidth=0.8)

# 3. Adicionar Diagramas de Azimute e Elevação (Conforme Referência)
# Vamos adicionar os diagramas geométricos ligando cada âncora à tag.

def draw_angle_geometry(ax, pos, color, tag_pos):
    """Adiciona as linhas e arcos de azimute/elevação da âncora para a tag."""
    x, y, z = pos
    tx, ty, tz = tag_pos
    
    # Eixos locais: apenas para referência visual da âncora
    ax.plot([x, x+1], [y, y], [z, z], color='gray', alpha=0.5, linestyle='-') # Eixo Local X
    ax.plot([x, x], [y, y+1], [z, z], color='gray', alpha=0.5, linestyle='-') # Eixo Local Y
    
    # Projeção da linha de visão no plano horizontal da âncora (Z = z)
    # ax.plot([x, tx], [y, ty], [z, z], color=color, alpha=0.8, linestyle='--') 

    # Linha de Visão Direta (LoS) até a tag
    line_to_tag = np.array([np.linspace(x, tx, 50), np.linspace(y, ty, 50), np.linspace(z, tz, 50)])
    ax.plot(line_to_tag[0], line_to_tag[1], line_to_tag[2], color='blue', alpha=0.5, linestyle=':')

    # Rótulos dos ângulos
    # dx = tx - x
    # dy = ty - y
    # dz = tz - z
    
    # Rótulo de azimute (ângulo no plano xy)
    # ax.text(x + 0.3*dx, y + 0.3*dy, z, r'$\phi$ (Azimute)', color='blue', fontsize=10)
    
    # Rótulo de elevação (ângulo vertical)
    # ax.text(x + 0.3*dx, y + 0.3*dy, z + 0.3*dz - 0.2, r'$\theta$ (Elevação)', color='red', fontsize=10)

for pos in ble_anchors:
    draw_angle_geometry(ax, pos, color='black', tag_pos=TAG)

# Desenhar as Bancadas e Obstáculos (Altura de 0.8m)
for obs in OBSTACLES:
    if obs["name"] == "Meeting Room":
        continue  # Não desenhar a sala de reunião como bancada
    
    o_dx = obs["w"]
    o_dy = abs(obs["h"])
    o_x = obs["x"]
    o_y = obs["y"] if obs["h"] > 0 else obs["y"] + obs["h"]
    o_dz = 0.8  # Altura de 0.8m conforme pedido
    
    # Caixa 3D para a bancada
    ax.bar3d(o_x, o_y, 0, o_dx, o_dy, o_dz, color='lightgray', alpha=0.4, edgecolor='darkgray')
    
    # Texto em cima da bancada
    ax.text(o_x + o_dx/2, o_y + o_dy/2, o_dz + 0.1, obs["name"], color='dimgray', fontsize=8, ha='center')

# Representação da pessoa (Caixa de 0.5 x 0.5 x 1.78m)
p_dx, p_dy, p_dz = 0.7, 0.7, 1.70
p_x, p_y, p_z = TAG[0] - p_dx/2, TAG[1] - p_dy/2, 0 # Base no chão, centro x,y alinhado com a tag

# Usar bar3d para desenhar a caixa translúcida
ax.bar3d(p_x, p_y, p_z, p_dx, p_dy, p_dz, color='orange', alpha=0.3, edgecolor='darkorange', label='Humano')

# Adicionar a Tag no Plot
ax.scatter([TAG[0]], [TAG[1]], [TAG[2]], color='green', marker='^', s=100, label='Tag BLE', zorder = 3)
ax.text(TAG[0], TAG[1], TAG[2] + 0.2, 'Tag', color='green', fontweight='bold', fontsize=11)

# 4. Radar mmWave e seu FoV
# Plano vermelho para representar o radar (plano vertical)
r_size = 0.3
radar_verts = [[
    (R1[0] - 0.05, R1[1] - r_size/2, R1[2] - r_size/2),
    (R1[0] - 0.05, R1[1] + r_size/2, R1[2] - r_size/2),
    (R1[0] - 0.05, R1[1] + r_size/2, R1[2] + r_size/2),
    (R1[0] - 0.05, R1[1] - r_size/2, R1[2] + r_size/2)
]]
radar_plane = Poly3DCollection(radar_verts, facecolors='red', edgecolors='darkred', alpha=0.8, zorder=1)
ax.add_collection3d(radar_plane)

# Dummy plot para forçar o retângulo vermelho na legenda, removendo o marcador 'x'
ax.plot([], [], 's', color='red', markersize=10, markeredgecolor='darkred', label='Radar mmWave')
ax.plot([R1[0], R1[0]], [R1[1], R1[1]], [0, R1[2]], 'k:', alpha=0.4) # Projeção vertical

# Desenhar o FoV do Radar (Cone ciano)
theta_center = np.deg2rad(0) # Apontando para o centro
theta_half = np.deg2rad(radar_fov_deg / 2)
ts = np.linspace(theta_center - theta_half, theta_center + theta_half, 60)
for t in (theta_center - theta_half, theta_center + theta_half):
    ax.plot([R1[0], R1[0] + radar_radius * np.cos(t)], [R1[1], R1[1] + radar_radius * np.sin(t)], [R1[2], R1[2]], color='cyan', linestyle='--')
xs = R1[0] + radar_radius * np.cos(ts)
ys = R1[1] + radar_radius * np.sin(ts)
zs = np.full_like(xs, R1[2])
ax.plot(xs, ys, zs, color='cyan', linestyle='--', label='Radar FoV')

# 5. Estética, Eixos e Legenda
for i, a in enumerate(ble_anchors):
    ax.text(a[0], a[1], a[2]+0.2, f'A{i+1}', color='blue', fontweight='bold', fontsize=11)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
# ax.set_title('Topologia Híbrida Experimental: BLE AoA + mmWave Radar')
ax.set_xlim([0.0, 10.0])
ax.set_ylim([-10.0, 0.0])
ax.set_zlim([0, 4.0])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)

# Ajuste do ângulo de visão para mostrar os grids e ângulos
ax.view_init(elev=22, azim=-55)

# Tornar os planos de fundo do eixo 3D completamente transparentes
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

plt.tight_layout()
plt.savefig("topologia_3d_detalhada.png", dpi=300, transparent=True) # Salvar sem fundo
plt.show()