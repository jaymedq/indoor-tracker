import matplotlib.pyplot as plt
import numpy as np

# Dados de exemplo controlados (Dois pontos para suportar o conceito de RMSE)
ground_truth_x = [4.0, 5.3]
ground_truth_y = [2.0, 1.7]
estimated_x = [4.5, 5.6]
estimated_y = [2.5, 2.1]

# Formato wide e menor
plt.figure(figsize=(7, 2.5))

# Plotar os pontos
plt.scatter(ground_truth_x, ground_truth_y, c='black', label=r'Ponto Real $(x_{\mathrm{true},i}, y_{\mathrm{true},i})$', s=100, zorder=3)
plt.scatter(estimated_x, estimated_y, c='red', marker='x', label=r'Ponto Estimado $(\hat{x}_{\mathrm{f},i}, \hat{y}_{\mathrm{f},i})$', s=100, zorder=3)

# Desenhar a linha de erro (Distância Euclidiana)
for i in range(len(ground_truth_x)):
    plt.plot([ground_truth_x[i], estimated_x[i]], 
             [ground_truth_y[i], estimated_y[i]], 
             'k--', alpha=0.5)
    # Adicionar anotação de erro (e_1, e_2)
    mid_x = (ground_truth_x[i] + estimated_x[i]) / 2
    mid_y = (ground_truth_y[i] + estimated_y[i]) / 2
    distance = np.sqrt((estimated_x[i] - ground_truth_x[i])**2 + (estimated_y[i] - ground_truth_y[i])**2)
    plt.text(mid_x + 0.2, mid_y - 0.1, f'$e_{i+1}={distance:.1f}$', fontsize=14, color='gray', ha='center', va='bottom')

plt.xlabel('X [m]', fontsize=11)
plt.ylabel('Y [m]', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=10)

# Adicionar a resolução dos cálculos no top right
calc_text = (
    r"$e_1^2 = (4.5 - 4.0)^2 + (2.5 - 2.0)^2 = 0.50$" + "\n" +
    r"$e_2^2 = (5.6 - 5.3)^2 + (2.1 - 1.7)^2 = 0.25$" + "\n" +
    r"$\mathrm{RMSE} = \sqrt{\left(\frac{0.50 + 0.25}{2}\right)} \approx 0.61\mathrm{m}$"
)
plt.text(0.98, 0.95, calc_text, transform=plt.gca().transAxes, fontsize=10, 
         ha='right', va='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='lightgray'))

# Focar o plot nos dois casos com respiro
plt.xlim(3.5, 6.5)
plt.ylim(1.2, 3.2)

plt.tight_layout()
plt.savefig('rmse_conceito_diagrama.png', dpi=300, transparent=True)
plt.show()