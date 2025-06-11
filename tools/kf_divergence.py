import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
import pandas as pd

# Load the data
df = pd.read_csv("fused_dataset.csv", sep=';')
df['sensor_fused_xyz'] = df['sensor_fused_xyz'].apply(eval)
df['ble_xyz'] = df['ble_xyz'].apply(eval)
df['centroid_xyz'] = df['centroid_xyz'].apply(eval)
df['real_xyz'] = df['real_xyz'].apply(eval)

# Radar origin
radar_placement = np.array([0.995, -7.825, 1.70])

# Calculate offsets
def calculate_offset_mmw(row):
    return np.linalg.norm(np.array(row['centroid_xyz']) - radar_placement)

def calculate_offset_ble(row):
    return np.linalg.norm(np.array(row['ble_xyz']) - radar_placement)

def calculate_offset_fusion(row):
    return np.linalg.norm(np.array(row['sensor_fused_xyz']) - radar_placement)

df['offset_mmw'] = df.apply(calculate_offset_mmw, axis=1)
df['offset_ble'] = df.apply(calculate_offset_ble, axis=1)
df['offset_fusion'] = df.apply(calculate_offset_fusion, axis=1)

# Group by distance
groups = list(df.groupby('distance'))
n_groups = len(groups)

# Prepare for storing metrics
distance_labels = []
entropy_f = []
entropy_b = []
entropy_w = []

kl_f_b = []
kl_f_w = []
kl_b_w = []

# Determine subplot grid size
ncols = 2
nrows = (n_groups + 1) // ncols

# Create subplots for histograms
fig, axes = plt.subplots(nrows, ncols)
axes = axes.flatten()

for idx, (distance, group_df) in enumerate(groups):
    fused_offset = group_df['offset_fusion']
    ble_offset = group_df['offset_ble']
    mmwave_offset = group_df['offset_mmw']

    # Compute probability distributions (normalize histograms)
    f_hist, _ = np.histogram(fused_offset, bins=100, density=True)
    b_hist, _ = np.histogram(ble_offset, bins=100, density=True)
    w_hist, _ = np.histogram(mmwave_offset, bins=100, density=True)

    f_prob = f_hist / np.sum(f_hist)
    b_prob = b_hist / np.sum(b_hist)
    w_prob = w_hist / np.sum(w_hist)

    # Avoid log(0)
    epsilon = 1e-10
    f_prob += epsilon
    b_prob += epsilon
    w_prob += epsilon

    # Entropy and Cross-Entropy
    H_f = entropy(f_prob, base=2)
    H_b = entropy(b_prob, base=2)
    H_w = entropy(w_prob, base=2)

    H_f_b = entropy(f_prob, b_prob, base=2)
    H_f_w = entropy(f_prob, w_prob, base=2)
    H_b_w = entropy(w_prob, b_prob, base=2)

    # KL Divergences
    DKL_f_b = H_f_b - H_f
    DKL_f_w = H_f_w - H_f
    DKL_b_w = H_b_w - H_w

    print("\nDistance:", distance)
    print(f"Entropy H(F): {H_f:.4f}")
    print(f"Cross-Entropy H(F, B): {H_f_b:.4f}")
    print(f"Cross-Entropy H(F, W): {H_f_w:.4f}")
    print(f"Cross-Entropy H(B, W): {H_b_w:.4f}")
    print(f"KL Divergence DKL(F‖B): {DKL_f_b:.4f}")
    print(f"KL Divergence DKL(F‖W): {DKL_f_w:.4f}")
    print(f"KL Divergence DKL(B‖W): {DKL_b_w:.4f}")

    # Store metrics
    distance_labels.append(distance)
    entropy_f.append(H_f)
    entropy_b.append(H_b)
    entropy_w.append(H_w)
    kl_f_b.append(DKL_f_b)
    kl_f_w.append(DKL_f_w)
    kl_b_w.append(DKL_b_w)

    # Plot histograms
    axes[idx].hist(fused_offset, bins=100, alpha=0.5, label='Fused', color='red')
    axes[idx].hist(ble_offset, bins=100, alpha=0.5, label='BLE', color='orange')
    axes[idx].hist(mmwave_offset, bins=100, alpha=0.5, label='mmWave', color='green')
    axes[idx].set_title(f"Distance = {distance}")
    axes[idx].set_xlabel("Offset (m)")
    axes[idx].set_ylabel("Frequency")
    axes[idx].legend()
    axes[idx].grid(True)

# Remove unused axes
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# === New Figure: Entropies per Distance ===
plt.figure(figsize=(10, 6))
plt.plot(distance_labels, entropy_f, marker='s', label='H(F) - Fused', color='red')
plt.plot(distance_labels, entropy_b, marker='o', label='H(B) - BLE', color='orange')
plt.plot(distance_labels, entropy_w, marker='^', label='H(W) - mmWave', color='green')
plt.xlabel("Distance")
plt.ylabel("Entropy")
plt.title("Entropy per Distance Group")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === New Figure: KL Divergences per Distance ===
plt.figure(figsize=(10, 6))
plt.plot(distance_labels, kl_f_b, marker='o', label='DKL(F‖B)', color = 'orange')
plt.plot(distance_labels, kl_f_w, marker='s', label='DKL(F‖W)', color = 'green')
plt.plot(distance_labels, kl_b_w, marker='^', label='DKL(B‖W)', color = 'red')
plt.xlabel("Distance")
plt.ylabel("KL Divergence")
plt.title("KL Divergences per Distance Group")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
