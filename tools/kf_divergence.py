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

# Calculate offsets
def calculate_offset_mmw(row):
    return np.linalg.norm(np.array(row['centroid_xyz']) - np.array(row["real_xyz"]))

def calculate_offset_ble(row):
    return np.linalg.norm(np.array(row['ble_xyz']) - np.array(row["real_xyz"]))

def calculate_offset_fusion(row):
    return np.linalg.norm(np.array(row['sensor_fused_xyz']) - np.array(row["real_xyz"]))

df['offset_mmw'] = df.apply(calculate_offset_mmw, axis=1)
df['offset_ble'] = df.apply(calculate_offset_ble, axis=1)
df['offset_fusion'] = df.apply(calculate_offset_fusion, axis=1)

# Group by distance
groups = list(df.groupby('distance'))
n_groups = len(groups)

# Determine subplot grid size
ncols = 2
nrows = (n_groups + 1) // ncols

# Create subplots
fig, axes = plt.subplots(nrows, ncols)
axes = axes.flatten()

for idx, (distance, group_df) in enumerate(groups):
    fused_offset = group_df['offset_fusion']
    ble_offset = group_df['offset_ble']
    mmwave_offset = group_df['offset_mmw']

    # Step 3: Compute probability distributions (normalize histograms)
    f_hist, _ = np.histogram(fused_offset, bins=100, density=True)
    b_hist, _ = np.histogram(ble_offset, bins=100, density=True)
    w_hist, _ = np.histogram(mmwave_offset, bins=100, density=True)

    f_prob = f_hist / np.sum(f_hist)
    b_prob = b_hist / np.sum(b_hist)
    w_prob = w_hist / np.sum(w_hist)

    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    f_prob += epsilon
    b_prob += epsilon
    w_prob += epsilon

    # Step 4: Compute Entropy and DKL
    H_f = entropy(f_prob, base=2)
    H_f_b = entropy(f_prob, b_prob, base=2)
    H_f_w = entropy(f_prob, w_prob, base=2)

    H_w = entropy(w_prob, base=2)
    H_b_w = entropy(w_prob, b_prob, base=2)

    DKL_f_b = H_f_b - H_f
    DKL_f_w = H_f_w - H_f
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

    # Histograms
    axes[idx].hist(fused_offset, bins=100, alpha=0.5, label='Fused')
    axes[idx].hist(ble_offset, bins=100, alpha=0.5, label='BLE')
    axes[idx].hist(mmwave_offset, bins=100, alpha=0.5, label='mmWave')

    axes[idx].set_title(f"Distance = {distance}")
    axes[idx].set_xlabel("Offset (m)")
    axes[idx].set_ylabel("Frequency")
    axes[idx].legend()
    axes[idx].grid(True)

# Remove any unused axes
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
