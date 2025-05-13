import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
import pandas as pd
import os

# Load the data
df = pd.read_csv("fused_dataset.csv", sep=';')
df['sensor_fused_xyz'] = df['sensor_fused_xyz'].apply(eval)
df['ble_xyz'] = df['ble_xyz'].apply(eval)
df['centroid_xyz'] = df['centroid_xyz'].apply(eval)
df['real_xyz'] = df['real_xyz'].apply(eval)


# Calculate offsets
def calculate_offset_mmw(row):
    return np.linalg.norm(np.array(row['centroid_xyz'])-np.array(row["real_xyz"]))

def calculate_offset_ble(row):
    return np.linalg.norm(np.array(row['ble_xyz'])-np.array(row["real_xyz"]))

def calculate_offset_fusion(row):
    return np.linalg.norm(np.array(row['sensor_fused_xyz'])-np.array(row["real_xyz"]))

df['offset_mmw'] = df.apply(calculate_offset_mmw, axis=1)
df['offset_ble'] = df.apply(calculate_offset_ble, axis=1)
df['offset_fusion'] = df.apply(calculate_offset_fusion, axis=1)


# Flatten to 1D for histogram-based DKL (you could also use 2D histograms)
fused_offset = df['offset_fusion']
ble_offset = df['offset_ble']
mmwave_offset = df['offset_mmw']

# Step 2: Histogram bins (shared across all sources)
bins = np.histogram_bin_edges(fused_offset, bins=30)  # You can tune the bin count

# Step 3: Compute probability distributions (normalize histograms)
f_hist, _ = np.histogram(fused_offset, bins=bins)
b_hist, _ = np.histogram(ble_offset, bins=bins)
w_hist, _ = np.histogram(mmwave_offset, bins=bins)

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

DKL_f_b = H_f_b - H_f
DKL_f_w = H_f_w - H_f

print(f"Entropy H(F): {H_f:.4f}")
print(f"Cross-Entropy H(F, B): {H_f_b:.4f}")
print(f"Cross-Entropy H(F, W): {H_f_w:.4f}")
print(f"KL Divergence DKL(F‖B): {DKL_f_b:.4f}")
print(f"KL Divergence DKL(F‖W): {DKL_f_w:.4f}")

# Step 5: Plot Histograms
plt.figure(figsize=(10, 6))
plt.hist(fused_offset, bins=bins, alpha=0.5, label='Fused offset')
plt.hist(ble_offset, bins=bins, alpha=0.5, label='BLE offset')
plt.hist(mmwave_offset, bins=bins, alpha=0.5, label='mmWave offset')
plt.legend()
plt.title("Histogram Comparison (distance from real point)")
plt.xlabel("Offset (m)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
