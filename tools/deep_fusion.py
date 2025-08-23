# dl_fusion_regressor.py
import ast
import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error

# -------------------------
# Dataset
# -------------------------
class SensorFusionDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, sep=';')
        # Drop rows with missing BLE or mmWave (same as you did)
        df = df[~np.isnan(df[['x_ble']]).any(axis=1)].reset_index(drop=True)

        # parse fields (safer than eval)
        df['centroid_xyz']   = df['centroid_xyz'].apply(eval)
        df['ble_xyz_filter'] = df['ble_xyz_filter'].apply(eval)
        df['real_xyz']       = df['real_xyz'].apply(eval)

        # features: mm x,y,z and ble x,y,z
        X = []
        Y = []
        for _, r in df.iterrows():
            mm = r['centroid_xyz']
            ble = r['ble_xyz_filter']
            gt = r['real_xyz']
            # skip if nan
            if any([np.isnan(v) for v in mm[:2]]) or any([np.isnan(v) for v in ble[:2]]):
                continue
            feat = np.array([mm[0], mm[1], mm[2], ble[0], ble[1], ble[2]], dtype=np.float32)
            target = np.array([gt[0], gt[1]], dtype=np.float32)
            X.append(feat)
            Y.append(target)

        self.X = np.vstack(X).astype(np.float32)
        self.Y = np.vstack(Y).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# -------------------------
# Model: MLP that predicts mean and log-variance
# -------------------------
class FusionMLP(nn.Module):
    def __init__(self, in_dim=6, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU()
        )
        # outputs: mu_x, mu_y, logvar_x, logvar_y
        self.head = nn.Linear(hidden//2, 4)

    def forward(self, x):
        h = self.net(x)
        out = self.head(h)
        mu = out[:, :2]
        logvar = out[:, 2:]
        return mu, logvar

# -------------------------
# Loss: heteroscedastic Gaussian NLL
# -------------------------
def nll_loss(mu, logvar, y):
    # logvar shape (B,2)
    inv_var = torch.exp(-logvar)
    diff2 = (y - mu) ** 2
    loss_per_dim = 0.5 * (inv_var * diff2 + logvar)  # 0.5*( (y-mu)^2/sigma^2 + log sigma^2 )
    return loss_per_dim.mean()

# -------------------------
# Training loop
# -------------------------
def train_model(csv_path, out_dir='dl_model_out', device='cpu',
                hidden=128, batch_size=64, lr=1e-3, epochs=100, val_split=0.1):
    ds = SensorFusionDataset(csv_path)
    N = len(ds)
    if N == 0:
        raise RuntimeError("No usable rows in dataset after filtering.")
    n_val = max(1, int(val_split * N))
    n_train = N - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = FusionMLP(in_dim=6, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float('inf')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs+1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mu, logvar = model(xb)
            loss = nll_loss(mu, logvar, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        preds = []
        gts = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                mu, logvar = model(xb)
                loss = nll_loss(mu, logvar, yb)
                val_losses.append(loss.item())
                preds.append(mu.cpu().numpy())
                gts.append(yb.cpu().numpy())
        mean_val = float(np.mean(val_losses)) if len(val_losses) else float('inf')

        # compute RMSE on val
        if preds:
            preds = np.vstack(preds)
            gts = np.vstack(gts)
            rmse = np.sqrt(mean_squared_error(gts, preds))
        else:
            rmse = np.nan

        print(f"Ep {ep:03d} TrainLoss {np.mean(train_losses):.4f} ValLoss {mean_val:.4f} ValRMSE {rmse:.4f}")

        # save best
        if mean_val < best_val:
            best_val = mean_val
            torch.save({'model_state': model.state_dict(), 'hidden': hidden}, out_dir / 'best.pth')
    # final save
    torch.save({'model_state': model.state_dict(), 'hidden': hidden}, out_dir / 'last.pth')
    print("Saved models to", out_dir)
    return out_dir / 'best.pth', ds

# -------------------------
# Inference and write results
# -------------------------
def predict_and_save(csv_path, model_path, out_csv='dl_fused_output.csv', device='cpu'):
    ds = SensorFusionDataset(csv_path)
    all_X = torch.tensor(ds.X, dtype=torch.float32).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model = FusionMLP(in_dim=6, hidden=ckpt.get('hidden', 128)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    with torch.no_grad():
        mu, logvar = model(all_X)
    mu = mu.cpu().numpy()
    logvar = logvar.cpu().numpy()
    # Read original df and add results in order (we filtered rows when building dataset)
    df = pd.read_csv(csv_path, sep=';')
    df = df[~np.isnan(df[['x_ble']]).any(axis=1)].reset_index(drop=True)
    df['centroid_xyz']   = df['centroid_xyz'].apply(eval)
    df['ble_xyz_filter'] = df['ble_xyz_filter'].apply(eval)
    df['real_xyz']       = df['real_xyz'].apply(eval)

    # build result columns aligned to the filtered rows used in dataset
    res_idx = []
    fused_xy = []
    fused_unc = []
    j = 0
    for i, r in df.iterrows():
        mm = r['centroid_xyz']
        ble = r['ble_xyz_filter']
        if any([np.isnan(v) for v in mm[:2]]) or any([np.isnan(v) for v in ble[:2]]):
            fused_xy.append([np.nan, np.nan])
            fused_unc.append([np.nan, np.nan])
            continue
        fused_xy.append([float(mu[j,0]), float(mu[j,1])])
        fused_unc.append([float(np.exp(logvar[j,0])), float(np.exp(logvar[j,1]))])  # variance
        j += 1

    df['dl_fused_xy'] = fused_xy
    df['dl_fused_var_xy'] = fused_unc
    df['sensor_fused_xyz'] = df['dl_fused_xy'].apply(lambda p: [p[0], p[1], 1.78] if not any(np.isnan(p)) else [np.nan,np.nan,np.nan])
    df.to_csv(out_csv, sep=';', index=False)
    print("Wrote:", out_csv)
    return df

# -------------------------
# CLI-ish usage
# -------------------------
if __name__ == "__main__":
    CSV = "FUSAO_PROCESSADA.csv"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_model, ds = train_model(CSV, out_dir='dl_out', device=device,
                                 hidden=128, batch_size=64, lr=1e-3, epochs=80, val_split=0.12)
    predict_and_save(CSV, "dl_out\\best.pth", out_csv='dl_fused_dataset.csv', device=device)
