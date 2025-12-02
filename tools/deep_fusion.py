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

def safe_point_parse(x):
    """Safely converts CSV input to a list of floats. Handles NaN, None, and empty lists/strings."""
    if pd.isna(x) or x in ('', 'nan', 'NaN'):
        return []
    try:
        # Use ast.literal_eval for list/tuple strings
        val = ast.literal_eval(x)
        if isinstance(val, (list, tuple)):
            # Ensure elements are convertible to float and not NaN if it was a list of NaNs
            return [v for v in val if not pd.isna(v)]
        return []
    except:
        # Fallback for unparseable strings
        return []

class SensorFusionDataset(Dataset):
    """
    Dataset class that prepares two input branches:
    1. MLP Branch: Aggregated mmWave features (9) + Filtered BLE position (3) = 12 features.
    2. CNN Branch: Raw IQ samples (I_1..4, Q_1..4) = 8 features.
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, sep=';')

        point_cloud_cols = ['x', 'y', 'z', 'x_static', 'y_static', 'z_static']
        structured_cols_to_parse = ['ble_xyz_replace_filter', 'real_xyz']
        
        for col in point_cloud_cols + structured_cols_to_parse:
            df[col] = df[col].apply(safe_point_parse) 

        df['x_ble'] = pd.to_numeric(df['x_ble'], errors='coerce')
        df = df[~np.isnan(df['x_ble'])].reset_index(drop=True)

        iq_cols = [f'{p}_{i}' for i in range(4, 0, -1) for p in ['i', 'q']] # i_4..i_1, q_4..q_1
        for col in iq_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Feature Engineering: Two branches
        MLP_X_list = []
        CNN_X_list = []
        Y_list = []
        
        # Cache for carry-forward imputation (stores the last successful 9 mmWave features)
        mm_feature_cache = None 
        # Cache for IQ features (stores the last successful 8 IQ features)
        iq_feature_cache = None 
        
        # mmWave Features: [MeanX, MeanY, MeanZ, MaxX, MaxY, MaxZ, CountDynamic, CountStatic, CentroidDistToRadar] (9 features)
        # BLE Features:    [BLE_x, BLE_y, BLE_z] (3 features)
        
        for _, r in df.iterrows():
            
            ble = r['ble_xyz_replace_filter']
            gt = r['real_xyz']

            if len(ble) < 3: # Must have valid BLE position data
                continue
            
            # MMWave Feature Aggregation (9 features)
            x_dyn, y_dyn, z_dyn = np.array(r['x']), np.array(r['y']), np.array(r['z'])
            x_stat, y_stat, z_stat = np.array(r['x_static']), np.array(r['y_static']), np.array(r['z_static'])
            x_all = np.concatenate([x_dyn, x_stat])
            
            mm_features = []
            
            if len(x_all) > 0:
                y_all = np.concatenate([y_dyn, y_stat])
                z_all = np.concatenate([z_dyn, z_stat])
                mm_features.extend([np.mean(x_all), np.mean(y_all), np.mean(z_all)])
                mm_features.extend([np.max(x_all), np.max(y_all), np.max(z_all)])
                mm_features.extend([len(x_dyn), len(x_stat)])
                mm_features.append(r['distance']) 
                mm_feature_cache = mm_features
            elif mm_feature_cache is not None:
                mm_features = mm_feature_cache
            else:
                mm_features.extend([0.0] * 8 + [r['distance']])

            # BLE IQ Feature Preparation (8 features for CNN)
            iq_data = r[iq_cols].values.astype(np.float32)
            
            if not np.isnan(iq_data).any():
                # IQ data is clean, reshape for CNN (4 antennas x I/Q)
                # For a 1D CNN: (8,) vector
                iq_features = iq_data
                iq_feature_cache = iq_features
            elif iq_feature_cache is not None:
                # Carry-Forward for IQ data
                iq_features = iq_feature_cache
            else:
                # Fill missing IQ with zeros
                iq_features = np.zeros(8, dtype=np.float32)
                
            # --- Final Assembly ---
            mlp_feat = np.array(mm_features + list(ble), dtype=np.float32)
            target = np.array([gt[0], gt[1]], dtype=np.float32)
            
            MLP_X_list.append(mlp_feat)
            CNN_X_list.append(iq_features)
            Y_list.append(target)

        self.MLP_X = np.vstack(MLP_X_list).astype(np.float32)
        # Reshape for 1D CNN: (N, 8, 1) to represent 8 features as a sequence of length 8 with 1 channel
        self.CNN_X = np.vstack(CNN_X_list).astype(np.float32)[:, :, np.newaxis] 
        self.Y = np.vstack(Y_list).astype(np.float32)

    def __len__(self):
        return len(self.MLP_X)

    def __getitem__(self, idx):
        # Return two inputs (for MLP and CNN branches) and the target
        return self.MLP_X[idx], self.CNN_X[idx], self.Y[idx]

# Model: Multi-Branch Deep Fusion Network
class CNNBranch(nn.Module):
    """
    Processes raw BLE IQ samples using a 1D CNN inspired by your Keras example.
    Input shape: (Batch, Sequence Length=8, Channels=1)
    """
    def __init__(self, output_dim=64):
        super().__init__()
        # 1D CNN for sequences/time-series data like IQ samples
        self.conv_net = nn.Sequential(
            # Input: (1, 8) -> Conv (e.g., kernel 3)
            # kernel size 3 will allow the network to see local I/Q patterns
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Output: (32, 4)

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Output: (64, 2)
        )
        # Flatten and Dense layer
        self.head = nn.Sequential(
            nn.Flatten(),
            # 64 channels * 2 length = 128 features before dense layer
            nn.Linear(64 * 2, output_dim), 
            nn.ReLU()
        )

    def forward(self, x):
        # Input x shape: (B, 8, 1). Need to permute to (B, C, L) = (B, 1, 8) for Conv1D
        x = x.permute(0, 2, 1) 
        h = self.conv_net(x)
        return self.head(h)


class MultiBranchFusionNet(nn.Module):
    """
    Fuses features from the CNN Branch (IQ samples) and the MLP Branch (mmWave/BLE positioning).
    """
    def __init__(self, mlp_in_dim=12, cnn_out_dim=64, hidden=128):
        super().__init__()
        
        # Branch 1: IQ Sample Processing (CNN)
        self.cnn_branch = CNNBranch(output_dim=cnn_out_dim)
        
        # Branch 2: Aggregated Feature Processing (MLP)
        self.mlp_branch = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU()
        )
        
        # Fusion and Regression Head
        fusion_dim = cnn_out_dim + (hidden // 2)
        
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU()
        )
        
        # outputs: mu_x, mu_y, logvar_x, logvar_y (4 outputs)
        self.regression_head = nn.Linear(hidden // 2, 4)

    def forward(self, mlp_x, cnn_x):
        # Forward pass through branches
        cnn_features = self.cnn_branch(cnn_x)
        mlp_features = self.mlp_branch(mlp_x)
        
        # Concatenate features (Fusion)
        fused = torch.cat((mlp_features, cnn_features), dim=1)
        
        # Regression Head
        h = self.fusion_net(fused)
        out = self.regression_head(h)
        
        mu = out[:, :2]
        logvar = out[:, 2:]
        return mu, logvar

# Loss: heteroscedastic Gaussian NLL (Unchanged)
def nll_loss(mu, logvar, y):
    # logvar shape (B,2)
    inv_var = torch.exp(-logvar)
    diff2 = (y - mu) ** 2
    loss_per_dim = 0.5 * (inv_var * diff2 + logvar)  # 0.5*( (y-mu)^2/sigma^2 + log sigma^2 )
    return loss_per_dim.mean()

# Training loop (Updated to handle two inputs)
def train_model(csv_path, out_dir='dl_model_out', device='cpu',
                hidden=128, batch_size=64, lr=1e-3, epochs=100, val_split=0.1):
    
    print("Loading Multi-Branch Dataset...")
    ds = SensorFusionDataset(csv_path)
    N = len(ds)
    if N == 0:
        raise RuntimeError("No usable rows in dataset after filtering.")
    n_val = max(1, int(val_split * N))
    n_train = N - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Instantiate the Multi-Branch model
    model = MultiBranchFusionNet(mlp_in_dim=12, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float('inf')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs+1):
        model.train()
        train_losses = []
        # Note: xb is the MLP input, cnn_xb is the CNN input
        for mlp_xb, cnn_xb, yb in train_loader:
            mlp_xb = mlp_xb.to(device)
            cnn_xb = cnn_xb.to(device)
            yb = yb.to(device)
            
            mu, logvar = model(mlp_xb, cnn_xb)
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
            for mlp_xb, cnn_xb, yb in val_loader:
                mlp_xb = mlp_xb.to(device)
                cnn_xb = cnn_xb.to(device)
                yb = yb.to(device)
                
                mu, logvar = model(mlp_xb, cnn_xb)
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

# Inference and write results (Updated to handle two inputs)
def predict_and_save(csv_path, model_path, out_csv='dl_fused_output.csv', device='cpu'):
    # This part must re-run the same feature creation logic as the Dataset.__init__
    ds = SensorFusionDataset(csv_path)
    
    all_MLP_X = torch.tensor(ds.MLP_X, dtype=torch.float32).to(device)
    all_CNN_X = torch.tensor(ds.CNN_X, dtype=torch.float32).to(device)
    
    # Load model and prepare for inference
    ckpt = torch.load(model_path, map_location=device)
    # Instantiate the Multi-Branch model
    model = MultiBranchFusionNet(mlp_in_dim=12, hidden=ckpt.get('hidden', 128)).to(device) 
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    with torch.no_grad():
        mu, logvar = model(all_MLP_X, all_CNN_X)
    mu = mu.cpu().numpy()
    logvar = logvar.cpu().numpy()

    # Read original df to align results
    df = pd.read_csv(csv_path, sep=';')
    
    # --- Align Parsing and Filtering ---
    point_cloud_cols = ['x', 'y', 'z', 'x_static', 'y_static', 'z_static']
    structured_cols_to_parse = ['ble_xyz_replace_filter', 'real_xyz']

    for col in point_cloud_cols + structured_cols_to_parse:
        df[col] = df[col].apply(safe_point_parse)
        
    df['x_ble'] = pd.to_numeric(df['x_ble'], errors='coerce')
    
    # Replicate the row filtering logic to match the model's output length
    df = df[~np.isnan(df['x_ble'])].reset_index(drop=True)
    
    # Build result columns aligned to the filtered rows used in dataset
    fused_xy = []
    fused_unc = []
    j = 0 # Index for the model prediction results (mu, logvar)
    
    for i, r in df.iterrows():
        # Re-run the eligibility check for the row using the same logic as the dataset builder
        ble = r['ble_xyz_replace_filter']
        is_ble_valid = len(ble) == 3

        if not is_ble_valid:
            # Row was excluded from training/inference
            fused_xy.append([np.nan, np.nan])
            fused_unc.append([np.nan, np.nan])
            continue
        
        # This row was used in the dataset and has a prediction
        fused_xy.append([float(mu[j,0]), float(mu[j,1])])
        fused_unc.append([float(np.exp(logvar[j,0])), float(np.exp(logvar[j,1]))])  # variance
        j += 1

    df['dl_fused_xy'] = fused_xy
    df['dl_fused_var_xy'] = fused_unc
    df['dl_sensor_fused_xyz'] = df['dl_fused_xy'].apply(lambda p: [p[0], p[1], 1.78] if not any(np.isnan(p)) else [np.nan,np.nan,np.nan])
    df.to_csv(out_csv, sep=';', index=False)
    print("Wrote:", out_csv)
    return df

# CLI usage
if __name__ == "__main__":
    CSV = "fused_dataset.csv" 
    
    full_dataset = pd.read_csv(CSV, sep=';')
    
    # Initialize empty files with headers
    full_dataset.head(0).to_csv("dl_train_dataset.csv", sep=';', index=False)
    full_dataset.head(0).to_csv("dl_test_dataset.csv", sep=';', index=False)
    
    i = 0
    # Grouping by 'distance' for stratified split (kept original logic)
    for key, group in full_dataset.groupby('distance'):
        train_size = int(0.8 * len(group))
        train_group = group.iloc[:train_size]
        test_group = group.iloc[train_size:]
        
        header_needed = (i == 0)

        # Append to train/test files
        train_group.to_csv("dl_train_dataset.csv", mode='a', header=header_needed, sep=';', index=False)
        test_group.to_csv("dl_test_dataset.csv", mode='a', header=header_needed, sep=';', index=False)
        i += 1
        
    # Training and Inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Train the model using the training data
    best_model, _ = train_model("dl_train_dataset.csv", out_dir='dl_out', device=device,
                                 hidden=128, batch_size=64, lr=1e-3, epochs=160, val_split=0.12)
    
    # Predict and save results using the FULL dataset (or test set if required)
    predict_and_save("dl_test_dataset.csv", "dl_out/best.pth", out_csv='dl_fused_output_enhanced.csv', device=device)
    print("Enhanced DL fusion complete. Results in dl_fused_output_enhanced.csv")