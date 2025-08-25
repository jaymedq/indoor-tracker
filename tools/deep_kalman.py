# dl_to_kalman.py
import ast
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from deep_fusion import FusionMLP

# ---------------------------
# Helper: load dataset and model outputs
# ---------------------------
def build_measurements_from_model(csv_path, model_path, device='cpu'):
    df = pd.read_csv(csv_path, sep=';')
    df = df[~np.isnan(df[['x_ble']]).any(axis=1)].reset_index(drop=True)
    df['centroid_xyz']   = df['centroid_xyz'].apply(eval)
    df['ble_xyz_filter'] = df['ble_xyz_filter'].apply(eval)
    df['real_xyz']       = df['real_xyz'].apply(eval)

    # Build feature matrix using same criteria as training
    feats = []
    times = []
    valid_idx = []
    for i, r in df.iterrows():
        mm = r['centroid_xyz']
        ble = r['ble_xyz_filter']
        ts = dt.datetime.strptime(r['timestamp'], "%Y-%m-%d %H:%M:%S")
        if any([np.isnan(v) for v in mm[:2]]) or any([np.isnan(v) for v in ble[:2]]):
            feats.append(None)
            times.append(ts)
            continue
        feats.append([mm[0], mm[1], mm[2], ble[0], ble[1], ble[2]])
        times.append(ts)
        valid_idx.append(i)

    # Load model
    ckpt = torch.load(model_path, map_location=device)
    hidden = ckpt.get('hidden', 128)
    model = FusionMLP(in_dim=6, hidden=hidden).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Run model on valid features
    X = np.array([f for f in feats if f is not None], dtype=np.float32)
    with torch.no_grad():
        xb = torch.tensor(X).to(device)
        mu, logvar = model(xb)
        mu = mu.cpu().numpy()
        var = np.exp(logvar.cpu().numpy())  # variance

    # Map results back into full-length arrays (None -> nan)
    z_list = [None] * len(feats)
    r_list = [None] * len(feats)
    j = 0
    for i in range(len(feats)):
        if feats[i] is None:
            z_list[i] = np.array([np.nan, np.nan])
            r_list[i] = np.array([[np.nan, 0.0],[0.0, np.nan]])
        else:
            z_list[i] = mu[j].copy()
            r_list[i] = np.diag(var[j].copy())
            j += 1

    return df, z_list, r_list, times

# ---------------------------
# Estimate Q from measurements (offline)
# Convert DL positions into state vector [x,y,vx,vy] by finite differences,
# compute w_k = x_{k+1} - F(dt_k) x_k, and take sample covariance.
# ---------------------------
def estimate_process_noise_Q(z_list, times, min_valid=10):
    # Build states where both t and t+1 are valid
    states = []
    ts_vals = []
    for i in range(len(z_list)):
        if np.any(np.isnan(z_list[i])):
            states.append(None)
            ts_vals.append(times[i])
        else:
            # placeholder for state; velocity will be computed below
            states.append(np.array([z_list[i][0], z_list[i][1], np.nan, np.nan]))
            ts_vals.append(times[i])

    # Compute velocities by finite differences where possible
    for i in range(1, len(states)):
        if states[i] is not None and states[i-1] is not None:
            dt = (ts_vals[i] - ts_vals[i-1]).total_seconds()
            if dt <= 0 or dt > 5.0:
                continue
            dx = (states[i][0] - states[i-1][0]) / dt
            dy = (states[i][1] - states[i-1][1]) / dt
            states[i-1][2] = dx
            states[i-1][3] = dy
    # For last valid, try backward difference
    for i in range(len(states)-1,0,-1):
        if states[i] is not None and np.isnan(states[i][2]):
            # try from previous valid
            # find previous valid index j < i
            j = i-1
            while j >= 0 and states[j] is None:
                j -= 1
            if j >= 0:
                dt = (ts_vals[i] - ts_vals[j]).total_seconds()
                if dt > 0 and dt <= 5.0:
                    dx = (states[i][0] - states[j][0]) / dt
                    dy = (states[i][1] - states[j][1]) / dt
                    states[j][2] = dx
                    states[j][3] = dy

    # Now build transition residuals w_k = x_{k+1} - F(dt) x_k
    residuals = []
    for i in range(len(states)-1):
        s_k = states[i]
        s_k1 = states[i+1]
        if s_k is None or s_k1 is None:
            continue
        if np.isnan(s_k).any() or np.isnan(s_k1).any():
            continue
        dt = (ts_vals[i+1] - ts_vals[i]).total_seconds()
        if dt <= 0 or dt > 5.0:
            continue
        F = np.array([[1,0,dt,0],
                      [0,1,0,dt],
                      [0,0,1,0],
                      [0,0,0,1]], dtype=float)
        s_k = s_k.reshape(4,1)
        s_k1 = s_k1.reshape(4,1)
        w = (s_k1 - (F @ s_k)).reshape(4)
        residuals.append(w)
    if len(residuals) < min_valid:
        # fallback: small diagonal Q
        Q = np.diag([0.1, 0.1, 0.5, 0.5])
        print(f"[estimate_process_noise_Q] insufficient residuals ({len(residuals)}), returning fallback Q")
        return Q

    R = np.cov(np.vstack(residuals).T, bias=False)  # 4x4
    # ensure symmetric positive semi-definite
    R = (R + R.T) / 2.0
    # add tiny floor for numerical stability
    eps = 1e-8
    R += np.eye(4) * eps
    return R

# ---------------------------
# Simple Kalman Filter (CV)
# ---------------------------
class KalmanFilter2D:
    def __init__(self, Q, R):
        # State vector [x, y, vx, vy]
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 10.0
        self.F = np.eye(4)
        self.F[0, 2] = 1.0
        self.F[1, 3] = 1.0
        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.Q = Q
        self.R = R

    def predict(self, dt=None):
        if dt is not None:
            self.dt = dt
            self.F[0,2] = dt
            self.F[1,3] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update_pos(self, z, R=None):
        if R is None:
            R = self.R
        z = z.reshape(2, 1)  # ensure column vector
        y = z - self.H @ self.x                # innovation (2x1)
        S = self.H @ self.P @ self.H.T + R     # innovation covariance (2x2)
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain (4x2)
        self.x = self.x + K @ y                # update state (4x1)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x


# ---------------------------
# Run the full pipeline
# ---------------------------
def run_dl_to_kf(csv_path, model_path, out_csv='dl_kf_output.csv', device='cpu'):
    df, z_list, r_list, times = build_measurements_from_model(csv_path, model_path, device=device)
    # estimate Q from z_list + times
    print("Estimating Q from DL predictions...")
    Q = estimate_process_noise_Q(z_list, times)
    print("Estimated Q:\n", Q)

    kf = KalmanFilter2D(Q=Q, R=Q)
    tracked = []
    prev_t = None
    # initialize state using first available z
    for i in range(len(z_list)):
        if not np.any(np.isnan(z_list[i])):
            kf.x[0:2, 0] = z_list[i].reshape(2)  # <-- reshape to match (2,1)
            # initial velocity
            kf.x[2:4, 0] = 0.0
            prev_t = times[i]
            break
    if prev_t is None:
        raise RuntimeError("No valid DL measurement in data")

    for i in range(len(z_list)):
        t = times[i]
        dt_val = (t - prev_t).total_seconds()
        if dt_val <= 0 or dt_val > 5.0:
            dt_val = 1.0
        prev_t = t

        # predict
        kf.predict(dt_val)

        # update if DL measurement exists
        if not np.any(np.isnan(z_list[i])):
            z = z_list[i].reshape(2)
            R = r_list[i]
            # guard against degenerate R
            if np.isnan(R).any():
                R = np.diag([1.0, 1.0])
            kf.update_pos(z.reshape(2,1), R)

        tracked.append(kf.x.copy())

    # write results
    tracked_arr = np.array(tracked)
    df['kf_x'] = list(tracked_arr[:,0])
    df['kf_y'] = list(tracked_arr[:,1])
    df['kf_vx'] = list(tracked_arr[:,2])
    df['kf_vy'] = list(tracked_arr[:,3])
    df['sensor_fused_xyz'] = df[['kf_x','kf_y']].apply(lambda r: [r['kf_x'][0], r['kf_y'][0], 1.78], axis=1)
    df.to_csv(out_csv, sep=';', index=False)
    print("Wrote", out_csv)
    return df, Q

# ---------------------------
# If run as script
# ---------------------------
if __name__ == "__main__":
    CSV = "FUSAO_PROCESSADA.csv"
    MODEL = "dl_out/best.pth"   # path to your trained heteroscedastic DL model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df_out, Q_est = run_dl_to_kf(CSV, MODEL, out_csv='dl_kf_output.csv', device=device)
