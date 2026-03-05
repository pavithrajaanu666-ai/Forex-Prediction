import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
import xgboost as xgb
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

print("Starting Hybrid Quant AI v3 Training...")

# ---------------- CONFIG ----------------
PAIRS = ["EURUSD","GBPUSD","USDJPY","XAUUSD"]

TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "H1": mt5.TIMEFRAME_H1,
    "D1": mt5.TIMEFRAME_D1
}

SEQ_LEN = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Timeframe-aware DATA_SIZE
DATA_SIZE = {"M1":30000, "H1":20000, "D1":5000}

# ---------------- MT5 INIT ----------------
if not mt5.initialize():
    print("MT5 initialization failed")
    quit()

# ---------------- FEATURE ENGINEERING ----------------
def build_features(df):
    df["MA10"] = df["close"].rolling(10).mean()
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["Return"] = df["close"].pct_change()
    df["Volatility"] = df["high"] - df["low"]
    df["Momentum"] = df["close"].shift(1) - df["close"].shift(5)
    df["Lag1"] = df["close"].shift(1)
    df["Lag2"] = df["close"].shift(2)
    df["Lag3"] = df["close"].shift(3)

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100/(1+rs))

    df.dropna(inplace=True)
    return df

# ---------------- TFT MODEL ----------------
class TemporalFusionT(nn.Module):
    def __init__(self,input_size,hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(hidden_size*2,1)
        self.gate = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc = nn.Linear(hidden_size*2,1)

    def forward(self,x):
        out,_ = self.lstm(x)
        attn_weights = torch.softmax(self.attn_fc(out), dim=1)
        context = torch.sum(attn_weights*out, dim=1)
        gated = context * torch.sigmoid(self.gate(context))
        return self.fc(gated)

# ---------------- DGN MODEL ----------------
class DGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,32)
        self.fc2 = nn.Linear(32,1)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ---------------- TRAIN LOOP ----------------
for pair in PAIRS:
    for tf_name, tf in TIMEFRAMES.items():
        print(f"\nTraining {pair} - {tf_name}")
        df_size = DATA_SIZE[tf_name]
        rates = mt5.copy_rates_from_pos(pair, tf, 0, df_size)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = build_features(df)

        df["target"] = df["close"].pct_change().shift(-1)
        df.dropna(inplace=True)

        X = df.drop(["time","target"], axis=1)
        y = df["target"].values

        # ---------------- SCALING ----------------
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        X_scaled = feature_scaler.fit_transform(X)
        y_scaled = target_scaler.fit_transform(y.reshape(-1,1)).flatten()

        split = int(len(X_scaled)*0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y_scaled[:split], y_scaled[split:]

        # ---------------- XGBOOST ----------------
        xgb_model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.03,
            tree_method="hist"  # CPU/GPU safe
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)

        # ---------------- TFT ----------------
        X_seq = np.array([X_scaled[i:i+SEQ_LEN] for i in range(len(X_scaled)-SEQ_LEN)])
        y_seq = y_scaled[SEQ_LEN:]

        split_seq = int(len(X_seq)*0.8)
        X_train_seq = torch.tensor(X_seq[:split_seq], dtype=torch.float32).to(DEVICE)
        y_train_seq = torch.tensor(y_seq[:split_seq], dtype=torch.float32).to(DEVICE)
        X_test_seq = torch.tensor(X_seq[split_seq:], dtype=torch.float32).to(DEVICE)

        tft = TemporalFusionT(X_seq.shape[2]).to(DEVICE)
        optimizer = torch.optim.Adam(tft.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        for epoch in range(8):
            pred = tft(X_train_seq).squeeze()
            loss = loss_fn(pred, y_train_seq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            tft_pred = tft(X_test_seq).cpu().numpy().flatten()

        # ---------------- STACKING ----------------
        min_len = min(len(xgb_pred), len(tft_pred), len(y_test))
        xgb_pred = xgb_pred[:min_len]
        tft_pred = tft_pred[:min_len]
        y_stack = y_test[:min_len]

        stack = np.column_stack((xgb_pred, tft_pred))
        stack_tensor = torch.tensor(stack, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y_stack, dtype=torch.float32).to(DEVICE)

        dgn = DGN().to(DEVICE)
        optimizer = torch.optim.Adam(dgn.parameters(), lr=0.001)
        for epoch in range(20):
            pred = dgn(stack_tensor).squeeze()
            loss = loss_fn(pred, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_pred = dgn(stack_tensor).cpu().numpy().flatten()

        # ---------------- INVERSE SCALING ----------------
        final_pred_inv = target_scaler.inverse_transform(final_pred.reshape(-1,1)).flatten()
        y_true_inv = target_scaler.inverse_transform(y_stack.reshape(-1,1)).flatten()

        # ---------------- METRICS ----------------
        rmse = np.sqrt(mean_squared_error(y_true_inv, final_pred_inv))
        mse = mean_squared_error(y_true_inv, final_pred_inv)
        r2 = r2_score(y_true_inv, final_pred_inv)
        print(f"RMSE: {rmse:.6f} | R2: {r2:.6f}")

        # ---------------- SAVE MODELS ----------------
        path = f"models/{pair}/{tf_name}"
        os.makedirs(path, exist_ok=True)

        joblib.dump(feature_scaler, f"{path}/feature_scaler.pkl")
        joblib.dump(target_scaler, f"{path}/target_scaler.pkl")
        xgb_model.save_model(f"{path}/xgb.json")
        torch.save(tft.state_dict(), f"{path}/tft.pth")
        torch.save(dgn.state_dict(), f"{path}/dgn.pth")

        metrics = {"RMSE": float(rmse), "MSE": float(mse), "R2": float(r2)}
        with open(f"{path}/metrics.json","w") as f:
            json.dump(metrics, f, indent=4)

print("\nALL MODELS TRAINED SUCCESSFULLY")
mt5.shutdown()