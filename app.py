import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import joblib
import plotly.graph_objects as go
import xgboost as xgb

st.set_page_config(layout="wide")

# ================= LOGIN =================
def login():
    st.title("Hybrid Quant AI v3")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "admin" and pwd == "admin123":
            st.session_state["auth"] = True
        else:
            st.error("Invalid Credentials")

if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    login()
    st.stop()

# ================= SIDEBAR =================
st.sidebar.title("Control Panel")

mode = st.sidebar.selectbox(
    "Mode",
    ["Single Pair", "Compare 4 Pairs"]
)

pair = st.sidebar.selectbox(
    "Currency Pair",
    ["EURUSD","GBPUSD","USDJPY","XAUUSD"]
)

tf = st.sidebar.selectbox(
    "Timeframe",
    ["M1","H1","D1"]
)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto Refresh (60s)")

# ================= MODELS =================
class TFT(torch.nn.Module):
    def __init__(self,input_size,hidden_size=64):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attn_fc = torch.nn.Linear(hidden_size*2,1)
        self.gate = torch.nn.Linear(hidden_size*2, hidden_size*2)
        self.fc = torch.nn.Linear(hidden_size*2,1)
    def forward(self,x):
        out,_ = self.lstm(x)
        attn_weights = torch.softmax(self.attn_fc(out), dim=1)
        context = torch.sum(attn_weights*out, dim=1)
        gated = context * torch.sigmoid(self.gate(context))
        return self.fc(gated)

class DGN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2,32)
        self.fc2 = torch.nn.Linear(32,1)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ================= FEATURES =================
def build_features(df):
    df['MA10'] = df['close'].rolling(10).mean()
    df['EMA20'] = df['close'].ewm(span=20).mean()
    df['Return'] = df['close'].pct_change()
    df['Volatility'] = df['high'] - df['low']
    df['Momentum'] = df['close'].shift(1) - df['close'].shift(5)
    df['Lag1'] = df['close'].shift(1)
    df['Lag2'] = df['close'].shift(2)
    df['Lag3'] = df['close'].shift(3)

    # === RSI ===
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100/(1+rs))

    df.dropna(inplace=True)
    return df

# ================= PREDICTION =================
def predict(pair, tf):
    tf_map = {"M1": mt5.TIMEFRAME_M1, "H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1}
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rates = mt5.copy_rates_from_pos(pair, tf_map[tf], 0, 8000)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = build_features(df)
    X = df.drop(['time'], axis=1)

    path = f"models/{pair}/{tf}"

    # --- Load XGBoost ---
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(f"{path}/xgb.json")

    # --- Load scalers ---
    feature_scaler = joblib.load(f"{path}/feature_scaler.pkl")
    target_scaler = joblib.load(f"{path}/target_scaler.pkl")

    # --- XGBoost prediction ---
    X_scaled = feature_scaler.transform(X)
    xgb_pred = xgb_model.predict(X_scaled[-1:])[0]

    # --- TFT prediction ---
    seq = torch.tensor(X_scaled[-20:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    tft = TFT(X.shape[1]).to(DEVICE)
    tft.load_state_dict(torch.load(f"{path}/tft.pth", map_location=DEVICE))
    tft.eval()
    tft_scaled = tft(seq).detach().cpu().numpy()
    tft_pred = target_scaler.inverse_transform(tft_scaled)[0][0]

    # --- DGN stacking ---
    dgn = DGN().to(DEVICE)
    dgn.load_state_dict(torch.load(f"{path}/dgn.pth", map_location=DEVICE))
    dgn.eval()
    stack = torch.tensor([[xgb_pred, tft_pred]], dtype=torch.float32).to(DEVICE)
    final_pred = dgn(stack).detach().cpu().numpy()[0][0]

    current = df['close'].iloc[-1]

    return df, current, final_pred

# ================= MAIN =================
if st.button("Run AI Forecast"):

    if not mt5.initialize():
        st.error("MT5 Failed")
        st.stop()

    pairs = ["EURUSD","GBPUSD","USDJPY","XAUUSD"] if mode=="Compare 4 Pairs" else [pair]

    for p in pairs:
        df, current, predicted = predict(p, tf)
        move = predicted
        signal = "BUY" if move>0 else "SELL" if move<0 else "HOLD"
        pct = move*100

        st.markdown(f"## {p} - {tf}")
        col1,col2 = st.columns([3,1])

        # --- Candlestick chart ---
        with col1:
            fig = go.Figure(data=[go.Candlestick(
                x=df['time'].tail(100),
                open=df['open'].tail(100),
                high=df['high'].tail(100),
                low=df['low'].tail(100),
                close=df['close'].tail(100)
            )])
            fig.add_hline(y=current+predicted, line_dash="dash", line_color="yellow")
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig,use_container_width=True)

        # --- Signal display ---
        with col2:
            color = "green" if signal=="BUY" else "red" if signal=="SELL" else "gray"
            st.markdown(
                f"""
                <div style="padding:20px;
                            border-radius:10px;
                            border:2px solid {color};
                            text-align:center;">
                    <h2>Current</h2>
                    <h1>{round(current,5)}</h1>
                    <h2>Predicted</h2>
                    <h1 style="color:{color};">{round(current+predicted,5)}</h1>
                    <h2 style="color:{color};">{signal}</h2>
                    <p>Expected Move: {round(pct,3)}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    mt5.shutdown()