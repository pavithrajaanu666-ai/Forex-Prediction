import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import plotly.graph_objects as go
import xgboost as xgb
import yfinance as yf

# MT5 Cloud-la work aagaadhu, so try-except block
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

st.set_page_config(layout="wide", page_title="Hybrid Quant AI")

# ================= LOGIN =================
def login():
    st.title("Hybrid Quant AI v3")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "admin" and pwd == "admin123":
            st.session_state["auth"] = True
            st.rerun()
        else:
            st.error("Invalid Credentials")

if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    login()
    st.stop()

# ================= SIDEBAR =================
st.sidebar.title("Control Panel")
mode = st.sidebar.selectbox("Mode", ["Single Pair", "Compare 4 Pairs"])
pair = st.sidebar.selectbox("Currency Pair", ["EURUSD","GBPUSD","USDJPY","XAUUSD"])
tf = st.sidebar.selectbox("Timeframe", ["M1","H1","D1"])
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
    # Model expects exact these features in order
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

# ================= HYBRID DATA FETCHING (FIXED) =================
def get_hybrid_data(pair, tf):
    tf_map_mt5 = {"M1": 1, "H1": 16385, "D1": 16408}
    tf_map_yf = {"M1": "1m", "H1": "1h", "D1": "1d"}
    
    success = False
    df = pd.DataFrame()

    # 1. Try MT5 (Local)
    if mt5 is not None:
        if mt5.initialize():
            rates = mt5.copy_rates_from_pos(pair, tf_map_mt5[tf], 0, 1000)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                success = True
            mt5.shutdown()

    # 2. Fallback to YFinance (Cloud)
    if not success:
        ticker = f"{pair}=X" if "USD" in pair else pair
        if pair == "XAUUSD": ticker = "GC=F"
        
        data = yf.download(ticker, period="5d", interval=tf_map_yf[tf])
        if not data.empty:
            df = data.reset_index()
            # Clean columns names (fixes MultiIndex issue)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.columns = [c.lower() for c in df.columns]
            df.rename(columns={'datetime': 'time', 'date': 'time'}, inplace=True)
            
            # CRITICAL: Match MT5 column structure to fix ValueError
            df['tick_volume'] = df.get('volume', 0)
            df['spread'] = 0
            df['real_volume'] = 0
            
            # Select only columns MT5 would provide
            df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
            success = True
            
    return df, success

# ================= PREDICTION =================
def predict(pair, tf):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_raw, success = get_hybrid_data(pair, tf)
    
    if not success or df_raw.empty:
        st.error(f"Data fetch failed for {pair}")
        return None, None, None

    df = build_features(df_raw.copy())
    
    # Drop time but KEEP other columns in training order
    X = df.drop(['time'], axis=1)

    path = f"models/{pair}/{tf}"
    try:
        # Load XGBoost
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(f"{path}/xgb.json")

        # Load scalers
        feature_scaler = joblib.load(f"{path}/feature_scaler.pkl")
        target_scaler = joblib.load(f"{path}/target_scaler.pkl")

        # XGBoost prediction
        X_scaled = feature_scaler.transform(X) # Fixed: X now matches scaler features
        xgb_pred = xgb_model.predict(X_scaled[-1:])[0]

        # TFT prediction
        seq = torch.tensor(X_scaled[-20:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        tft = TFT(X.shape[1]).to(DEVICE)
        tft.load_state_dict(torch.load(f"{path}/tft.pth", map_location=DEVICE))
        tft.eval()
        tft_scaled = tft(seq).detach().cpu().numpy()
        tft_pred = target_scaler.inverse_transform(tft_scaled)[0][0]

        # DGN stacking
        dgn = DGN().to(DEVICE)
        dgn.load_state_dict(torch.load(f"{path}/dgn.pth", map_location=DEVICE))
        dgn.eval()
        stack = torch.tensor([[xgb_pred, tft_pred]], dtype=torch.float32).to(DEVICE)
        final_pred = dgn(stack).detach().cpu().numpy()[0][0]

        current = df['close'].iloc[-1]
        return df, current, final_pred

    except Exception as e:
        st.error(f"Model error for {pair}: {e}")
        return None, None, None

# ================= MAIN =================
if st.button("Run AI Forecast"):
    pairs = ["EURUSD","GBPUSD","USDJPY","XAUUSD"] if mode=="Compare 4 Pairs" else [pair]

    for p in pairs:
        with st.spinner(f"Predicting {p}..."):
            df, current, predicted = predict(p, tf)
            
        if df is not None:
            move = predicted
            signal = "BUY" if move > 0 else "SELL" if move < 0 else "HOLD"
            pct = move * 100

            st.markdown(f"### {p} Result")
            col1, col2 = st.columns([3, 1])

            with col1:
                fig = go.Figure(data=[go.Candlestick(
                    x=df['time'].tail(50),
                    open=df['open'].tail(50), high=df['high'].tail(50),
                    low=df['low'].tail(50), close=df['close'].tail(50)
                )])
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                color = "green" if signal == "BUY" else "red" if signal == "SELL" else "white"
                st.metric("Current Price", round(current, 5))
                st.metric("Predicted Move", f"{round(pct, 4)}%", delta=round(move, 5))
                st.markdown(f"## Signal: :{color}[{signal}]")
