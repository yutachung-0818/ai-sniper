import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import warnings

warnings.filterwarnings('ignore')

# --- 頁面設定 ---
st.set_page_config(page_title="Quant Engine", layout="wide", initial_sidebar_state="expanded")

# --- CSS 樣式 ---
st.markdown("""
<style>
    .verdict-box { padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    .action-text { font-size: 26px; font-weight: bold; text-transform: uppercase; }
    .big-score { font-size: 40px; font-weight: bold; margin: 0; }
    .metric-box { background-color: #1E1E1E; padding: 15px; border-radius: 8px; border-left: 5px solid #555; text-align: center; }
    .strategy-card { background-color: #1E1E1E; padding: 15px; border: 1px solid #333; border-radius: 8px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 數據層 (僅在獲取原始數據時快取)
# ==========================================
@st.cache_data(ttl=3600)
def get_data_smart(ticker):
    is_tw = ticker.isdigit() or ticker.endswith('.TW') or ticker.endswith('.TWO')
    yf_ticker = f"{ticker}.TW" if (ticker.isdigit()) else ticker
    try: 
        df = yf.Ticker(yf_ticker).history(period="10y")
        if (df.empty or len(df) < 50) and is_tw:
            yf_ticker = yf_ticker.replace('.TW', '.TWO') if '.TW' in yf_ticker else yf_ticker.replace('.TWO', '.TW')
            df = yf.Ticker(yf_ticker).history(period="10y")
    except Exception as e:
        return None, f"數據下載異常: {str(e)}", is_tw

    if df is None or df.empty or len(df) < 250:
        return None, "數據長度不足(需至少1年)", is_tw
    
    df = df.ffill().bfill()
    bm_ticker = "0050.TW" if is_tw else "SPY"
    try:
        bm = yf.Ticker(bm_ticker).history(period="10y")
        df['BM_Close'] = bm['Close'].reindex(df.index, method='ffill').ffill().bfill()
    except Exception:
        df['BM_Close'] = df['Close']
    return df, "OK", is_tw

def add_triple_barrier(df, ub_mult=1.5, lb_mult=1.5, t_max=15):
    data = df.copy()
    labels, hit_bars_list, n = [], [], len(data)
    closes, atrs = data['Close'].values, data['ATR'].values
    for i in range(n):
        if i + t_max >= n:
            labels.append(np.nan); hit_bars_list.append(np.nan); continue
        hit, hit_bars = 0, t_max 
        up_b, dn_b = closes[i] + (ub_mult * atrs[i]), closes[i] - (lb_mult * atrs[i])
        for j in range(1, t_max + 1):
            if closes[i+j] >= up_b: hit = 1; hit_bars = j; break
            elif closes[i+j] <= dn_b: hit = 0; hit_bars = j; break
        labels.append(hit); hit_bars_list.append(hit_bars) 
    data['Target'] = labels; data['Hit_Bars'] = hit_bars_list
    return data

def feature_engineering(df):
    data = df.copy()
    try:
        data['BM_MA50'] = data['BM_Close'].rolling(50).mean()
        data['BM_MA200'] = data['BM_Close'].rolling(200).mean()
        data['Market_Regime'] = (data['BM_MA50'] > data['BM_MA200']).astype(int)
        data['MA20'], data['MA60'] = data['Close'].rolling(20).mean(), data['Close'].rolling(60).mean()
        data['Price_to_MA20'], data['Price_to_MA60'] = data['Close'] / data['MA20'], data['Close'] / data['MA60']
        data['RS'] = (data['Close'] / data['Close'].shift(20)) / (data['BM_Close'] / data['BM_Close'].shift(20))
        ema12, ema26 = data['Close'].ewm(span=12, adjust=False).mean(), data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD_Hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
        delta = data['Close'].diff()
        gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001)
        data['RSI'] = 100 - (100 / (1 + gain/loss))
        data['Vol_Surge'] = data['Volume'] / data['Volume'].rolling(20).mean().replace(0, 1)
        data['TR'] = np.maximum((data['High'] - data['Low']), np.maximum(abs(data['High'] - data['Close'].shift(1)), abs(data['Low'] - data['Close'].shift(1))))
        data['ATR'], data['ATR_Pct'] = data['TR'].rolling(14).mean(), data['TR'].rolling(14).mean() / data['Close'] * 100 
        data = data.dropna()
        return add_triple_barrier(data) if len(data) > 150 else None
    except Exception as e:
        st.warning(f"特徵計算異常: {str(e)}")
        return None

# ==========================================
# 2. AI 運算邏輯
# ==========================================
def get_calibrated_models(pos_weight):
    inner_cv = TimeSeriesSplit(n_splits=3)
    params = {'n_estimators': 30, 'max_depth': 4, 'scale_pos_weight': pos_weight, 'eval_metric': 'logloss', 'verbosity': 0}
    xgb_cal = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**params), method='sigmoid', cv=inner_cv)
    rf_cal = CalibratedClassifierCV(estimator=RandomForestClassifier(n_estimators=30, max_depth=5, class_weight='balanced'), method='sigmoid', cv=inner_cv)
    lr_cal = CalibratedClassifierCV(estimator=LogisticRegression(class_weight='balanced', max_iter=500), method='sigmoid', cv=inner_cv)
    return xgb_cal, rf_cal, lr_cal

def run_calibrated_kelly(ai_prob, oos_acc, baseline_acc, rrr=0.85):
    if oos_acc <= baseline_acc or ai_prob <= 0.50: return 0.0
    edge_weight = (oos_acc - baseline_acc) / (1.0 - baseline_acc)
    raw_kelly = ai_prob - ((1.0 - ai_prob) / rrr)
    return max(0.0, min(raw_kelly * edge_weight * 0.5, 0.25))

# ==========================================
# 3. UI 主循環
# ==========================================
st.sidebar.title("🧠 Quant Engine")
user_input = st.sidebar.text_input("輸入代碼 (逗號分隔)", "NVDA, 2330, ONDS, HIMS")
capital = st.sidebar.number_input("本金配置", value=100000)
run_btn = st.sidebar.button("🚀 執行全市場驗證")

if run_btn:
    tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]
    summary_report, equity_curves = [], {}
    progress_bar = st.progress(0)

    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers), text=f"分析中: {ticker}")
        df_raw, status, is_tw = get_data_smart(ticker)
        
        if status != "OK":
            st.error(f"{ticker}: {status}"); continue
            
        df = feature_engineering(df_raw)
        if df is None:
            st.error(f"{ticker}: 樣本數不足以進行 Walk-Forward"); continue

        # 核心驗證與回測
        features = ['RSI', 'ATR_Pct', 'Price_to_MA20', 'Price_to_MA60', 'Vol_Surge', 'MACD_Hist', 'RS', 'Market_Regime']
        
        # 💡 【重大修復】：過濾掉末端沒有標籤的數據，避免 NaN 進入 accuracy_score
        train_df = df.dropna(subset=['Target'])
        X, y = train_df[features].values, train_df['Target'].values
        dates, atrs, hits = train_df.index, train_df['ATR_Pct'].values, train_df['Hit_Bars'].values
        
        if len(X) < 200: 
            st.error(f"{ticker}: 資料點過少"); continue

        tscv = TimeSeriesSplit(n_splits=3)
        oos_scores, bt_results = [], []
        
        for tr_idx, val_idx in tscv.split(X):
            X_tr, X_val, y_tr, y_val = X[tr_idx], X[val_idx], y[tr_idx], y[val_idx]
            scaler = StandardScaler()
            X_tr_s, X_val_s = scaler.fit_transform(X_tr), scaler.transform(X_val)
            
            p_weight = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
            m1, m2, m3 = get_calibrated_models(p_weight)
            m1.fit(X_tr_s, y_tr); m2.fit(X_tr_s, y_tr); m3.fit(X_tr_s, y_tr)
            
            prob = (m1.predict_proba(X_val_s)[:, 1]*0.4 + m2.predict_proba(X_val_s)[:, 1]*0.3 + m3.predict_proba(X_val_s)[:, 1]*0.3)
            oos_scores.append(accuracy_score(y_val, (prob > 0.5).astype(int)))
            
            val_dates, val_atrs, val_hits = dates[val_idx], atrs[val_idx], hits[val_idx]
            for k in range(len(y_val)):
                bt_results.append({'Date': val_dates[k], 'Prob': prob[k], 'Target': y_val[k], 'ATR_Pct': val_atrs[k], 'Hit_Bars': val_hits[k]})

        oos_acc, base_acc = np.mean(oos_scores), max(y.mean(), 1-y.mean())
        
        # 今日預測 (使用包含最新一日的完整 df)
        scaler_f = StandardScaler()
        X_s = scaler_f.fit_transform(X)
        m1_f, m2_f, m3_f = get_calibrated_models((len(y)-y.sum())/max(y.sum(),1))
        m1_f.fit(X_s, y); m2_f.fit(X_s, y); m3_f.fit(X_s, y)
        
        X_latest = scaler_f.transform(df[features].tail(1).values)
        today_prob = (m1_f.predict_proba(X_latest)[0,1]*0.4 + m2_f.predict_proba(X_latest)[0,1]*0.3 + m3_f.predict_proba(X_latest)[0,1]*0.3)
        kelly = run_calibrated_kelly(today_prob, oos_acc, base_acc)

        # 回測淨值 (狀態機)
        bt_df = pd.DataFrame(bt_results).set_index('Date').sort_index()
        bt_df = bt_df[~bt_df.index.duplicated(keep='first')]
        daily_ret = pd.Series(0.0, index=df.index)
        
        in_trade_until = pd.Timestamp.min.tz_localize(df.index.tz) if df.index.tz else pd.Timestamp.min
        
        for date, row in bt_df.iterrows():
            if date <= in_trade_until: continue
            k_pos = run_calibrated_kelly(row['Prob'], oos_acc, base_acc)
            if k_pos > 0:
                ret = k_pos * (1.4 * row['ATR_Pct']/100 if row['Target']==1 else -1.6 * row['ATR_Pct']/100)
                exit_idx = min(df.index.get_loc(date) + int(row['Hit_Bars']), len(df)-1)
                daily_ret.loc[df.index[exit_idx]] = ret
                in_trade_until = df.index[exit_idx]
        
        equity = (1 + daily_ret).cumprod()
        display_name = f"{ticker}(TW)" if is_tw else ticker
        equity_curves[display_name] = equity 
        
        summary_report.append({
            "代碼": display_name,
            "Alpha": "✅" if oos_acc > base_acc else "❌",
            "OOS勝率": f"{oos_acc*100:.1f}%",
            "今日建議": "📈 BUY" if (kelly > 0 and oos_acc > base_acc) else "🛑 HOLD",
            "倉位": f"{kelly*100:.1f}%",
            "夏普": f"{np.sqrt(252)*daily_ret.mean()/(daily_ret.std()+1e-9):.2f}"
        })

    # 輸出結果
    st.subheader("🏆 全市場量化報告")
    if summary_report:
        st.table(pd.DataFrame(summary_report))
        fig = go.Figure()
        for name, curve in equity_curves.items():
            if curve.iloc[-1] != 1.0:
                fig.add_trace(go.Scatter(x=curve.index, y=curve, name=name))
        fig.update_layout(template="plotly_dark", height=400, title="樣本外淨值推演", yaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("請在側邊欄輸入代碼並點擊執行。")
