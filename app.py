import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import requests
import io

# --- 頁面設定 ---
st.set_page_config(page_title="AI Sniper: Ultimate Edition", layout="wide")

# --- CSS 優化 ---
st.markdown("""
<style>
    .verdict-box { padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    .action-text { font-size: 30px; font-weight: bold; text-transform: uppercase; }
    .big-score { font-size: 50px; font-weight: bold; margin: 0; }
    .metric-box { background-color: #262730; padding: 15px; border-radius: 8px; border-left: 5px solid #555; height: 100%; text-align: center; }
    .opt-card { background-color: #2b0030; padding: 15px; border-radius: 8px; border-left: 5px solid #E040FB; margin-bottom: 10px; }
    .strategy-card { background-color: #1E1E1E; padding: 15px; border: 1px solid #333; border-radius: 8px; margin-bottom: 10px; }
    .tp-text { color: #00E676; font-weight: bold; font-size: 16px; }
    .sl-text { color: #FF5252; font-weight: bold; font-size: 16px; }
    .ts-text { color: #29B6F6; font-weight: bold; font-size: 16px; }
    .scan-card { background-color: #1E1E1E; padding: 10px; border: 1px solid #444; border-radius: 5px; margin-bottom: 5px;}
    .tag-bull { background-color: #004d40; color: #00E676; padding: 2px 8px; border-radius: 4px; font-size: 12px; border: 1px solid #00E676; }
    .tag-bear { background-color: #3e2723; color: #FF5252; padding: 2px 8px; border-radius: 4px; font-size: 12px; border: 1px solid #FF5252; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 數據層 (含選擇權與防呆)
# ==========================================

def get_data_smart(ticker, market_type):
    def try_download(symbol):
        try:
            d = yf.Ticker(symbol).history(period="2y")
            return d, yf.Ticker(symbol)
        except: return pd.DataFrame(), None

    df, stock_obj = try_download(ticker)
    
    # 台股後綴自動切換 (防呆機制)
    if (df.empty or len(df) < 50) and "台股" in market_type:
        alt = ticker.replace(".TW", ".TWO") if ticker.endswith(".TW") else ticker.replace(".TWO", ".TW")
        st.toast(f"🔄 嘗試切換代碼: {alt}")
        df, stock_obj = try_download(alt)
            
    if df.empty or len(df) < 150: return None, None, "DATA_TOO_SHORT"
    
    # 補值
    df = df.ffill().bfill()
    
    # 抓取大盤基準
    try:
        bm_ticker = "0050.TW" if "台股" in market_type else "SPY"
        bm = yf.Ticker(bm_ticker).history(period="2y")
        # 對齊索引並補值
        df['BM_Close'] = bm['Close'].reindex(df.index, method='ffill').ffill().bfill()
    except: 
        df['BM_Close'] = df['Close']
        
    df['Rel_Str'] = df['Close'] / df['BM_Close']
    return df, stock_obj, "OK"

def get_options_data(stock_obj):
    """ 獲取選擇權 Put/Call Ratio """
    try:
        dates = stock_obj.options
        if not dates: return None
        # 抓最近期合約
        chain = stock_obj.option_chain(dates[0])
        calls = chain.calls['volume'].sum() 
        puts = chain.puts['volume'].sum()
        
        if calls == 0: calls = 1 # 避免除以零
        
        pcr = round(puts / calls, 2)
        sent = "極度避險(看漲?)" if pcr > 1.2 else ("極度樂觀(看跌?)" if pcr < 0.6 else "中性")
        return {"pcr": pcr, "sent": sent, "c": int(calls), "p": int(puts)}
    except: return None

def feature_engineering(df):
    data = df.copy()
    try:
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA60'] = data['Close'].rolling(60).mean()
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001)
        data['RSI'] = 100 - (100 / (1 + gain/loss))
        
        data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Vol'] = data['Log_Ret'].rolling(20).std()
        
        # 籌碼 POC
        price_bins = pd.cut(data['Close'], bins=50)
        vol_profile = data.groupby(price_bins, observed=True)['Volume'].sum()
        data['POC'] = vol_profile.idxmax().mid
        
        data['TR'] = np.maximum((data['High'] - data['Low']), 
                                np.maximum(abs(data['High'] - data['Close'].shift(1)), 
                                           abs(data['Low'] - data['Close'].shift(1))))
        data['ATR'] = data['TR'].rolling(14).mean()
        data['Target'] = (data['Close'].shift(-5) > data['Close']).astype(int)
        
        # 清洗數據
        valid_data = data.dropna()
        if len(valid_data) < 100: return None
        data = data.ffill().bfill()
        return data
    except: return None

# ==========================================
# 2. 三大 AI 模型 (AI Council)
# ==========================================

def run_ensemble_models(df, opt_data=None):
    features = ['RSI', 'Vol', 'MA20', 'ATR']
    # 分割訓練與預測集
    train_df = df.iloc[:-5].dropna()
    latest_row = df.iloc[[-1]]
    
    X_train = train_df[features]
    y_train = train_df['Target']
    X_latest = latest_row[features]
    
    if len(X_train) < 100: return None
    
    try:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_latest_s = scaler.transform(X_latest)
        
        # 1. XGBoost
        xgb_m = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss', use_label_encoder=False)
        xgb_m.fit(X_train_s, y_train)
        p_xgb = xgb_m.predict_proba(X_latest_s)[0][1]
        
        # 2. Random Forest
        rf_m = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf_m.fit(X_train_s, y_train)
        p_rf = rf_m.predict_proba(X_latest_s)[0][1]
        
        # 3. Logistic Regression
        lr_m = LogisticRegression()
        lr_m.fit(X_train_s, y_train)
        p_lr = lr_m.predict_proba(X_latest_s)[0][1]
        
        final = (p_xgb * 0.4) + (p_rf * 0.3) + (p_lr * 0.3)
        
        # 選擇權加權 (若 PCR > 1.2 視為恐慌底部，給予加分)
        if opt_data and opt_data['pcr'] > 1.2:
            final = min(final + 0.1, 1.0)
            
        threshold = 0.6
        con = sum([1 if p > threshold else 0 for p in [p_xgb, p_rf, p_lr]])
        
        return {"final": final, "con": con, "xgb": p_xgb, "rf": p_rf, "lr": p_lr}
    except: return None

def run_black_litterman(df, ai_prob):
    returns = df['Log_Ret'].dropna()
    mu_mkt = returns.mean() * 252 
    sigma = returns.std() * np.sqrt(252)
    view = (ai_prob - 0.5) * 3
    ai_ret = mu_mkt + (view * sigma)
    tau = 0.5
    bl_ret = (mu_mkt + tau * ai_ret) / (1 + tau)
    kelly = (bl_ret - 0.04) / (max(sigma, 0.01)**2)
    
    # 策略濾網：若信心不足，建議空手
    if ai_prob > 0.65 and kelly <= 0: kelly = 0.15 # 強勢股強制試單
    elif ai_prob < 0.55: kelly = 0
    
    # [修正] 凱利上限：單筆不超過 50%
    kelly = max(0, min(kelly, 0.5))
    
    return mu_mkt, ai_ret, bl_ret, kelly, sigma

# ==========================================
# 3. 策略視覺化 (邏輯修正版)
# ==========================================

def calculate_strategy_levels(entry_price, df):
    curr = df.iloc[-1]
    atr = curr['ATR']
    
    # 停利：
    tp1 = entry_price * 1.30
    tp2 = entry_price * 2.00
    
    # 加碼：根據 ATR 計算
    dca_1 = entry_price - (1.0 * atr) 
    dca_2 = entry_price - (2.5 * atr)
    
    # 移動鎖利 (Trailing Stop)
    # 邏輯：必須比 DCA_2 還要低，否則會出現矛盾
    recent_low = df['Low'].iloc[-20:].min()
    trailing_stop = min(recent_low, dca_2 * 0.95) 
    
    return {
        "tp1": tp1, "tp2": tp2,
        "dca1": dca_1, "dca2": dca_2,
        "ts": trailing_stop
    }

# ==========================================
# 4. 全景掃描器
# ==========================================
@st.cache_data(ttl=3600)
def get_tickers(mode):
    if mode == "US":
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            h = {"User-Agent": "Mozilla/5.0"}
            df = pd.read_html(requests.get(url, headers=h).text)[0]
            return [t.replace('.', '-') for t in df['Symbol'].tolist()]
        except: return ['AAPL', 'NVDA', 'TSLA', 'AMD', 'MSFT'] 
    else:
        # 台股熱門與權值股清單
        return [
            "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2412.TW", "2881.TW", "2882.TW", "2303.TW", "1590.TW",
            "2603.TW", "2609.TW", "2615.TW", "3037.TW", "3231.TW", "2356.TW", "2376.TW", "3017.TW", "3035.TW", "3661.TW",
            "6669.TW", "3008.TW", "3045.TW", "4938.TW", "5871.TW", "2891.TW", "2002.TW", "1605.TW", "2327.TW", "2379.TW",
            "2886.TW", "2891.TW", "1216.TW", "2884.TW", "3711.TW", "2892.TW", "2885.TW", "5880.TW", "2880.TW", "1101.TW",
            "3481.TW", "2409.TW", "6770.TW", "5347.TWO", "3293.TWO", "8069.TWO", "6147.TWO", "3105.TWO", "5483.TWO"
        ]

def scan_market_panoramic(tickers):
    data = yf.download(tickers, period="5d", group_by='ticker', threads=True)
    res = []
    for t in tickers:
        try:
            df = data[t].ffill().bfill()
            if df.empty or len(df) < 2: continue
            v_now, v_prev = df['Volume'].iloc[-1], df['Volume'].iloc[-2]
            p_now, p_prev = df['Close'].iloc[-1], df['Close'].iloc[-2]
            if v_prev == 0: continue
            v_chg = (v_now - v_prev) / v_prev
            p_chg = (p_now - p_prev) / p_prev
            
            if v_chg > 0.5:
                signal_type = "Bull" if p_chg > 0 else "Bear"
                res.append({"Code": t, "Price": round(p_now,2), "Chg%": round(p_chg*100,2), "Vol_Chg%": round(v_chg*100,2), "Type": signal_type})
        except: continue
    return pd.DataFrame(res)

# ==========================================
# UI 介面
# ==========================================
with st.sidebar:
    st.header("🧠 終極核心")
    app_mode = st.radio("模式", ["個股深度分析 (All-in-One)", "美股全景掃描", "台股全景掃描"])
    st.markdown("---")
    
    if "個股" in app_mode:
        market = st.selectbox("市場", ["tw 台股", "us 美股"])
        raw = st.text_input("代碼", value="1590" if market == "tw 台股" else "NVDA")
        if "台股" in market:
            ticker = f"{raw}.TW" if not raw.endswith((".TW", ".TWO")) else raw
            sym = "NT$"
        else:
            ticker = raw
            sym = "$"
        cap = st.number_input("資金", value=100000)
        run_btn = st.button("🚀 啟動全功能分析")

if "掃描" in app_mode:
    t_type = "US" if "美股" in app_mode else "TW"
    st.title(f"📡 {t_type} 全景爆量雷達")
    if st.button("🔍 開始全景掃描"):
        with st.spinner("掃描中..."):
            df_s = scan_market_panoramic(get_tickers(t_type))
            if not df_s.empty:
                df_s = df_s.sort_values("Vol_Chg%", ascending=False).head(30)
                for i, r in df_s.iterrows():
                    color = "#00E676" if r['Type'] == "Bull" else "#FF5252"
                    tag = "📈 多方攻擊" if r['Type'] == "Bull" else "📉 空方殺盤"
                    st.markdown(f"""
                    <div class="scan-card" style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="flex:1"><b>{r['Code']}</b> <span style="color:#aaa">${r['Price']}</span><br><span style="font-size:12px;color:{color}">{tag}</span></div>
                        <div style="flex:1; text-align:right;"><span style="font-size:16px;color:#29B6F6">量增 +{r['Vol_Chg%']}%</span></div>
                    </div>""", unsafe_allow_html=True)
            else: st.warning("今日無顯著爆量股")

elif "個股" in app_mode and 'run_btn' in locals() and run_btn:
    with st.spinner(f"正在執行 {ticker} 終極健檢 (AI+選擇權+策略)..."):
        df_raw, stock_obj, status = get_data_smart(ticker, market)
    
    if status == "DATA_TOO_SHORT":
        st.error(f"❌ {ticker} 歷史數據不足。")
    elif df_raw is None:
        st.error(f"❌ 無法獲取數據。")
    else:
        df = feature_engineering(df_raw)
        opt_data = get_options_data(stock_obj)
        
        if df is None: st.error("❌ 有效數據不足。")
        else:
            res = run_ensemble_models(df, opt_data)
            if res:
                prob = res['final']
                mu, ai, bl, kelly, sig = run_black_litterman(df, prob)
                
                curr = df.iloc[-1]
                price = curr['Close']
                ma60 = curr['MA60']
                trend_ok = price > ma60
                
                if res['con'] >= 2 and prob > 0.65 and trend_ok:
                    color = "#00E676"; msg = "強力買進 (Strong Buy)"; desc = "AI 高度共識且順勢"
                elif res['con'] >= 1 and prob > 0.55 and trend_ok:
                    color = "#29B6F6"; msg = "偏多操作 (Bullish)"; desc = "趨勢正確"
                elif not trend_ok:
                    color = "#FFA15A"; msg = "觀望 (Hold)"; desc = "股價低於季線"
                else:
                    color = "#FF5252"; msg = "賣出/空手 (Bearish)"; desc = "AI 看空"

                st.title(f"{ticker} 終極決策報告")
                
                st.markdown(f"""
                <div class="verdict-box" style="background-color:{color}22; border:2px solid {color}">
                    <div class="action-text" style="color:{color}">{msg}</div>
                    <div class="big-score" style="color:{color}">{int(prob*100)}%</div>
                    <div>{desc}</div>
                </div>""", unsafe_allow_html=True)
                
                st.subheader("⚖️ AI 委員會 (The Council)")
                c1, c2, c3 = st.columns(3)
                def get_vote_color(p): return "#00E676" if p > 0.6 else ("#FF5252" if p < 0.4 else "#FFA15A")
                with c1:
                    st.markdown(f"""<div class="metric-box" style="border-left-color:{get_vote_color(res['xgb'])}">
                    <b>XGBoost</b><br><span style="font-size:24px">{int(res['xgb']*100)}%</span></div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="metric-box" style="border-left-color:{get_vote_color(res['rf'])}">
                    <b>Random Forest</b><br><span style="font-size:24px">{int(res['rf']*100)}%</span></div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""<div class="metric-box" style="border-left-color:{get_vote_color(res['lr'])}">
                    <b>Logistic Reg</b><br><span style="font-size:24px">{int(res['lr']*100)}%</span></div>""", unsafe_allow_html=True)
                
                st.write("")
                
                col_opt, col_fund = st.columns([1, 1])
                with col_opt:
                    if opt_data:
                        pcr = opt_data['pcr']
                        pcr_c = "#00E676" if pcr > 1.2 or pcr < 0.6 else "#FFA15A"
                        st.markdown(f"""
                        <div class="opt-card">
                            <h4 style="color:#FFF; margin:0">🟣 選擇權籌碼</h4>
                            <div style="font-size:24px; font-weight:bold; color:{pcr_c}">PCR: {pcr}</div>
                            <div>情緒: {opt_data['sent']}</div>
                            <div style="font-size:12px;color:#ccc">Call: {opt_data['c']:,} | Put: {opt_data['p']:,}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.info("⚠️ 無選擇權數據 (僅美股或熱門股提供)")
                with col_fund:
                    st.markdown(f"""
                    <div class="bl-card">
                        <h4 style="color:#FFF; margin:0">💰 凱利資金配置</h4>
                        <div style="font-size:24px; font-weight:bold; color:#FFF">{sym}{int(cap*kelly):,}</div>
                        <div>建議配置: {int(kelly*100)}% (Max 50%)</div>
                    </div>""", unsafe_allow_html=True)

                st.divider()

                strat = calculate_strategy_levels(price, df)
                
                c_chart, c_plan = st.columns([2, 1])
                with c_chart:
                    st.subheader("📊 策略戰術圖")
                    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線")])
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='yellow', width=2), name='季線'))
                    fig.add_hline(y=strat['tp1'], line_dash="dash", line_color="#00E676", annotation_text="TP1")
                    fig.add_hline(y=strat['dca1'], line_dash="dot", line_color="#FF5252", annotation_text="加碼1")
                    fig.add_hline(y=strat['ts'], line_dash="solid", line_color="#29B6F6", annotation_text="移動鎖利")
                    fig.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                with c_plan:
                    st.subheader("📝 交易劇本")
                    st.markdown(f"""
                    <div class="strategy-card" style="border-left-color:#00E676">
                        <b>💰 獲利目標</b><br>
                        <span class="tp-text">1. {sym}{strat['tp1']:.2f}</span> (賣3成)<br>
                        <span class="tp-text">2. {sym}{strat['tp2']:.2f}</span> (賣5成)
                    </div>
                    <div class="strategy-card" style="border-left-color:#FF5252">
                        <b>🛡️ 加碼防線</b><br>
                        <span class="sl-text">1. {sym}{strat['dca1']:.2f}</span> (加10%)<br>
                        <span class="sl-text">2. {sym}{strat['dca2']:.2f}</span> (加10%)
                    </div>
                    <div class="strategy-card" style="border-left-color:#29B6F6">
                        <b>🛑 最終防線 (Trailing Stop)</b><br>
                        <span class="ts-text">{sym}{strat['ts']:.2f}</span><br>
                        <span style="font-size:12px;color:#aaa">跌破此線全數清倉</span>
                    </div>
                    """, unsafe_allow_html=True)
