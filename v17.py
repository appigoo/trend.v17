# ====================== app.py ======================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import requests
from datetime import datetime
import os

# ====================== 頁面設定 ======================
st.set_page_config(
    page_title="美股實時監控系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 美股實時監控系統")
st.markdown("**使用 yfinance 即時抓取 + Plotly 互動圖表 + Telegram 警示**")

# ====================== Sidebar ======================
st.sidebar.header("⚙️ 設定面板")

tickers_input = st.sidebar.text_input(
    "股票代號 (多檔用逗號分隔，例如 TSLA,NIO,AAPL)",
    value="TSLA,NIO"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

interval_options = ["1分鐘", "5分鐘", "15分鐘", "30分鐘", "日K", "週K", "月K"]
selected_interval_cn = st.sidebar.selectbox("時間週期", interval_options, index=4)

auto_refresh = st.sidebar.checkbox("✅ 自動刷新", value=True)
refresh_sec = st.sidebar.slider("刷新間隔 (秒)", min_value=60, max_value=300, value=60, step=30)

# yfinance 對應表
interval_map = {
    "1分鐘": "1m", "5分鐘": "5m", "15分鐘": "15m", "30分鐘": "30m",
    "日K": "1d", "週K": "1wk", "月K": "1mo"
}
yf_interval = interval_map[selected_interval_cn]

period_map = {
    "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
    "1d": "2y", "1wk": "5y", "1mo": "max"
}
yf_period = period_map[yf_interval]

# Telegram 設定 (從 st.secrets 安全讀取)
telegram_enabled = False
try:
    TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
    telegram_enabled = True
except:
    st.sidebar.warning("⚠️ Telegram 警示未設定，請在 .streamlit/secrets.toml 加入 TELEGRAM_BOT_TOKEN 與 TELEGRAM_CHAT_ID")

# ====================== 快取資料抓取 ======================
@st.cache_data(ttl=30, show_spinner=False)
def get_stock_data(ticker: str, interval: str, period: str):
    try:
        df = yf.download(ticker, interval=interval, period=period, auto_adjust=False, prepost=False)
        if df.empty:
            return pd.DataFrame()
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"{ticker} 資料抓取失敗: {e}")
        return pd.DataFrame()

# ====================== 技術指標計算 ======================
def calculate_indicators(df: pd.DataFrame):
    if df.empty or len(df) < 200:
        return df

    ema_periods = [5, 10, 20, 30, 40, 60, 120, 200]
    colors = ["#00FF00", "#FFFF00", "#FF8800", "#FF0000", "#AA00FF", "#0088FF", "#00FFFF", "#8800FF"]

    for p in ema_periods:
        df[f'EMA{p}'] = df['Close'].ewm(span=p, adjust=False).mean()

    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA15'] = df['Close'].rolling(15).mean()

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['DIF'] - df['DEA']

    return df, ema_periods, colors

# ====================== 支撐阻力 (Pivot) ======================
def detect_support_resistance(df: pd.DataFrame, window: int = 5, num_levels: int = 5):
    if len(df) < window * 2:
        return [], []
    resist = []
    support = []
    for i in range(window, len(df) - window):
        # Pivot High
        if all(df['High'].iloc[i] >= df['High'].iloc[i - j] for j in range(1, window + 1)) and \
           all(df['High'].iloc[i] >= df['High'].iloc[i + j] for j in range(1, window + 1)):
            resist.append(df['High'].iloc[i])
        # Pivot Low
        if all(df['Low'].iloc[i] <= df['Low'].iloc[i - j] for j in range(1, window + 1)) and \
           all(df['Low'].iloc[i] <= df['Low'].iloc[i + j] for j in range(1, window + 1)):
            support.append(df['Low'].iloc[i])
    return sorted(resist[-num_levels:], reverse=True), sorted(support[-num_levels:])

# ====================== 訊號偵測 ======================
def detect_signals(df: pd.DataFrame, resist: list, support: list, current_price: float):
    if len(df) < 3:
        return {}

    signals = {}
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # MACD 金叉 / 死叉
    if prev['DIF'] <= prev['DEA'] and last['DIF'] > last['DEA']:
        signals['MACD_gold'] = f"MACD 金叉 (DIF:{last['DIF']:.4f} > DEA:{last['DEA']:.4f})"
    if prev['DIF'] >= prev['DEA'] and last['DIF'] < last['DEA']:
        signals['MACD_death'] = f"MACD 死叉 (DIF:{last['DIF']:.4f} < DEA:{last['DEA']:.4f})"

    # EMA5 與 EMA20 交叉
    if 'EMA5' in df.columns and 'EMA20' in df.columns:
        if prev['EMA5'] <= prev['EMA20'] and last['EMA5'] > last['EMA20']:
            signals['EMA5_gold'] = "EMA5 上穿 EMA20 → 多頭排列開始"
        if prev['EMA5'] >= prev['EMA20'] and last['EMA5'] < last['EMA20']:
            signals['EMA5_death'] = "EMA5 下穿 EMA20 → 空頭排列開始"

    # 所有 EMA 多頭排列
    ema_cols = [c for c in df.columns if c.startswith('EMA')]
    if len(ema_cols) >= 5:
        ema_values = [last[c] for c in sorted(ema_cols, key=lambda x: int(x[3:]))]
        if all(ema_values[i] > ema_values[i+1] for i in range(len(ema_values)-1)):
            signals['all_bull'] = "✅ 所有 EMA 多頭排列完成"

    # 成交量暴增 (超過 5 日均量 2 倍)
    if len(df) >= 5:
        vol_mean = df['Volume'].iloc[-6:-1].mean()
        if last['Volume'] > 2 * vol_mean and vol_mean > 0:
            signals['volume_spike'] = f"🚀 放量暴增！當前 {last['Volume']/1e6:.1f}萬股 (均量 {vol_mean/1e6:.1f}萬股)"

    # 價格突破阻力 / 跌破支撐
    if resist and current_price > max(resist) * 1.001:
        signals['break_resist'] = f"🔥 突破阻力位 {max(resist):.2f}"
    if support and current_price < min(support) * 0.999:
        signals['break_support'] = f"❄️ 跌破支撐位 {min(support):.2f}"

    return signals

# ====================== Telegram 發送 ======================
def send_telegram(ticker: str, msg: str):
    if not telegram_enabled:
        return
    try:
        text = f"【{ticker} 警示】\n{msg}\n時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=5)
    except:
        pass  # 靜默失敗

# ====================== Plotly 圖表 ======================
def create_chart(df: pd.DataFrame, ema_periods: list, colors: list, resist: list, support: list):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.55, 0.20, 0.25],
        subplot_titles=("K線 + EMA/MA", "成交量", "MACD")
    )

    # 1. K線 (台股配色：紅漲綠跌)
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            increasing_line_color='#FF0000', increasing_fillcolor='#FF0000',
            decreasing_line_color='#00AA00', decreasing_fillcolor='#00AA00',
            name="K線"
        ), row=1, col=1
    )

    # EMA & MA
    for i, p in enumerate(ema_periods):
        fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA{p}'], name=f'EMA{p}', line=dict(color=colors[i], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], name='MA5', line=dict(color='#FFFFFF', width=2, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA15'], name='MA15', line=dict(color='#FFAA00', width=2, dash='dot')), row=1, col=1)

    # 支撐阻力虛線
    current_price = df['Close'].iloc[-1]
    for r in resist:
        fig.add_hline(y=r, line=dict(color="#FF00FF", width=1, dash="dash"), row=1, col=1, annotation_text=f"阻 {r:.2f}", annotation_position="top right")
    for s in support:
        fig.add_hline(y=s, line=dict(color="#00FFFF", width=1, dash="dash"), row=1, col=1, annotation_text=f"支 {s:.2f}", annotation_position="bottom right")

    # 2. 成交量 (紅漲綠跌)
    colors_vol = ['#FF0000' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#00AA00' for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name="成交量", marker_color=colors_vol, showlegend=False),
        row=2, col=1
    )

    # 3. MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['DIF'], name='DIF', line=dict(color='#00FFFF')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['DEA'], name='DEA', line=dict(color='#FF8800')), row=3, col=1)
    fig.add_trace(
        go.Bar(x=df.index, y=df['MACD_hist'],
               name='MACD Hist',
               marker_color=['#00FF00' if v >= 0 else '#FF0000' for v in df['MACD_hist']]),
        row=3, col=1
    )

    fig.update_layout(
        height=900,
        title_text=f"{tickers[0] if len(tickers)==1 else '多檔'} {selected_interval_cn} 圖表",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    return fig

# ====================== 趨勢判斷 ======================
def get_trend(df: pd.DataFrame):
    ema_cols = sorted([c for c in df.columns if c.startswith('EMA')], key=lambda x: int(x[3:]))
    if len(ema_cols) < 3:
        return "盤整"
    vals = [df[c].iloc[-1] for c in ema_cols]
    if all(vals[i] > vals[i+1] for i in range(len(vals)-1)):
        return "🔥 強勢多頭"
    if all(vals[i] < vals[i+1] for i in range(len(vals)-1)):
        return "❄️ 強勢空頭"
    if df['Close'].iloc[-1] > df['EMA200'].iloc[-1] if 'EMA200' in df.columns else df['Close'].iloc[-1] > df['EMA60'].iloc[-1]:
        return "📈 多頭趨勢"
    return "📉 空頭趨勢"

# ====================== 主畫面渲染函式 ======================
def render_dashboard(ticker: str):
    df_raw = get_stock_data(ticker, yf_interval, yf_period)
    if df_raw.empty:
        st.error(f"❌ 無法取得 {ticker} 資料")
        return

    df, ema_periods, ema_colors = calculate_indicators(df_raw)
    resist, support = detect_support_resistance(df)
    current_price = df['Close'].iloc[-1]
    signals = detect_signals(df, resist, support, current_price)

    # 儲存警示紀錄 (全域)
    if 'alert_history' not in st.session_state:
        st.session_state.alert_history = []

    # 發送新警示
    for key, msg in signals.items():
        full_msg = f"{ticker} {msg}"
        if full_msg not in [a['msg'] for a in st.session_state.alert_history[-10:]]:  # 避免重複
            send_telegram(ticker, msg)
            st.session_state.alert_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "ticker": ticker,
                "msg": full_msg
            })

    # ====================== 上方指標卡片 ======================
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
    pct = (current_price - prev_close) / prev_close * 100

    with col1:
        st.metric("📍 當前價格", f"{current_price:.2f}", f"{pct:+.2f}%")
    with col2:
        st.metric("趨勢", get_trend(df))
    with col3:
        vol = df['Volume'].iloc[-1] / 1_000_000
        st.metric("成交量", f"{vol:.1f}萬股")
    with col4:
        st.metric("最新更新", datetime.now().strftime("%H:%M:%S"))

    # EMA 當前數值表格
    st.subheader("📊 即時 EMA 數值")
    ema_data = []
    for i, p in enumerate(ema_periods):
        val = df[f'EMA{p}'].iloc[-1]
        ema_data.append({"週期": f"EMA{p}", "數值": round(val, 2), "顏色": ema_colors[i]})
    ema_df = pd.DataFrame(ema_data)
    st.dataframe(
        ema_df.style.apply(lambda row: [f'color: {row["顏色"]}' if col == '數值' else '' for col in ema_df.columns], axis=1),
        hide_index=True, use_container_width=True
    )

    # ====================== 圖表 ======================
    fig = create_chart(df, ema_periods, ema_colors, resist, support)
    st.plotly_chart(fig, use_container_width=True)

    # ====================== MACD 數值 ======================
    st.subheader("📉 MACD 即時數值")
    c1, c2, c3 = st.columns(3)
    c1.metric("DIF", f"{df['DIF'].iloc[-1]:.4f}")
    c2.metric("DEA", f"{df['DEA'].iloc[-1]:.4f}")
    hist = df['MACD_hist'].iloc[-1]
    c3.metric("MACD柱狀", f"{hist:.4f}", delta="正柱" if hist > 0 else "負柱")

    # ====================== 警示面板 ======================
    st.subheader("🚨 即時警示")
    if signals:
        for msg in signals.values():
            st.success(msg)
    else:
        st.info("目前無觸發警示")

    # 歷史警示 (最近 20 筆)
    if st.session_state.alert_history:
        history_df = pd.DataFrame(st.session_state.alert_history[-20:])
        st.subheader("📜 最近警示紀錄")
        st.dataframe(history_df, use_container_width=True)

        csv = history_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="💾 下載警示紀錄 CSV",
            data=csv,
            file_name=f"alert_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ====================== 主程式 ======================
if not tickers:
    st.warning("請輸入至少一個股票代號")
    st.stop()

if len(tickers) == 1:
    render_dashboard(tickers[0])
else:
    tab_list = st.tabs(tickers)
    for idx, ticker in enumerate(tickers):
        with tab_list[idx]:
            render_dashboard(ticker)

# ====================== 自動刷新 ======================
if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()

st.caption("💡 提示：首次載入可能需等待 3~8 秒。資料來源 yfinance • 部署於 Streamlit Cloud 請記得設定 secrets.toml")
