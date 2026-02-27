import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import requests
import csv
import os
import json

# ── 頁面設定 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="美股即時監控系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS 美化 ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }

    /* ── Metric 卡片 ── */
    [data-testid="stMetric"] {
        background: #1e2235;
        border-radius: 10px;
        padding: 14px 16px;
        border: 1px solid #2e3456;
    }
    [data-testid="stMetricLabel"] > div {
        font-size: 1rem !important;
        color: #aab4cc !important;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    [data-testid="stMetricValue"] > div {
        font-size: 1.8rem !important;
        color: #ffffff !important;
        font-weight: 700;
    }
    [data-testid="stMetricDelta"] > div {
        font-size: 1rem !important;
        font-weight: 600;
    }

    /* ── EMA 數值列 ── */
    .ema-bar {
        background: #151825;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 6px 0 10px 0;
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        border: 1px solid #252840;
    }
    .ema-item {
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        white-space: nowrap;
    }
    .ema-label { opacity: 0.75; font-size: 0.82rem; }

    /* ── 趨勢標籤 ── */
    .trend-card {
        background: #1e2235;
        border-radius: 10px;
        padding: 14px 16px;
        border: 1px solid #2e3456;
        text-align: center;
    }
    .trend-title { font-size: 1rem; color: #aab4cc; font-weight: 600; margin-bottom: 4px; }
    .trend-bull { color: #00ee66; font-weight: 800; font-size: 1.6rem; }
    .trend-bear { color: #ff4455; font-weight: 800; font-size: 1.6rem; }
    .trend-side { color: #ffcc00; font-weight: 800; font-size: 1.6rem; }

    /* ── 警示面板 ── */
    .alert-box {
        padding: 12px 16px;
        border-radius: 8px;
        margin: 5px 0;
        font-size: 0.95rem;
        font-weight: 500;
        line-height: 1.5;
    }
    .alert-bull { background: #0d2e18; border-left: 5px solid #00ee66; color: #88ffbb; }
    .alert-bear { background: #2e0d0d; border-left: 5px solid #ff4455; color: #ffaaaa; }
    .alert-vol  { background: #0d1e38; border-left: 5px solid #44aaff; color: #aaddff; }
    .alert-info { background: #28260d; border-left: 5px solid #ffcc00; color: #fff0aa; }
</style>
""", unsafe_allow_html=True)

# ── 常數設定 ──────────────────────────────────────────────────────────────────
INTERVAL_MAP = {
    "1分鐘": "1m", "5分鐘": "5m", "15分鐘": "15m", "30分鐘": "30m",
    "日K": "1d", "週K": "1wk", "月K": "1mo"
}
PERIOD_MAP = {
    "1m": "1d", "5m": "5d", "15m": "10d", "30m": "30d",
    "1d": "1y", "1wk": "3y", "1mo": "5y"
}
EMA_CONFIGS = [
    (5,   "#00ff88"), (10,  "#ccff00"), (20,  "#ffaa00"),
    (30,  "#ff5500"), (40,  "#cc00ff"), (60,  "#0088ff"),
    (120, "#00ccff"), (200, "#8866ff"),
]
MA_CONFIGS = [(5, "#ffffff", "dash"), (15, "#ffdd66", "dot")]

# ── 警示記錄 (session state) ─────────────────────────────────────────────────
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []
if "sent_alerts" not in st.session_state:
    st.session_state.sent_alerts = set()

# ── Telegram 發送 ─────────────────────────────────────────────────────────────
def send_telegram(msg: str):
    try:
        token   = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=5)
    except Exception:
        pass  # secrets 未設定時靜默忽略

def add_alert(symbol: str, msg: str, atype: str = "info"):
    now = datetime.now().strftime("%H:%M:%S")
    entry = {"時間": now, "股票": symbol, "訊息": msg, "類型": atype}
    key = f"{symbol}|{msg}"
    if key not in st.session_state.sent_alerts:
        st.session_state.alert_log.insert(0, entry)
        st.session_state.alert_log = st.session_state.alert_log[:200]
        st.session_state.sent_alerts.add(key)
        send_telegram(f"📊 [{symbol}] {msg}")

# ── 數據抓取 ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    period = PERIOD_MAP.get(interval, "1y")
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    return df

# ── 技術指標計算 ──────────────────────────────────────────────────────────────
def calc_ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def calc_ma(series, n):
    return series.rolling(n).mean()

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    dif = ema_fast - ema_slow
    dea = calc_ema(dif, signal)
    hist = (dif - dea) * 2
    return dif, dea, hist

def calc_pivot(df, left=5, right=5):
    highs, lows = [], []
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    idx   = df.index
    n = len(df)
    for i in range(left, n - right):
        if high[i] == max(high[i-left:i+right+1]):
            highs.append((idx[i], high[i]))
        if low[i]  == min(low[i-left:i+right+1]):
            lows.append((idx[i], low[i]))
    return highs, lows

def detect_trend(df) -> str:
    if len(df) < 200:
        return "盤整"
    ema5  = calc_ema(df["Close"], 5).iloc[-1]
    ema20 = calc_ema(df["Close"], 20).iloc[-1]
    ema60 = calc_ema(df["Close"], 60).iloc[-1]
    ema200= calc_ema(df["Close"], 200).iloc[-1]
    if ema5 > ema20 > ema60 > ema200:
        return "多頭"
    if ema5 < ema20 < ema60 < ema200:
        return "空頭"
    return "盤整"

# ── 警示邏輯 ──────────────────────────────────────────────────────────────────
def run_alerts(symbol, df):
    if len(df) < 30:
        return
    close = df["Close"]
    vol   = df["Volume"]

    # MACD 金叉/死叉
    dif, dea, _ = calc_macd(close)
    if dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-2] <= dea.iloc[-2]:
        add_alert(symbol, "MACD 金叉 🟢", "bull")
    if dif.iloc[-1] < dea.iloc[-1] and dif.iloc[-2] >= dea.iloc[-2]:
        add_alert(symbol, "MACD 死叉 🔴", "bear")

    # EMA5 穿 EMA20
    ema5  = calc_ema(close, 5)
    ema20 = calc_ema(close, 20)
    if ema5.iloc[-1] > ema20.iloc[-1] and ema5.iloc[-2] <= ema20.iloc[-2]:
        add_alert(symbol, "EMA5 上穿 EMA20（多頭排列開始）⬆️", "bull")
    if ema5.iloc[-1] < ema20.iloc[-1] and ema5.iloc[-2] >= ema20.iloc[-2]:
        add_alert(symbol, "EMA5 下穿 EMA20（空頭排列開始）⬇️", "bear")

    # 全 EMA 多頭排列
    emas = [calc_ema(close, n).iloc[-1] for n, _ in EMA_CONFIGS]
    if all(emas[i] > emas[i+1] for i in range(len(emas)-1)):
        add_alert(symbol, "所有 EMA 多頭排列 🚀", "bull")

    # 成交量暴增
    vol_ma5 = vol.rolling(5).mean().iloc[-1]
    if vol.iloc[-1] > vol_ma5 * 2:
        add_alert(symbol, f"成交量暴增 {vol.iloc[-1]/vol_ma5:.1f}x 均量 📊", "vol")

    # 支撐阻力突破
    pivots_h, pivots_l = calc_pivot(df.tail(60))
    price = close.iloc[-1]
    if pivots_h:
        resist = max(p[1] for p in pivots_h)
        if price > resist:
            add_alert(symbol, f"突破阻力位 ${resist:.2f} ⚡", "bull")
    if pivots_l:
        support = min(p[1] for p in pivots_l)
        if price < support:
            add_alert(symbol, f"跌破支撐位 ${support:.2f} ⚠️", "bear")

# ── 繪圖 ─────────────────────────────────────────────────────────────────────
def build_chart(symbol, df, interval_label):
    if df.empty:
        return None

    close = df["Close"]
    vol   = df["Volume"]

    # 計算所有指標
    ema_series = {n: calc_ema(close, n) for n, _ in EMA_CONFIGS}
    ma_series  = {n: calc_ma(close, n)  for n, _, _ in MA_CONFIGS}
    dif, dea, hist = calc_macd(close)
    pivots_h, pivots_l = calc_pivot(df.tail(100))

    # 建立子圖
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.2, 0.25],
        vertical_spacing=0.02,
        subplot_titles=(f"{symbol} K線圖 ({interval_label})", "成交量", "MACD (12, 26, 9)"),
    )
    for ann in fig.layout.annotations:
        ann.font.size = 15
        ann.font.color = "#ccddee"

    # ── K線 ──
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=close,
        increasing_line_color="#ff4444", increasing_fillcolor="#ff4444",
        decreasing_line_color="#00cc44", decreasing_fillcolor="#00cc44",
        name="K線", showlegend=False,
    ), row=1, col=1)

    # EMA 線
    for n, color in EMA_CONFIGS:
        fig.add_trace(go.Scatter(
            x=df.index, y=ema_series[n],
            line=dict(color=color, width=1.5),
            name=f"EMA{n}", opacity=0.9,
        ), row=1, col=1)

    # MA 線
    for n, color, dash in MA_CONFIGS:
        fig.add_trace(go.Scatter(
            x=df.index, y=ma_series[n],
            line=dict(color=color, width=2, dash=dash),
            name=f"MA{n}",
        ), row=1, col=1)

    # 支撐阻力水平線
    if pivots_h:
        resist = max(p[1] for p in pivots_h)
        fig.add_hline(y=resist, line=dict(color="#ff8888", dash="dash", width=1.5),
                      annotation_text=f"⬛ 阻力 {resist:.2f}",
                      annotation_font=dict(size=13, color="#ff8888"),
                      annotation_bgcolor="rgba(30,10,10,0.7)",
                      row=1, col=1)
    if pivots_l:
        support = min(p[1] for p in pivots_l)
        fig.add_hline(y=support, line=dict(color="#88ff88", dash="dash", width=1.5),
                      annotation_text=f"⬛ 支撐 {support:.2f}",
                      annotation_font=dict(size=13, color="#88ff88"),
                      annotation_bgcolor="rgba(10,30,10,0.7)",
                      row=1, col=1)

    # 最高/最低標記
    max_idx = df["High"].idxmax()
    min_idx = df["Low"].idxmin()
    fig.add_annotation(x=max_idx, y=df["High"].max(),
        text=f"▲ 最高 {df['High'].max():.2f}", showarrow=True,
        arrowhead=2, arrowcolor="#ff4444", arrowwidth=2,
        font=dict(color="#ff8888", size=13, family="Arial Black"),
        bgcolor="rgba(30,10,10,0.75)", bordercolor="#ff4444", borderwidth=1,
        row=1, col=1)
    fig.add_annotation(x=min_idx, y=df["Low"].min(),
        text=f"▼ 最低 {df['Low'].min():.2f}", showarrow=True,
        arrowhead=2, arrowcolor="#00cc44", arrowwidth=2,
        font=dict(color="#88ffaa", size=13, family="Arial Black"),
        bgcolor="rgba(10,30,10,0.75)", bordercolor="#00cc44", borderwidth=1,
        row=1, col=1)

    # ── 成交量 ──
    colors_vol = ["#ff4444" if c >= o else "#00cc44"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=vol, marker_color=colors_vol,
        name="成交量", showlegend=False,
    ), row=2, col=1)

    # 均量線
    vol_ma5 = vol.rolling(5).mean()
    fig.add_trace(go.Scatter(
        x=df.index, y=vol_ma5,
        line=dict(color="#ffaa00", width=1),
        name="VOL MA5",
    ), row=2, col=1)

    # 異常放量標記
    anomaly = vol > vol_ma5 * 2
    fig.add_trace(go.Scatter(
        x=df.index[anomaly], y=vol[anomaly],
        mode="markers", marker=dict(color="#ff00ff", size=6, symbol="star"),
        name="異常放量",
    ), row=2, col=1)

    # ── MACD ──
    bar_colors = ["#ff4444" if v >= 0 else "#00cc44" for v in hist]
    fig.add_trace(go.Bar(
        x=df.index, y=hist, marker_color=bar_colors,
        name="MACD柱", showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=dif, line=dict(color="#ffaa00", width=1.2), name="DIF",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=dea, line=dict(color="#0088ff", width=1.2), name="DEA",
    ), row=3, col=1)

    # 金叉/死叉標記
    for i in range(1, len(dif)):
        if dif.iloc[i] > dea.iloc[i] and dif.iloc[i-1] <= dea.iloc[i-1]:
            fig.add_annotation(x=dif.index[i], y=dif.iloc[i],
                text="⬆ 金叉", showarrow=True, arrowhead=2,
                arrowcolor="#ffdd00", arrowwidth=2,
                font=dict(color="#ffee55", size=12, family="Arial Black"),
                bgcolor="rgba(30,28,0,0.75)", bordercolor="#ffdd00",
                row=3, col=1)
        if dif.iloc[i] < dea.iloc[i] and dif.iloc[i-1] >= dea.iloc[i-1]:
            fig.add_annotation(x=dif.index[i], y=dif.iloc[i],
                text="⬇ 死叉", showarrow=True, arrowhead=2,
                arrowcolor="#ff6666", arrowwidth=2,
                font=dict(color="#ff8888", size=12, family="Arial Black"),
                bgcolor="rgba(30,0,0,0.75)", bordercolor="#ff6666",
                row=3, col=1)

    fig.update_layout(
        height=860,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#111520",
        font=dict(family="Arial, sans-serif", size=13, color="#ccddee"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=12, color="#ddeeff"),
            bgcolor="rgba(14,17,23,0.7)",
            bordercolor="#2e3456", borderwidth=1,
        ),
        margin=dict(l=12, r=12, t=50, b=10),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e2235", tickfont=dict(size=12))
    fig.update_yaxes(showgrid=True, gridcolor="#1e2235", tickfont=dict(size=12))

    return fig

# ── 渲染單一股票 ──────────────────────────────────────────────────────────────
def render_symbol(symbol, interval_label, show_alerts):
    interval = INTERVAL_MAP[interval_label]
    with st.spinner(f"載入 {symbol} 數據中..."):
        df = fetch_data(symbol, interval)

    if df is None or df.empty:
        st.error(f"❌ 無法取得 {symbol} 數據，請確認代號是否正確。")
        return

    # 計算當前值
    close    = df["Close"]
    last_close = float(close.iloc[-1])
    last_open  = float(df["Open"].iloc[-1])
    chg        = last_close - float(close.iloc[-2]) if len(close) > 1 else 0
    chg_pct    = chg / float(close.iloc[-2]) * 100 if len(close) > 1 else 0
    vol_now    = int(df["Volume"].iloc[-1])

    # 趨勢判斷
    trend = detect_trend(df)

    # 頂部指標列
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("最新價格", f"${last_close:.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
    col2.metric("成交量 (萬股)", f"{vol_now/10000:.1f}")
    col3.metric("最高", f"${df['High'].iloc[-1]:.2f}")
    col4.metric("最低", f"${df['Low'].iloc[-1]:.2f}")
    trend_class = {"多頭": "trend-bull", "空頭": "trend-bear", "盤整": "trend-side"}[trend]
    trend_icon  = {"多頭": "▲", "空頭": "▼", "盤整": "◆"}[trend]
    with col5:
        st.markdown(
            f'<div class="trend-card">'
            f'<div class="trend-title">趨勢判斷</div>'
            f'<div class="{trend_class}">{trend_icon} {trend}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # EMA 數值列（大字清晰版）
    ema_items = []
    for n, color in EMA_CONFIGS:
        val = float(calc_ema(close, n).iloc[-1])
        above = "↑" if last_close > val else "↓"
        ema_items.append(
            f'<span class="ema-item" style="color:{color}">'
            f'<span class="ema-label">EMA{n} </span>{val:.2f} '
            f'<span style="font-size:0.75rem;opacity:0.6">{above}</span>'
            f'</span>'
        )
    st.markdown(
        '<div class="ema-bar">' + "".join(ema_items) + '</div>',
        unsafe_allow_html=True,
    )

    # K線 + 成交量 + MACD 圖
    fig = build_chart(symbol, df, interval_label)
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    # 警示
    if show_alerts:
        run_alerts(symbol, df)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 美股監控系統")
    st.markdown("---")

    raw_input = st.text_area("股票代號（逗號分隔）", value="TSLA,AAPL,NVDA", height=80)
    symbols = [s.strip().upper() for s in raw_input.replace("，", ",").split(",") if s.strip()]

    interval_label = st.selectbox("時間週期", list(INTERVAL_MAP.keys()), index=4)

    st.markdown("---")
    auto_refresh = st.toggle("自動刷新", value=False)
    refresh_sec  = st.slider("刷新間隔（秒）", 60, 300, 60, step=30, disabled=not auto_refresh)

    st.markdown("---")
    show_alerts = st.toggle("啟用警示偵測", value=True)

    if st.button("🗑️ 清除警示記錄"):
        st.session_state.alert_log = []
        st.session_state.sent_alerts = set()
        st.toast("警示記錄已清除")

    if st.session_state.alert_log:
        csv_data = pd.DataFrame(st.session_state.alert_log).to_csv(index=False, encoding="utf-8-sig")
        st.download_button("📥 匯出警示 CSV", csv_data, "alerts.csv", "text/csv")

    st.markdown("---")
    st.caption("數據來源：Yahoo Finance\n\n⚠️ 僅供參考，不構成投資建議")

# ── 主區域：Tab 切換 ──────────────────────────────────────────────────────────
st.title("🇺🇸 美股即時監控系統")

if not symbols:
    st.info("請在左側輸入股票代號")
    st.stop()

tabs = st.tabs([f"📊 {s}" for s in symbols])
for tab, symbol in zip(tabs, symbols):
    with tab:
        render_symbol(symbol, interval_label, show_alerts)

# ── 警示面板 ──────────────────────────────────────────────────────────────────
if st.session_state.alert_log:
    st.markdown("---")
    st.subheader("🔔 警示訊息記錄")
    type_map = {
        "bull": "alert-bull", "bear": "alert-bear",
        "vol": "alert-vol",   "info": "alert-info",
    }
    for entry in st.session_state.alert_log[:30]:
        cls = type_map.get(entry["類型"], "alert-info")
        st.markdown(
            f'<div class="alert-box {cls}">🕐 {entry["時間"]}　【{entry["股票"]}】　{entry["訊息"]}</div>',
            unsafe_allow_html=True,
        )

# ── 自動刷新 ──────────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_sec)
    st.cache_data.clear()
    st.rerun()
