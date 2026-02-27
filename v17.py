import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import requests

# ══════════════════════════════════════════════════════════════════════════════
# 頁面設定
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="美股即時監控系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }

    /* Metric 卡片 */
    [data-testid="stMetric"] {
        background: #1e2235; border-radius: 10px;
        padding: 12px 14px; border: 1px solid #2e3456;
    }
    [data-testid="stMetricLabel"] > div {
        font-size: 0.9rem !important; color: #aab4cc !important;
        font-weight: 600; letter-spacing: 0.03em;
    }
    [data-testid="stMetricValue"] > div {
        font-size: 1.55rem !important; color: #ffffff !important; font-weight: 700;
    }
    [data-testid="stMetricDelta"] > div { font-size: 0.9rem !important; font-weight: 600; }

    /* EMA 數值列 */
    .ema-bar {
        background: #151825; border-radius: 8px; padding: 9px 14px;
        margin: 6px 0 8px 0; display: flex; flex-wrap: wrap;
        gap: 12px; border: 1px solid #252840;
    }
    .ema-item { font-size: 0.9rem; font-weight: 600; white-space: nowrap; }
    .ema-label { opacity: 0.7; font-size: 0.78rem; }

    /* 趨勢卡片 */
    .trend-card {
        background: #1e2235; border-radius: 10px;
        padding: 12px 14px; border: 1px solid #2e3456; text-align: center;
    }
    .trend-title { font-size: 0.9rem; color: #aab4cc; font-weight: 600; margin-bottom: 4px; }
    .trend-bull  { color: #00ee66; font-weight: 800; font-size: 1.45rem; }
    .trend-bear  { color: #ff4455; font-weight: 800; font-size: 1.45rem; }
    .trend-side  { color: #ffcc00; font-weight: 800; font-size: 1.45rem; }

    /* 多週期摘要列 */
    .mtf-header {
        background: #151825; border-radius: 10px; padding: 10px 16px;
        margin: 4px 0; border: 1px solid #252840;
        display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    }
    .mtf-period { font-size: 0.85rem; color: #aab4cc; font-weight: 700; min-width: 52px; }
    .mtf-price  { font-size: 1.05rem; color: #ffffff; font-weight: 700; }
    .mtf-chg-up { font-size: 0.88rem; color: #00ee66; font-weight: 600; }
    .mtf-chg-dn { font-size: 0.88rem; color: #ff4455; font-weight: 600; }
    .mtf-trend-bull { background:#0d2e18; color:#00ee66; border-radius:4px; padding:2px 8px; font-size:0.82rem; font-weight:700; }
    .mtf-trend-bear { background:#2e0d0d; color:#ff4455; border-radius:4px; padding:2px 8px; font-size:0.82rem; font-weight:700; }
    .mtf-trend-side { background:#28260d; color:#ffcc00; border-radius:4px; padding:2px 8px; font-size:0.82rem; font-weight:700; }
    .mtf-macd-bull  { color:#00ee66; font-size:0.82rem; }
    .mtf-macd-bear  { color:#ff4455; font-size:0.82rem; }
    .mtf-ema-bull   { color:#00ee66; font-size:0.82rem; }
    .mtf-ema-bear   { color:#ff4455; font-size:0.82rem; }
    .mtf-divider    { height:28px; width:1px; background:#2e3456; flex-shrink:0; }

    /* 區塊標題 */
    .mtf-section-title {
        font-size: 1.1rem; font-weight: 700; color: #ddeeff;
        padding: 8px 0 4px 0; border-bottom: 2px solid #2e3456;
        margin: 14px 0 8px 0;
    }

    /* 警示面板 */
    .alert-box {
        padding: 11px 16px; border-radius: 8px; margin: 4px 0;
        font-size: 0.92rem; font-weight: 500; line-height: 1.5;
    }
    .alert-bull { background:#0d2e18; border-left:5px solid #00ee66; color:#88ffbb; }
    .alert-bear { background:#2e0d0d; border-left:5px solid #ff4455; color:#ffaaaa; }
    .alert-vol  { background:#0d1e38; border-left:5px solid #44aaff; color:#aaddff; }
    .alert-info { background:#28260d; border-left:5px solid #ffcc00; color:#fff0aa; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 常數
# ══════════════════════════════════════════════════════════════════════════════
INTERVAL_MAP = {
    "1m":  ("1分鐘",  "1d"),
    "5m":  ("5分鐘",  "5d"),
    "15m": ("15分鐘", "10d"),
    "30m": ("30分鐘", "30d"),
    "1d":  ("日K",    "1y"),
    "1wk": ("週K",    "3y"),
    "1mo": ("月K",    "5y"),
}
ALL_INTERVALS   = list(INTERVAL_MAP.keys())
INTERVAL_LABELS = {k: v[0] for k, v in INTERVAL_MAP.items()}

EMA_CONFIGS = [
    (5,   "#00ff88"), (10,  "#ccff00"), (20,  "#ffaa00"),
    (30,  "#ff5500"), (40,  "#cc00ff"), (60,  "#0088ff"),
    (120, "#00ccff"), (200, "#8866ff"),
]
MA_CONFIGS = [(5, "#ffffff", "dash"), (15, "#ffdd66", "dot")]

# ══════════════════════════════════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════════════════════════════════
if "alert_log"   not in st.session_state: st.session_state.alert_log   = []
if "sent_alerts" not in st.session_state: st.session_state.sent_alerts = set()

# ══════════════════════════════════════════════════════════════════════════════
# Telegram
# ══════════════════════════════════════════════════════════════════════════════
def send_telegram(msg: str):
    try:
        token   = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=5,
        )
    except Exception:
        pass

def add_alert(symbol: str, period: str, msg: str, atype: str = "info"):
    now = datetime.now().strftime("%H:%M:%S")
    key = f"{symbol}|{period}|{msg}"
    if key not in st.session_state.sent_alerts:
        st.session_state.alert_log.insert(0,
            {"時間": now, "股票": symbol, "週期": period, "訊息": msg, "類型": atype})
        st.session_state.alert_log = st.session_state.alert_log[:200]
        st.session_state.sent_alerts.add(key)
        send_telegram(f"📊 [{symbol} {period}] {msg}")

# ══════════════════════════════════════════════════════════════════════════════
# 數據抓取
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    _, period = INTERVAL_MAP[interval]
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# 技術指標
# ══════════════════════════════════════════════════════════════════════════════
def calc_ema(s, n):  return s.ewm(span=n, adjust=False).mean()
def calc_ma(s, n):   return s.rolling(n).mean()

def calc_macd(s, fast=12, slow=26, sig=9):
    dif  = calc_ema(s, fast) - calc_ema(s, slow)
    dea  = calc_ema(dif, sig)
    return dif, dea, (dif - dea) * 2

def calc_pivot(df, interval: str = "1d"):
    """
    依週期動態調整掃描參數，並用「價格合理範圍過濾（±30%）」
    確保阻力/支撐位一定在當前價格附近，不出現歷史舊極值。
    """
    # 依週期決定 left、right、掃描的最近 N 根 K 線
    pivot_cfg = {
        "1m":  (3, 3, 120),
        "5m":  (3, 3, 100),
        "15m": (3, 3, 80),
        "30m": (3, 3, 60),
        "1d":  (5, 5, 60),
        "1wk": (3, 3, 40),
        "1mo": (2, 2, 24),   # 月K只看近24根(2年)，避免抓到5年前低點
    }
    left, right, tail_n = pivot_cfg.get(interval, (3, 3, 60))

    sub = df.tail(tail_n)
    if len(sub) < left + right + 2:
        return [], []

    hi, lo, idx = sub["High"].values, sub["Low"].values, sub.index
    current_price = float(df["Close"].iloc[-1])

    # 只接受距離當前價格 ±30% 以內的 pivot（過濾歷史遠古價位）
    price_lo = current_price * 0.70
    price_hi = current_price * 1.30

    highs, lows = [], []
    for i in range(left, len(sub) - right):
        if hi[i] == max(hi[i-left:i+right+1]) and price_lo <= hi[i] <= price_hi:
            highs.append((idx[i], float(hi[i])))
        if lo[i] == min(lo[i-left:i+right+1]) and price_lo <= lo[i] <= price_hi:
            lows.append((idx[i], float(lo[i])))

    return highs, lows

def detect_trend(df) -> str:
    if len(df) < 60: return "盤整"
    c = df["Close"]
    e5, e20, e60 = calc_ema(c,5).iloc[-1], calc_ema(c,20).iloc[-1], calc_ema(c,60).iloc[-1]
    e200 = calc_ema(c,200).iloc[-1] if len(df) >= 200 else None
    if e200:
        if e5>e20>e60>e200: return "多頭"
        if e5<e20<e60<e200: return "空頭"
    else:
        if e5>e20>e60: return "多頭"
        if e5<e20<e60: return "空頭"
    return "盤整"

def get_macd_signal(df) -> str:
    if len(df) < 30: return "—"
    dif, dea, _ = calc_macd(df["Close"])
    if dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-2] <= dea.iloc[-2]: return "⬆金叉"
    if dif.iloc[-1] < dea.iloc[-1] and dif.iloc[-2] >= dea.iloc[-2]: return "⬇死叉"
    return "DIF↑" if dif.iloc[-1] > dea.iloc[-1] else "DIF↓"

def get_ema_signal(df) -> str:
    if len(df) < 20: return "—"
    c = df["Close"]
    e5, e20 = calc_ema(c,5), calc_ema(c,20)
    if e5.iloc[-1] > e20.iloc[-1] and e5.iloc[-2] <= e20.iloc[-2]: return "多排↑"
    if e5.iloc[-1] < e20.iloc[-1] and e5.iloc[-2] >= e20.iloc[-2]: return "空排↓"
    return "EMA↑" if e5.iloc[-1] > e20.iloc[-1] else "EMA↓"

# ══════════════════════════════════════════════════════════════════════════════
# 警示邏輯
# ══════════════════════════════════════════════════════════════════════════════
def run_alerts(symbol, period_label, df):
    if len(df) < 30: return
    close, vol = df["Close"], df["Volume"]

    dif, dea, _ = calc_macd(close)
    if dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-2] <= dea.iloc[-2]:
        add_alert(symbol, period_label, "MACD 金叉 🟢", "bull")
    if dif.iloc[-1] < dea.iloc[-1] and dif.iloc[-2] >= dea.iloc[-2]:
        add_alert(symbol, period_label, "MACD 死叉 🔴", "bear")

    e5, e20 = calc_ema(close,5), calc_ema(close,20)
    if e5.iloc[-1] > e20.iloc[-1] and e5.iloc[-2] <= e20.iloc[-2]:
        add_alert(symbol, period_label, "EMA5 上穿 EMA20 ⬆️", "bull")
    if e5.iloc[-1] < e20.iloc[-1] and e5.iloc[-2] >= e20.iloc[-2]:
        add_alert(symbol, period_label, "EMA5 下穿 EMA20 ⬇️", "bear")

    emas = [calc_ema(close,n).iloc[-1] for n,_ in EMA_CONFIGS]
    if all(emas[i] > emas[i+1] for i in range(len(emas)-1)):
        add_alert(symbol, period_label, "所有 EMA 多頭排列 🚀", "bull")

    vol_ma5 = vol.rolling(5).mean().iloc[-1]
    if vol.iloc[-1] > vol_ma5 * 2:
        add_alert(symbol, period_label, f"成交量暴增 {vol.iloc[-1]/vol_ma5:.1f}x 均量 📊", "vol")

    # 支撐/阻力突破警示（含週期參數 + 價格合理性過濾）
    itvl_key = {v[0]: k for k, v in INTERVAL_MAP.items()}.get(period_label, "1d")
    pivots_h, pivots_l = calc_pivot(df, interval=itvl_key)
    price      = float(close.iloc[-1])
    prev_price = float(close.iloc[-2]) if len(close) > 1 else price

    if pivots_h:
        # 取「剛被突破」的阻力位：prev <= resist < price（由下往上突破）
        broken = [p[1] for p in pivots_h if prev_price <= p[1] < price]
        if broken:
            add_alert(symbol, period_label, f"突破阻力位 ${max(broken):.2f} ⚡", "bull")

    if pivots_l:
        # 取「剛被跌破」的支撐位：price < support <= prev（由上往下跌破）
        broken = [p[1] for p in pivots_l if price < p[1] <= prev_price]
        if broken:
            add_alert(symbol, period_label, f"跌破支撐位 ${min(broken):.2f} ⚠️", "bear")

# ══════════════════════════════════════════════════════════════════════════════
# 建立 K 線圖
# ══════════════════════════════════════════════════════════════════════════════
def build_chart(symbol, df, interval_label, compact=False, max_bars=90):
    if df.empty: return None

    # ── 限制最多顯示 90 根 K 線，避免圖表擁擠 ──
    # EMA/MACD 用完整數據計算（保留歷史），再截取最後 90 根顯示
    MAX_BARS = max(10, int(max_bars))   # 使用者自訂，最少10根
    close_full, vol_full = df["Close"], df["Volume"]
    ema_s_full = {n: calc_ema(close_full, n) for n, _ in EMA_CONFIGS}
    ma_s_full  = {n: calc_ma(close_full,  n) for n, _, _ in MA_CONFIGS}
    dif_full, dea_full, hist_full = calc_macd(close_full)

    # 截取最後 90 根用於繪圖
    df   = df.tail(MAX_BARS).copy()
    close, vol = df["Close"], df["Volume"]
    ema_s = {n: s.tail(MAX_BARS) for n, s in ema_s_full.items()}
    ma_s  = {n: s.tail(MAX_BARS) for n, s in ma_s_full.items()}
    dif   = dif_full.tail(MAX_BARS)
    dea   = dea_full.tail(MAX_BARS)
    hist  = hist_full.tail(MAX_BARS)

    # 支撐阻力用截取後的資料
    itvl_code = {v[0]: k for k, v in INTERVAL_MAP.items()}.get(interval_label, "1d")
    pivots_h, pivots_l = calc_pivot(df, interval=itvl_code)

    # ── 消除休市空白：把 DatetimeIndex 轉成字串當 category label ──────────
    # Plotly category 軸只顯示實際存在的類別，自動跳過休市間隙
    intraday = interval_label in {"1分鐘","5分鐘","15分鐘","30分鐘"}
    fmt = "%m/%d %H:%M" if intraday else "%y/%m/%d"
    xlabels = [t.strftime(fmt) for t in df.index]
    # 所有 series 也配對成同樣的字串 index，確保對齊
    vol_ma5 = vol.rolling(5).mean()

    chart_h = 520 if compact else 820
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.56, 0.19, 0.25], vertical_spacing=0.02,
        subplot_titles=(f"{symbol} ({interval_label})", "成交量", "MACD"),
    )
    ann_size = 11 if compact else 13
    for ann in fig.layout.annotations:
        ann.font.size  = ann_size
        ann.font.color = "#ccddee"

    # K 線
    fig.add_trace(go.Candlestick(
        x=xlabels, open=df["Open"], high=df["High"], low=df["Low"], close=close,
        increasing_line_color="#00cc44", increasing_fillcolor="#00cc44",
        decreasing_line_color="#ff4444", decreasing_fillcolor="#ff4444",
        name="K線", showlegend=False,
    ), row=1, col=1)

    # EMA 線
    for n, color in EMA_CONFIGS:
        fig.add_trace(go.Scatter(
            x=xlabels, y=ema_s[n],
            line=dict(color=color, width=1.3), name=f"EMA{n}", opacity=0.9,
        ), row=1, col=1)

    # MA 線
    for n, color, dash in MA_CONFIGS:
        fig.add_trace(go.Scatter(
            x=xlabels, y=ma_s[n],
            line=dict(color=color, width=1.8, dash=dash), name=f"MA{n}",
        ), row=1, col=1)

    # 支撐阻力
    if pivots_h:
        r = max(p[1] for p in pivots_h)
        fig.add_hline(y=r, line=dict(color="#ff8888", dash="dash", width=1.5),
                      annotation_text=f"阻力 {r:.2f}",
                      annotation_font=dict(size=12, color="#ff8888"),
                      annotation_bgcolor="rgba(30,10,10,0.8)", row=1, col=1)
    if pivots_l:
        s = min(p[1] for p in pivots_l)
        fig.add_hline(y=s, line=dict(color="#88ff88", dash="dash", width=1.5),
                      annotation_text=f"支撐 {s:.2f}",
                      annotation_font=dict(size=12, color="#88ff88"),
                      annotation_bgcolor="rgba(10,30,10,0.8)", row=1, col=1)

    # 最高最低
    max_pos = int(df["High"].values.argmax())
    min_pos = int(df["Low"].values.argmin())
    fig.add_annotation(x=xlabels[max_pos], y=float(df["High"].max()),
        text=f"▲ {df['High'].max():.2f}", showarrow=True,
        arrowhead=2, arrowcolor="#ff4444", arrowwidth=2,
        font=dict(color="#ff8888", size=11, family="Arial Black"),
        bgcolor="rgba(30,10,10,0.85)", bordercolor="#ff4444", borderwidth=1,
        row=1, col=1)
    fig.add_annotation(x=xlabels[min_pos], y=float(df["Low"].min()),
        text=f"▼ {df['Low'].min():.2f}", showarrow=True,
        arrowhead=2, arrowcolor="#ff4444", arrowwidth=2,
        font=dict(color="#ff8888", size=11, family="Arial Black"),
        bgcolor="rgba(30,10,10,0.85)", bordercolor="#ff4444", borderwidth=1,
        row=1, col=1)

    # ── 成交量 ──────────────────────────────────────────────────────────────
    col_vol = ["#00cc44" if c >= o else "#ff4444"
               for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=xlabels, y=vol, marker_color=col_vol,
                         name="成交量", showlegend=False), row=2, col=1)
    vol_ma5 = vol.rolling(5).mean()
    fig.add_trace(go.Scatter(x=xlabels, y=vol_ma5,
                              line=dict(color="#ffaa00", width=1.5), name="VOL MA5"), row=2, col=1)

    # 異常放量：只標記「最顯著的幾根」，用柱子邊框高亮 + 頂部小鑽石
    # 策略：同一段密集放量只取最大的那根，避免連續出現滿屏標注
    anomaly_mask = (vol > vol_ma5 * 2).values
    if anomaly_mask.any():
        # 把連續異常段落找出來，每段只取量最大的那根
        groups, in_group, g_start = [], False, 0
        for i, flag in enumerate(anomaly_mask):
            if flag and not in_group:
                in_group, g_start = True, i
            elif not flag and in_group:
                groups.append((g_start, i - 1))
                in_group = False
        if in_group:
            groups.append((g_start, len(anomaly_mask) - 1))

        # 每段取量最大的 bar 的 integer position
        rep_pos = []
        for g0, g1 in groups:
            seg_vals = vol.values[g0:g1+1]
            rep_pos.append(g0 + int(seg_vals.argmax()))

        rep_x    = [xlabels[p]  for p in rep_pos]
        rep_vol  = [float(vol.values[p])    for p in rep_pos]
        rep_ma   = [float(vol_ma5.values[p]) if not np.isnan(vol_ma5.values[p]) else 1
                    for p in rep_pos]
        mult_txt = [f"異常放量 {v/max(m,1):.1f}x 均量"
                    for v, m in zip(rep_vol, rep_ma)]

        # 柱頂鑽石標記（不加擁擠文字，hover 查看倍數）
        fig.add_trace(go.Scatter(
            x=rep_x, y=rep_vol,
            mode="markers",
            marker=dict(color="#ff00ff", size=11, symbol="diamond",
                        line=dict(color="#ffffff", width=1.2)),
            name="異常放量",
            hovertext=mult_txt,
            hoverinfo="text+x",
        ), row=2, col=1)

    # ── MACD ────────────────────────────────────────────────────────────────
    bar_col = ["#ff4444" if v >= 0 else "#00cc44" for v in hist]
    fig.add_trace(go.Bar(x=xlabels, y=hist, marker_color=bar_col,
                         name="MACD柱", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=xlabels, y=dif,
                              line=dict(color="#ffaa00", width=1.5), name="DIF"), row=3, col=1)
    fig.add_trace(go.Scatter(x=xlabels, y=dea,
                              line=dict(color="#0088ff", width=1.5), name="DEA"), row=3, col=1)

    # ── 金叉/死叉（智能去擁擠）────────────────────────────────────────────
    # 收集所有原始交叉點
    raw_crosses = []
    for i in range(1, len(dif)):
        if dif.iloc[i] > dea.iloc[i] and dif.iloc[i-1] <= dea.iloc[i-1]:
            raw_crosses.append((i, "gold"))
        elif dif.iloc[i] < dea.iloc[i] and dif.iloc[i-1] >= dea.iloc[i-1]:
            raw_crosses.append((i, "dead"))

    # 間距過濾：相鄰標注至少 min_gap 根 K 線，且同方向連發只取最新
    total_bars = len(dif)
    min_gap    = max(6, total_bars // 20)
    max_labels = 3 if compact else 5

    filtered, last_pos, last_type = [], -9999, None
    for pos, ctype in reversed(raw_crosses):
        gap_ok  = (pos - last_pos) >= min_gap or last_pos == -9999
        diff_ok = (ctype != last_type) or last_pos == -9999
        if gap_ok and diff_ok:
            filtered.insert(0, (pos, ctype))
            last_pos, last_type = pos, ctype
        if len(filtered) >= max_labels:
            break

    # 繪製：金叉標在底部（ay 正值=往下偏移），死叉標在頂部（ay 負值=往上偏移）
    # 固定像素偏移，不依賴 MACD 數值範圍，確保 compact/full 都清晰
    base_px = 38 if compact else 46

    for seq, (pos, ctype) in enumerate(filtered):
        x_val  = xlabels[pos]
        y_val  = float(dif.iloc[pos])
        extra  = 1 + (seq % 2) * 0.45    # 偶數序號偏移更遠，水平錯開
        if ctype == "gold":
            ay_px  = int(base_px * extra)     # 正 = 箭頭朝上，標籤在下方
            text   = "⬆ 金叉"
            fcol, bgcol, bcol, acol = "#ffee55", "rgba(36,32,0,0.92)", "#bbaa00", "#ddcc00"
        else:
            ay_px  = -int(base_px * extra)    # 負 = 箭頭朝下，標籤在上方
            text   = "⬇ 死叉"
            fcol, bgcol, bcol, acol = "#ff9999", "rgba(36,0,0,0.92)", "#bb3333", "#cc4444"

        fig.add_annotation(
            x=x_val, y=y_val, text=text,
            showarrow=True, arrowhead=2, arrowwidth=1.5,
            ax=0, ay=ay_px,
            arrowcolor=acol,
            font=dict(color=fcol, size=9 if compact else 10, family="Arial Black"),
            bgcolor=bgcol, bordercolor=bcol, borderwidth=1, borderpad=3,
            row=3, col=1,
        )

    leg_sz = 8 if compact else 11

    # ── x 軸刻度標籤：依週期選擇合適格式 ─────────────────────────────────
    # 日K以下用日期+時間，日K及以上只用日期
    intraday_intervals = {"1分鐘","5分鐘","15分鐘","30分鐘"}
    if interval_label in intraday_intervals:
        tick_fmt = "%m/%d %H:%M"
        # 每隔幾根顯示一個刻度，避免密集
        n_ticks  = 8
    else:
        tick_fmt = "%Y/%m/%d"
        n_ticks  = 8

    # 用整數位置作為 x 軸刻度位置（category 模式下 x 軸是 0,1,2,...）
    total   = len(df)
    step    = max(1, total // n_ticks)
    tick_positions = list(range(0, total, step))
    tick_labels    = [df.index[i].strftime(tick_fmt) for i in tick_positions]

    fig.update_layout(
        height=chart_h, template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#111520",
        font=dict(family="Arial, sans-serif", size=10 if compact else 11, color="#ccddee"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            font=dict(size=leg_sz, color="#ddeeff"),
            bgcolor="rgba(14,17,23,0.85)", bordercolor="#2e3456", borderwidth=1,
            itemsizing="constant",
            traceorder="normal",
        ),
        margin=dict(l=6, r=6, t=36 if compact else 44, b=4),
        xaxis_rangeslider_visible=False,
        # category 類型：plotly 只顯示有數據的 bar，自動跳過休市空白
        xaxis_type="category",
        xaxis2_type="category",
        xaxis3_type="category",
    )

    # 套用自訂刻度到所有 x 軸
    for axis_name in ["xaxis", "xaxis2", "xaxis3"]:
        fig.update_layout(**{
            axis_name: dict(
                type="category",
                showgrid=True, gridcolor="#1a1e30",
                tickfont=dict(size=9 if compact else 10),
                tickmode="array",
                tickvals=tick_positions,
                ticktext=tick_labels,
                tickangle=-35,
            )
        })

    fig.update_yaxes(showgrid=True, gridcolor="#1a1e30",
                     tickfont=dict(size=9 if compact else 10))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# 多週期摘要列
# ══════════════════════════════════════════════════════════════════════════════
def render_mtf_summary(symbol, selected_intervals, show_alerts):
    st.markdown(f'<div class="mtf-section-title">🔀 多週期總覽 — {symbol}</div>',
                unsafe_allow_html=True)
    rows = []
    for itvl in selected_intervals:
        label, _ = INTERVAL_MAP[itvl]
        df = fetch_data(symbol, itvl)
        if df.empty:
            rows.append(
                f'<div class="mtf-header"><span class="mtf-period">{label}</span>'
                f'<span style="color:#555">數據載入失敗</span></div>')
            continue

        if show_alerts:
            run_alerts(symbol, label, df)

        close   = df["Close"]
        last    = float(close.iloc[-1])
        prev    = float(close.iloc[-2]) if len(close) > 1 else last
        chg     = last - prev
        pct     = chg / prev * 100 if prev else 0
        hi      = float(df["High"].iloc[-1])
        lo      = float(df["Low"].iloc[-1])
        vol_k   = int(df["Volume"].iloc[-1]) // 10000

        chg_cls   = "mtf-chg-up" if chg >= 0 else "mtf-chg-dn"
        chg_arrow = "▲" if chg >= 0 else "▼"

        trend     = detect_trend(df)
        t_cls     = {"多頭":"mtf-trend-bull","空頭":"mtf-trend-bear","盤整":"mtf-trend-side"}[trend]
        t_icon    = {"多頭":"▲","空頭":"▼","盤整":"◆"}[trend]

        macd_s    = get_macd_signal(df)
        macd_cls  = "mtf-macd-bull" if any(x in macd_s for x in ["金叉","↑"]) else "mtf-macd-bear"

        ema_s     = get_ema_signal(df)
        ema_cls   = "mtf-ema-bull" if any(x in ema_s for x in ["↑","多"]) else "mtf-ema-bear"

        rows.append(
            f'<div class="mtf-header">'
            f'  <span class="mtf-period">{label}</span>'
            f'  <div class="mtf-divider"></div>'
            f'  <span class="mtf-price">${last:.2f}</span>'
            f'  <span class="{chg_cls}">{chg_arrow} {chg:+.2f} ({pct:+.2f}%)</span>'
            f'  <div class="mtf-divider"></div>'
            f'  <span style="color:#6688aa;font-size:0.82rem">H:{hi:.2f}　L:{lo:.2f}　量:{vol_k}萬</span>'
            f'  <div class="mtf-divider"></div>'
            f'  <span class="{t_cls}">{t_icon} {trend}</span>'
            f'  <div class="mtf-divider"></div>'
            f'  <span class="{macd_cls}">MACD: {macd_s}</span>'
            f'  <span class="{ema_cls}">EMA: {ema_s}</span>'
            f'</div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 多週期 K 線圖
# ══════════════════════════════════════════════════════════════════════════════
def render_mtf_charts(symbol, selected_intervals, layout_mode, max_bars=90):
    if not selected_intervals:
        st.info("請至少選擇一個時間週期")
        return
    st.markdown(f'<div class="mtf-section-title">📊 多週期 K 線圖 — {symbol}</div>',
                unsafe_allow_html=True)

    if layout_mode == "並排（2欄）":
        pairs = [selected_intervals[i:i+2] for i in range(0, len(selected_intervals), 2)]
        for pair in pairs:
            cols = st.columns(len(pair))
            for col, itvl in zip(cols, pair):
                label, _ = INTERVAL_MAP[itvl]
                df = fetch_data(symbol, itvl)
                with col:
                    if df.empty:
                        st.error(f"{label} 無數據")
                    else:
                        fig = build_chart(symbol, df, label, compact=True, max_bars=max_bars)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True,
                                            config={"displayModeBar": False},
                                            key=f"mtf_{symbol}_{itvl}")
    else:
        for itvl in selected_intervals:
            label, _ = INTERVAL_MAP[itvl]
            df = fetch_data(symbol, itvl)
            if df.empty:
                st.error(f"{label} 無數據")
            else:
                fig = build_chart(symbol, df, label, compact=False, max_bars=max_bars)
                if fig:
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": True},
                                    key=f"mtf_{symbol}_{itvl}_full")

# ══════════════════════════════════════════════════════════════════════════════
# 單週期渲染
# ══════════════════════════════════════════════════════════════════════════════
def render_single(symbol, interval, show_alerts, max_bars=90):
    label, _ = INTERVAL_MAP[interval]
    with st.spinner(f"載入 {symbol} {label} 數據中..."):
        df = fetch_data(symbol, interval)

    if df.empty:
        st.error(f"❌ 無法取得 {symbol} 數據")
        return

    close   = df["Close"]
    last    = float(close.iloc[-1])
    prev    = float(close.iloc[-2]) if len(close) > 1 else last
    chg     = last - prev
    pct     = chg / prev * 100 if prev else 0
    vol_now = int(df["Volume"].iloc[-1])
    trend   = detect_trend(df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("最新價格",      f"${last:.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
    c2.metric("成交量（萬股）", f"{vol_now/10000:.1f}")
    c3.metric("本K最高",       f"${df['High'].iloc[-1]:.2f}")
    c4.metric("本K最低",       f"${df['Low'].iloc[-1]:.2f}")
    t_cls  = {"多頭":"trend-bull","空頭":"trend-bear","盤整":"trend-side"}[trend]
    t_icon = {"多頭":"▲","空頭":"▼","盤整":"◆"}[trend]
    with c5:
        st.markdown(
            f'<div class="trend-card"><div class="trend-title">趨勢判斷</div>'
            f'<div class="{t_cls}">{t_icon} {trend}</div></div>',
            unsafe_allow_html=True)

    # EMA 列
    items = []
    for n, color in EMA_CONFIGS:
        val   = float(calc_ema(close,n).iloc[-1])
        arrow = "↑" if last > val else "↓"
        items.append(
            f'<span class="ema-item" style="color:{color}">'
            f'<span class="ema-label">EMA{n} </span>{val:.2f}'
            f'<span style="font-size:0.72rem;opacity:0.6"> {arrow}</span></span>')
    st.markdown('<div class="ema-bar">' + "".join(items) + '</div>',
                unsafe_allow_html=True)

    fig = build_chart(symbol, df, label, max_bars=max_bars)
    if fig:
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": True},
                        key=f"single_{symbol}_{interval}")

    if show_alerts:
        run_alerts(symbol, label, df)

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📈 美股監控系統")
    st.markdown("---")

    raw_input = st.text_area("股票代號（逗號分隔）", value="TSLA,AAPL,NVDA", height=80)
    symbols   = [s.strip().upper() for s in raw_input.replace("，",",").split(",") if s.strip()]

    st.markdown("---")
    st.markdown("#### 📅 監控模式")
    mode = st.radio("", ["單一週期", "多週期同時監控"], horizontal=True,
                    label_visibility="collapsed")

    if mode == "單一週期":
        single_interval = st.selectbox(
            "時間週期",
            ALL_INTERVALS,
            format_func=lambda x: INTERVAL_LABELS[x],
            index=4,
        )
        layout_mode = None
        selected    = []

    else:
        st.markdown("**勾選要同時顯示的週期：**")
        selected    = []
        defaults    = {"5m", "15m", "1d"}
        left_col, right_col = st.columns(2)
        for i, itvl in enumerate(ALL_INTERVALS):
            col = left_col if i % 2 == 0 else right_col
            if col.checkbox(INTERVAL_LABELS[itvl], value=(itvl in defaults), key=f"cb_{itvl}"):
                selected.append(itvl)
        st.markdown("")
        layout_mode = st.radio("圖表排列方式",
                               ["並排（2欄）", "堆疊（全寬）"], horizontal=True)

    st.markdown("---")
    auto_refresh = st.toggle("自動刷新", value=False)
    refresh_sec  = st.slider("刷新間隔（秒）", 60, 300, 60, step=30, disabled=not auto_refresh)

    st.markdown("---")
    st.markdown("**📊 K 線顯示根數**")
    max_bars = st.number_input(
        "每張圖最多顯示幾根 K 線",
        min_value=20, max_value=500, value=90, step=10,
        help="建議：分鐘圖 60-120 根，日K 60-90 根，週K/月K 40-60 根",
    )

    st.markdown("---")
    show_alerts = st.toggle("啟用警示偵測", value=True)

    if st.button("🗑️ 清除警示記錄"):
        st.session_state.alert_log   = []
        st.session_state.sent_alerts = set()
        st.toast("警示記錄已清除")

    if st.session_state.alert_log:
        csv_data = pd.DataFrame(st.session_state.alert_log).to_csv(
            index=False, encoding="utf-8-sig")
        st.download_button("📥 匯出警示 CSV", csv_data, "alerts.csv", "text/csv")

    st.markdown("---")
    st.caption("數據來源：Yahoo Finance\n\n⚠️ 僅供參考，不構成投資建議")

# ══════════════════════════════════════════════════════════════════════════════
# 主區域
# ══════════════════════════════════════════════════════════════════════════════
st.title("🇺🇸 美股即時監控系統")

if not symbols:
    st.info("請在左側輸入股票代號")
    st.stop()

stock_tabs = st.tabs([f"📊 {s}" for s in symbols])

for tab, symbol in zip(stock_tabs, symbols):
    with tab:
        if mode == "單一週期":
            render_single(symbol, single_interval, show_alerts, max_bars=max_bars)

        else:
            if not selected:
                st.warning("⚠️ 請在左側至少勾選一個時間週期")
            else:
                # ① 多週期摘要
                render_mtf_summary(symbol, selected, show_alerts)
                st.markdown("---")
                # ② 多週期 K 線圖
                render_mtf_charts(symbol, selected, layout_mode, max_bars=max_bars)

# ══════════════════════════════════════════════════════════════════════════════
# 警示面板
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.alert_log:
    st.markdown("---")
    st.subheader("🔔 警示訊息記錄")
    cls_map = {"bull":"alert-bull","bear":"alert-bear","vol":"alert-vol","info":"alert-info"}
    for e in st.session_state.alert_log[:40]:
        cls    = cls_map.get(e["類型"], "alert-info")
        p_tag  = f'【{e["週期"]}】' if e.get("週期") else ""
        st.markdown(
            f'<div class="alert-box {cls}">'
            f'🕐 {e["時間"]}　【{e["股票"]}】{p_tag}　{e["訊息"]}'
            f'</div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 自動刷新
# ══════════════════════════════════════════════════════════════════════════════
if auto_refresh:
    time.sleep(refresh_sec)
    st.cache_data.clear()
    st.rerun()
