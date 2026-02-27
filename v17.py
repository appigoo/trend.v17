import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
import time
import io
from scipy.signal import argrelextrema

# 設定頁面配置
st.set_page_config(page_title="美股實時監控系統", layout="wide")

# 標題
st.title("美股實時監控系統")

# 側邊欄
st.sidebar.header("設定")
symbols_input = st.sidebar.text_input("股票代號（多個用逗號分隔，如 TSLA,NIO）", value="TSLA")
interval = st.sidebar.selectbox("時間週期", options=["1m", "5m", "15m", "30m", "1d", "1wk", "1mo"])
auto_refresh = st.sidebar.checkbox("自動刷新", value=True)
refresh_interval = st.sidebar.slider("刷新間隔（秒）", min_value=60, max_value=300, value=60)

# 額外參數
n_pivots = 100  # 用於支撐阻力的近N根K線
vol_multiplier = 2  # 成交量放大倍數
vol_ma_period = 5  # 均量期數

# Telegram 設定
telegram_token = st.secrets.get("TELEGRAM_BOT_TOKEN", None)
telegram_chat_id = st.secrets.get("TELEGRAM_CHAT_ID", None)

def send_telegram_message(message):
    if telegram_token and telegram_chat_id:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        payload = {"chat_id": telegram_chat_id, "text": message}
        try:
            requests.post(url, json=payload)
        except:
            pass

# 警示紀錄
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []
if "last_alerts" not in st.session_state:
    st.session_state.last_alerts = {}

# 解析多股票
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

# 趨勢判斷函數
def determine_trend(emas):
    if all(emas.diff().dropna() > 0):
        return "多頭"
    elif all(emas.diff().dropna() < 0):
        return "空頭"
    else:
        return "盤整"

# 主循環
tabs = st.tabs(symbols) if len(symbols) > 1 else [st.container()]

for i, symbol in enumerate(symbols):
    with tabs[i] if len(symbols) > 1 else tabs[0]:
        st.subheader(f"{symbol} 實時數據")

        # 獲取數據
        ticker = yf.Ticker(symbol)
        if interval in ["1m", "5m", "15m", "30m"]:
            period = "5d" if interval == "1m" else "30d"
        elif interval == "1d":
            period = "1y"
        elif interval == "1wk":
            period = "5y"
        else:
            period = "10y"
        df = ticker.history(period=period, interval=interval)
        df = df.dropna()

        if df.empty:
            st.error("無法獲取數據，請檢查股票代號。")
            continue

        # 計算指標
        # EMA
        ema_periods = [5, 10, 20, 30, 40, 60, 120, 200]
        ema_colors = ["green", "yellow", "orange", "red", "purple", "blue", "cyan", "indigo"]
        for p, color in zip(ema_periods, ema_colors):
            df[f"EMA{p}"] = df["Close"].ewm(span=p, adjust=False).mean()

        # MA
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA15"] = df["Close"].rolling(15).mean()

        # MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["DIF"] = ema12 - ema26
        df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
        df["MACD"] = df["DIF"] - df["DEA"]

        # 支撐阻力
        # 使用 argrelextrema 找近 n_pivots 根的局部高低點
        recent_df = df.tail(n_pivots)
        high_idx = argrelextrema(recent_df["High"].values, np.greater, order=5)[0]
        low_idx = argrelextrema(recent_df["Low"].values, np.less, order=5)[0]
        resistances = recent_df.iloc[high_idx]["High"].mean() if len(high_idx) > 0 else np.nan
        supports = recent_df.iloc[low_idx]["Low"].mean() if len(low_idx) > 0 else np.nan

        # 成交量
        df["VOL_MA5"] = df["Volume"].rolling(vol_ma_period).mean()

        # 最新數據
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        # 警示檢查
        alerts = []
        key = symbol

        # MACD 金叉/死叉
        if prev["DIF"] < prev["DEA"] and latest["DIF"] > latest["DEA"]:
            alerts.append("MACD 金叉")
        elif prev["DIF"] > prev["DEA"] and latest["DIF"] < latest["DEA"]:
            alerts.append("MACD 死叉")

        # 價格突破
        if not np.isnan(resistances) and prev["Close"] < resistances and latest["Close"] > resistances:
            alerts.append("價格突破阻力位")
        if not np.isnan(supports) and prev["Close"] > supports and latest["Close"] < supports:
            alerts.append("價格跌破支撐位")

        # 成交量暴增
        if latest["Volume"] > vol_multiplier * latest["VOL_MA5"]:
            alerts.append("成交量暴增")

        # EMA5 穿 EMA20
        if prev["EMA5"] < prev["EMA20"] and latest["EMA5"] > latest["EMA20"]:
            alerts.append("EMA5 上穿 EMA20（多頭排列開始）")
        elif prev["EMA5"] > prev["EMA20"] and latest["EMA5"] < latest["EMA20"]:
            alerts.append("EMA5 下穿 EMA20（空頭排列開始）")

        # 所有 EMA 多頭排列
        emas = latest[[f"EMA{p}" for p in ema_periods]]
        if all(emas.diff() > 0):
            alerts.append("所有 EMA 多頭排列")

        # 發送警示並記錄
        for alert in alerts:
            if st.session_state.last_alerts.get(f"{key}_{alert}", 0) < time.time() - 3600:  # 每小時一次
                message = f"{symbol} 警示: {alert}"
                send_telegram_message(message)
                st.session_state.alert_log.append({"時間": pd.Timestamp.now(), "股票": symbol, "警示": alert})
                st.session_state.last_alerts[f"{key}_{alert}"] = time.time()

        # 趨勢
        trend = determine_trend(emas)

        # 顯示 EMA 數值
        st.subheader("EMA 均線數值")
        ema_values = ", ".join([f"{p}: {latest[f'EMA{p}']:.2f} ({color})" for p, color in zip(ema_periods, ema_colors)])
        st.write(ema_values)

        # K線圖
        fig_candles = go.Figure()
        fig_candles.add_trace(go.Candlestick(x=df.index,
                                             open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                             name="K線", increasing_line_color="red", decreasing_line_color="green"))

        # 疊加 EMA 和 MA
        for p, color in zip(ema_periods, ema_colors):
            fig_candles.add_trace(go.Scatter(x=df.index, y=df[f"EMA{p}"], mode="lines", name=f"EMA{p}", line=dict(color=color)))
        fig_candles.add_trace(go.Scatter(x=df.index, y=df["MA5"], mode="lines", name="MA5", line=dict(color="black")))
        fig_candles.add_trace(go.Scatter(x=df.index, y=df["MA15"], mode="lines", name="MA15", line=dict(color="gray")))

        # 標記最高最低
        max_price = df["High"].max()
        min_price = df["Low"].min()
        fig_candles.add_annotation(x=df["High"].idxmax(), y=max_price, text=f"最高: {max_price:.2f}", showarrow=True)
        fig_candles.add_annotation(x=df["Low"].idxmin(), y=min_price, text=f"最低: {min_price:.2f}", showarrow=True)

        # 支撐阻力線
        if not np.isnan(resistances):
            fig_candles.add_hline(y=resistances, line_dash="dash", line_color="red", annotation_text="阻力")
        if not np.isnan(supports):
            fig_candles.add_hline(y=supports, line_dash="dash", line_color="green", annotation_text="支撐")

        fig_candles.update_layout(title=f"{symbol} K線圖", xaxis_title="時間", yaxis_title="價格", height=500)
        st.plotly_chart(fig_candles, use_container_width=True)

        # 成交量圖
        colors = ["red" if o < c else "green" for o, c in zip(df["Open"], df["Close"])]
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=df.index, y=df["Volume"], name="成交量", marker_color=colors))
        fig_vol.add_trace(go.Scatter(x=df.index, y=df["VOL_MA5"], mode="lines", name="VOL MA5", line=dict(color="blue")))

        # 標記異常放量
        abnormal = df[df["Volume"] > vol_multiplier * df["VOL_MA5"]]
        fig_vol.add_trace(go.Scatter(x=abnormal.index, y=abnormal["Volume"], mode="markers", name="異常放量", marker=dict(color="orange", size=10)))

        fig_vol.update_layout(title="成交量圖", xaxis_title="時間", yaxis_title="成交量（股）", height=300)
        st.plotly_chart(fig_vol, use_container_width=True)

        # MACD 圖
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["DIF"], mode="lines", name="DIF", line=dict(color="blue")))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["DEA"], mode="lines", name="DEA", line=dict(color="orange")))
        bar_colors = ["red" if v > 0 else "green" for v in df["MACD"]]
        fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD"], name="MACD", marker_color=bar_colors))

        # 標記金叉死叉
        crosses = np.where((df["DIF"].shift(1) < df["DEA"].shift(1)) & (df["DIF"] > df["DEA"]))[0]
        for idx in crosses:
            fig_macd.add_annotation(x=df.index[idx], y=df["MACD"][idx], text="金叉", showarrow=True)
        crosses = np.where((df["DIF"].shift(1) > df["DEA"].shift(1)) & (df["DIF"] < df["DEA"]))[0]
        for idx in crosses:
            fig_macd.add_annotation(x=df.index[idx], y=df["MACD"][idx], text="死叉", showarrow=True)

        fig_macd.update_layout(title="MACD 圖", xaxis_title="時間", yaxis_title="值", height=300)
        st.plotly_chart(fig_macd, use_container_width=True)

        # 警示面板
        st.subheader("警示訊息")
        for alert in alerts:
            st.warning(alert)

        # 目前趨勢
        st.subheader("目前趨勢")
        st.write(trend)

        # 最新 VOL
        st.write(f"當前成交量: {latest['Volume'] / 10000:.2f} 萬股")

# 底部：匯出警示紀錄
if st.session_state.alert_log:
    st.header("警示紀錄")
    log_df = pd.DataFrame(st.session_state.alert_log)
    csv = log_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("匯出 CSV", data=csv, file_name="alert_log.csv", mime="text/csv")

# 自動刷新
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
