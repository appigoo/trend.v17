import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import requests
import time

# --- 頁面設定 ---
st.set_page_config(page_title="美股實時監控系統", layout="wide")

# --- Telegram 通知功能 ---
def send_telegram_msg(message):
    try:
        token = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}"
        requests.get(url)
    except:
        pass # 若未設定 secrets 則跳過

# --- 數據抓取 ---
@st.cache_data(ttl=60)
def fetch_data(symbol, interval, period="1mo"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    return df

# --- 技術指標計算 ---
def calculate_indicators(df):
    # EMA 計算
    ema_periods = [5, 10, 20, 30, 40, 60, 120, 200]
    for p in range(len(ema_periods)):
        df[f'EMA{ema_periods[p]}'] = ta.ema(df['Close'], length=ema_periods[p])
    
    # MA 計算
    df['MA5'] = ta.sma(df['Close'], length=5)
    df['MA15'] = ta.sma(df['Close'], length=15)
    
    # MACD 計算
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    
    # 支撐與阻力 (Pivot Point)
    window = 10
    df['Support'] = df['Low'].rolling(window=window, center=True).min()
    df['Resistance'] = df['High'].rolling(window=window, center=True).max()
    
    return df

# --- 警示邏輯 ---
def check_alerts(df, symbol):
    alerts = []
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. MACD 金叉/死叉
    if prev['MACD_12_26_9'] < prev['MACDs_12_26_9'] and curr['MACD_12_26_9'] > curr['MACDs_12_26_9']:
        alerts.append(f"🚀 {symbol} MACD 金叉 (多頭)")
    elif prev['MACD_12_26_9'] > prev['MACDs_12_26_9'] and curr['MACD_12_26_9'] < curr['MACDs_12_26_9']:
        alerts.append(f"⚠️ {symbol} MACD 死叉 (空頭)")
        
    # 2. 支撐阻力突破
    if curr['Close'] > prev['Resistance']:
        alerts.append(f"📈 {symbol} 突破阻力位!")
    elif curr['Close'] < prev['Support']:
        alerts.append(f"📉 {symbol} 跌破支撐位!")
        
    # 3. 成交量暴增
    vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
    if curr['Volume'] > vol_ma5 * 2:
        alerts.append(f"📊 {symbol} 成交量異常放量 ({round(curr['Volume']/10000, 2)} 萬股)")
        
    # 4. EMA 交叉
    if prev['EMA5'] < prev['EMA20'] and curr['EMA5'] > curr['EMA20']:
        alerts.append(f"⚡ {symbol} EMA5 上穿 EMA20 (趨勢轉強)")
        
    # 5. 多頭排列
    ema_list = [curr['EMA5'], curr['EMA10'], curr['EMA20'], curr['EMA30'], curr['EMA40'], curr['EMA60']]
    if all(x > y for x, y in zip(ema_list, ema_list[1:])):
        alerts.append(f"💎 {symbol} 達成全 EMA 多頭排列!")

    return alerts

# --- 主介面 ---
st.title("📈 美股實時技術監控系統")

# Sidebar
with st.sidebar:
    st.header("控制面板")
    symbol_input = st.text_input("輸入股票代號 (多個請用逗號隔開)", "TSLA, NVDA, AAPL")
    interval = st.selectbox("時間週期", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"], index=5)
    refresh_rate = st.slider("自動刷新間隔 (秒)", 60, 300, 60)
    auto_refresh = st.checkbox("開啟自動刷新", value=False)

symbols = [s.strip().upper() for s in symbol_input.split(",")]
tabs = st.tabs(symbols)

for i, symbol in enumerate(symbols):
    with tabs[i]:
        try:
            # 獲取與計算數據
            df = fetch_data(symbol, interval)
            df = calculate_indicators(df)
            
            # 趨勢判斷
            curr = df.iloc[-1]
            ema_trend = "多頭" if curr['EMA5'] > curr['EMA20'] else "空頭" if curr['EMA5'] < curr['EMA20'] else "盤整"
            
            # --- 顯示資訊卡 ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("當前價格", f"${curr['Close']:.2f}", f"{curr['Close']-df.iloc[-2]['Close']:.2f}")
            c2.metric("成交量", f"{curr['Volume']/10000:.1f} 萬股")
            c3.metric("目前趨勢", ema_trend)
            c4.metric("阻力 / 支撐", f"{curr['Resistance']:.1f} / {curr['Support']:.1f}")

            # --- Plotly 圖表 ---
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.6, 0.15, 0.25])

            # 1. K線圖 (台股配色：紅漲綠跌)
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'], name="K線",
                                         increasing_line_color='#FF3232', decreasing_line_color='#00AB5E'), row=1, col=1)

            # 疊加 EMA
            ema_colors = ['#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#800080', '#0000FF', '#00FFFF', '#A020F0']
            for idx, p in enumerate([5, 10, 20, 30, 40, 60, 120, 200]):
                fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA{p}'], name=f'EMA{p}', 
                                         line=dict(color=ema_colors[idx], width=1)), row=1, col=1)
            
            # 支撐阻力線 (虛線)
            fig.add_hline(y=curr['Resistance'], line_dash="dash", line_color="red", annotation_text="阻力", row=1, col=1)
            fig.add_hline(y=curr['Support'], line_dash="dash", line_color="green", annotation_text="支撐", row=1, col=1)

            # 2. 成交量
            colors = ['#FF3232' if row['Open'] < row['Close'] else '#00AB5E' for index, row in df.iterrows()]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="成交量", marker_color=colors), row=2, col=1)

            # 3. MACD
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], name="DIF", line=dict(color='white')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], name="DEA", line=dict(color='yellow')), row=3, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['MACDh_12_26_9'], name="MACD柱"), row=3, col=1)

            fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # --- 警示訊息 ---
            st.subheader("🔔 實時警示紀錄")
            new_alerts = check_alerts(df, symbol)
            if new_alerts:
                for a in new_alerts:
                    st.warning(a)
                    if st.sidebar.button(f"發送通知: {symbol}", key=f"btn_{symbol}_{time.time()}"):
                        send_telegram_msg(a)
            else:
                st.write("目前無觸發信號")

        except Exception as e:
            st.error(f"無法獲取 {symbol} 的數據: {e}")

# --- 自動刷新 ---
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
