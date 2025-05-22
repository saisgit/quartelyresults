import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📈 Price Action Stock Analyzer")

# Sidebar Inputs
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
timeframe = st.sidebar.selectbox("Select Timeframe", ["Daily", "Hourly"])
lookback_days = st.sidebar.slider("Lookback Period (days)", 30, 365, 90)

# Download Data
interval = '1d' if timeframe == "Daily" else '1h'
start = datetime.now() - timedelta(days=lookback_days)
df = yf.download(symbol, start=start, interval=interval)

if df.empty:
    st.error("No data found. Check the symbol or timeframe.")
    st.stop()

df.dropna(inplace=True)

# ---- Price Action Logic ----

def find_support_resistance(df, window=3):
    supports = []
    resistances = []

    for i in range(window, len(df) - window):
        is_support = all(df['Low'].iloc[i] < df['Low'].iloc[i - j] and df['Low'].iloc[i] < df['Low'].iloc[i + j] for j in range(1, window + 1))
        is_resistance = all(df['High'].iloc[i] > df['High'].iloc[i - j] and df['High'].iloc[i] > df['High'].iloc[i + j] for j in range(1, window + 1))

        if is_support:
            supports.append((df.index[i], df['Low'].iloc[i]))
        if is_resistance:
            resistances.append((df.index[i], df['High'].iloc[i]))

    return supports, resistances

def find_trendlines(supports, resistances):
    def linear_fit(points):
        x = np.array([(p[0] - points[0][0]).days if isinstance(p[0], pd.Timestamp) else (p[0] - points[0][0]).total_seconds()/3600 for p in points])
        y = np.array([p[1] for p in points])
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, c, x

    support_trendline = linear_fit(supports[-3:]) if len(supports) >= 3 else None
    resistance_trendline = linear_fit(resistances[-3:]) if len(resistances) >= 3 else None
    return support_trendline, resistance_trendline

supports, resistances = find_support_resistance(df)
support_trend, resistance_trend = find_trendlines(supports, resistances)

# ---- Plotting ----

fig = go.Figure()

# Candlesticks
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="Candlestick"
))

# Support Zones
for ts, level in supports:
    fig.add_shape(type="line", x0=ts, y0=level, x1=df.index[-1], y1=level,
                  line=dict(color="green", width=1, dash="dot"))
    fig.add_annotation(x=ts, y=level, text="Support", showarrow=False, yshift=10, font=dict(color="green"))

# Resistance Zones
for ts, level in resistances:
    fig.add_shape(type="line", x0=ts, y0=level, x1=df.index[-1], y1=level,
                  line=dict(color="red", width=1, dash="dot"))
    fig.add_annotation(x=ts, y=level, text="Resistance", showarrow=False, yshift=-10, font=dict(color="red"))

# Trendlines
if support_trend:
    m, c, x_vals = support_trend
    trend_y = m * np.arange(len(df)) + c
    fig.add_trace(go.Scatter(x=df.index, y=trend_y, line=dict(color="green", dash="dash"), name="Support Trendline"))

if resistance_trend:
    m, c, x_vals = resistance_trend
    trend_y = m * np.arange(len(df)) + c
    fig.add_trace(go.Scatter(x=df.index, y=trend_y, line=dict(color="red", dash="dash"), name="Resistance Trendline"))

fig.update_layout(title=f"{symbol.upper()} - {timeframe} Chart with Price Action Analysis",
                  xaxis_title="Date",
                  yaxis_title="Price",
                  height=800)

st.plotly_chart(fig, use_container_width=True)
