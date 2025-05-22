import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# App title
st.title("Stock Price Action Analysis")
st.sidebar.header("Stock Settings")

# User inputs
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
timeframe = st.sidebar.selectbox("Select Timeframe", ["Daily", "Hourly"])

# Fetch stock data
if timeframe == "Daily":
    data = yf.download(ticker, period="60d", interval="1d")
else:
    data = yf.download(ticker, period="7d", interval="60m")

data.dropna(inplace=True)
st.write(data)
# Identify pivot points for support & resistance
def find_pivots(df, window=5):
    supports, resistances = [], []
    for i in range(window, len(df) - window):
        current_low = df["Low"].iloc[i]
        current_high = df["High"].iloc[i]
        window_low = df["Low"].iloc[i - window : i + window + 1].min()
        window_high = df["High"].iloc[i - window : i + window + 1].max()
        if np.isclose(current_low, window_low).any():
            supports.append((df.index[i], current_low))
        if np.isclose(current_high, window_high).any():
            resistances.append((df.index[i], current_high))
    return supports, resistances

supports, resistances = find_pivots(data)

# Compute trend lines
def compute_trend_line(pivots, total_points):
    if len(pivots) < 2:
        return None
    x = np.array([data.index.get_loc(time) for time, price in pivots])
    y = np.array([price for time, price in pivots])
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.arange(total_points)
    y_line = slope * x_line + intercept
    return y_line

support_trend = compute_trend_line(supports, len(data))
resistance_trend = compute_trend_line(resistances, len(data))

# Plot chart
fig = go.Figure()
fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name="Price"))

# Add support and resistance levels
for time, price in supports:
    fig.add_hline(y=price, line_dash="dash", line_color="green", annotation_text="Support")
for time, price in resistances:
    fig.add_hline(y=price, line_dash="dash", line_color="red", annotation_text="Resistance")

# Plot trend lines if available
if support_trend is not None:
    fig.add_trace(go.Scatter(x=data.index, y=support_trend, mode="lines", name="Support Trend", line=dict(color="blue")))
if resistance_trend is not None:
    fig.add_trace(go.Scatter(x=data.index, y=resistance_trend, mode="lines", name="Resistance Trend", line=dict(color="orange")))

fig.update_layout(title=f"{ticker} Price Action Analysis ({timeframe})", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False)

# Display chart
st.plotly_chart(fig, use_container_width=True)
