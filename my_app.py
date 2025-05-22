import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------------- Utility Functions ----------------

def get_stock_data(symbol, timeframe):
    interval = "1d" if timeframe == "Daily" else "1h"
    period = "6mo" if timeframe == "Daily" else "7d"
    df = yf.download(symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    print(df)
    return df

# def find_support_resistance1(df, order=5):
#     local_min = argrelextrema(df['Low'].values, np.less_equal, order=order)[0]
#     local_max = argrelextrema(df['High'].values, np.greater_equal, order=order)[0]
#     support_levels = df.iloc[local_min][['Low']].rename(columns={'Low': 'Support'})
#     resistance_levels = df.iloc[local_max][['High']].rename(columns={'High': 'Resistance'})
#     return support_levels, resistance_levels
def find_support_resistance(df, order=5):
    lows = df['Low'].values
    highs = df['High'].values

    local_min = argrelextrema(lows, np.less_equal, order=order)[0]
    local_max = argrelextrema(highs, np.greater_equal, order=order)[0]

    support_levels = pd.DataFrame({'Support': lows[local_min].flatten()}, index=df.index[local_min])
    resistance_levels = pd.DataFrame({'Resistance': highs[local_max].flatten()}, index=df.index[local_max])

    return support_levels, resistance_levels
    
def find_support_resistance(df, order=5):
    lows = df['Low'].values
    highs = df['High'].values
    local_min = argrelextrema(lows, np.less_equal, order=order)[0]
    local_max = argrelextrema(highs, np.greater_equal, order=order)[0]
    support_levels = pd.DataFrame(index=df.index[local_min], data={'Support': lows[local_min]})
    resistance_levels = pd.DataFrame(index=df.index[local_max], data={'Resistance': highs[local_max]})
    return support_levels, resistance_levels


def calculate_trendlines(df):
    trendlines = []

    for i in range(len(df) - 1):
        for j in range(i + 1, len(df)):
            x1, y1 = i, df['Close'].iloc[i]
            x2, y2 = j, df['Close'].iloc[j]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            line = slope * np.arange(len(df)) + intercept
            if all(df['Low'].values[k] <= line[k] for k in range(i, j + 1)):
                trendlines.append(("support", i, j, y1, y2))
            elif all(df['High'].values[k] >= line[k] for k in range(i, j + 1)):
                trendlines.append(("resistance", i, j, y1, y2))

    return trendlines[-2:] if trendlines else []

def plot_chart(df, support, resistance, trendlines):
    fig = go.Figure()

    # Price Candlesticks
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Price'))

    # Support levels
    for idx, row in support.iterrows():
        fig.add_hline(y=row['Support'], line_dash="dot", line_color="green",
                      annotation_text=f"Support: {row['Support']:.2f}", annotation_position="top left")

    # Resistance levels
    for idx, row in resistance.iterrows():
        fig.add_hline(y=row['Resistance'], line_dash="dot", line_color="red",
                      annotation_text=f"Resistance: {row['Resistance']:.2f}", annotation_position="bottom left")

    # Trendlines
    for kind, i, j, y1, y2 in trendlines:
        fig.add_trace(go.Scatter(
            x=[df.index[i], df.index[j]],
            y=[y1, y2],
            mode='lines',
            line=dict(color='blue' if kind == "support" else "orange", width=2),
            name=f"{kind.title()} Trendline"
        ))

    fig.update_layout(title="Price Action Analysis",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      xaxis_rangeslider_visible=False)
    return fig

# ---------------- Streamlit UI ----------------

st.set_page_config(layout="wide")
st.title("📈 Stock Price Action Dashboard")

# ---------------- Streamlit UI ----------------

with st.form("stock_form"):
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA):", value="DIVISLAB.NS")
    timeframe = st.selectbox("Select Timeframe:", ["Daily", "Hourly"])
    submitted = st.form_submit_button("Submit")

if submitted:
    try:
        df = get_stock_data(symbol, timeframe)
        print(df)
        support, resistance = find_support_resistance(df)
        print(support)
        trendlines = calculate_trendlines(df)
        fig = plot_chart(df, support, resistance, trendlines)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
