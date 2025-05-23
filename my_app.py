import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta # For technical indicators and patterns
from scipy.signal import savgol_filter
import numpy as np
import urllib
from datetime import datetime, timedelta
# --- Configuration ---
st.set_page_config(layout="wide", page_title="Advanced Technical Analysis Dashboard")

# --- Helper Functions ---

#@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_stock_data(ticker, period, interval):
    """Fetches stock data from Yahoo Finance."""
    try:
        #data = yf.download(ticker, period=period, interval=interval)
        try:
            proxyServer = urllib.request.getproxies()['http']
        except KeyError:
            proxyServer = ""
        tday = datetime.strptime('23052025', '%d%m%Y').date() + timedelta(days=1)
        startDay = tday - timedelta(days=period)
        data = yf.download(
                tickers=ticker+".NS", start=startDay, end=tday,
                # period=period,
                interval=interval,
                proxy=proxyServer,
                progress=False,
                multi_level_index = False,
                timeout=10
            )
        #st.write(data)
        if data.empty:
            st.error(f"No data found for {ticker} with period {period} and interval {interval}. Please check the ticker symbol or selected period/interval.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}. Please ensure the ticker is valid.")
        return None

def detect_support_resistance(df, window=10):
    """
    Detects support and resistance levels using a simplified method.
    This method looks for local minima/maxima within a window.
    More advanced methods might involve pivot points, Fibonacci, etc.
    """
    if df.empty or len(df) < window * 2:
        return [], []

    # Apply a smoothing filter
    # Adjust polynomial order based on data length, but keeping it simple for now
    if len(df) > 50: # Ensure enough points for smoothing
        smoothed_close = savgol_filter(df['Close'], window * 2 - 1 if (window * 2 - 1) % 2 == 1 else window * 2, 3)
    else:
        smoothed_close = df['Close'].values

    local_minima_indices = []
    local_maxima_indices = []

    for i in range(window, len(smoothed_close) - window):
        # Check for local minimum
        if all(smoothed_close[i] < smoothed_close[j] for j in range(i - window, i)) and \
           all(smoothed_close[i] < smoothed_close[j] for j in range(i + 1, i + window + 1)):
            local_minima_indices.append(i)

        # Check for local maximum
        if all(smoothed_close[i] > smoothed_close[j] for j in range(i - window, i)) and \
           all(smoothed_close[i] > smoothed_close[j] for j in range(i + 1, i + window + 1)):
            local_maxima_indices.append(i)

    supports = [df['Low'].iloc[i] for i in local_minima_indices]
    resistances = [df['High'].iloc[i] for i in local_maxima_indices]

    return supports, resistances

def plot_candlestick_with_indicators(df, ticker, timeframe, show_patterns=False, supports=[], resistances=[]):
    """Plots candlestick chart with technical indicators and patterns."""

    if df.empty:
        st.warning("No data to plot.")
        return

    # Create subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.07,
                        row_heights=[0.6, 0.2, 0.2])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Candlestick'), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='rgba(0,128,0,0.5)'), row=2, col=1)

    # Add Moving Averages (example: SMA 20, 50)
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='blue', width=1)), row=1, col=1)
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', width=1)), row=1, col=1)

    # Add Bollinger Bands
    if 'BBL_5_2.0' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BBL_5_2.0'], mode='lines', name='BB Lower', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BBM_5_2.0'], mode='lines', name='BB Middle', line=dict(color='purple', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BBU_5_2.0'], mode='lines', name='BB Upper', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)

    # Add RSI
    if 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], mode='lines', name='RSI', line=dict(color='green', width=1)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, annotation_text="Overbought", annotation_position="top left")
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, annotation_text="Oversold", annotation_position="bottom left")

    # Add MACD
    # if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns and 'MACDh_12_26_9' in df.columns:
    #     fig.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], mode='lines', name='MACD', line=dict(color='fuchsia', width=1)), row=3, col=1)
    #     fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], mode='lines', name='Signal', line=dict(color='red', width=1)), row=3, col=1)
    #     fig.add_trace(go.Bar(x=df.index, y=df['MACDh_12_26_9'], name='Histogram', marker_color='rgba(128,0,128,0.5)'), row=3, col=1)


    # Add Support and Resistance lines
    for support in supports:
        fig.add_hline(y=support, line_dash="solid", line_color="lime", row=1, col=1, annotation_text=f"Support: {support:.2f}", annotation_position="bottom right")
    for resistance in resistances:
        fig.add_hline(y=resistance, line_dash="solid", line_color="crimson", row=1, col=1, annotation_text=f"Resistance: {resistance:.2f}", annotation_position="top right")

    # Chart Patterns (basic visualization - needs more robust pattern detection)
    # if show_patterns:
    #     # Example: Plotting 'Hammer' pattern if detected by pandas_ta
    #     # pandas_ta adds columns like 'CDL_HAMMER' with values like 100 (bullish) or -100 (bearish)
    #     # We'll just mark where it appears. A more sophisticated approach would highlight candles.
    #     hammer_patterns = df[df['CDL_HAMMER'] != 0]
    #     if not hammer_patterns.empty:
    #         for idx, row in hammer_patterns.iterrows():
    #             fig.add_annotation(x=idx, y=row['Low'] * 0.98, text="Hammer", showarrow=True, arrowhead=1, row=1, col=1)

    #     doji_patterns = df[df['CDL_DOJI'] != 0]
    #     if not doji_patterns.empty:
    #         for idx, row in doji_patterns.iterrows():
    #             fig.add_annotation(x=idx, y=row['High'] * 1.02, text="Doji", showarrow=True, arrowhead=1, row=1, col=1)


    fig.update_layout(
        title={
            'text': f'{ticker} Stock Price - {timeframe} Timeframe',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_rangeslider_visible=False,
        height=800,
        template="plotly_dark", # Modern dark theme
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2_title="Volume",
        yaxis3_title="Indicator Values",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Update axis labels visibility for better presentation
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1, showgrid=False)
    fig.update_yaxes(title_text="RSI/MACD", row=3, col=1, showgrid=False)

    st.plotly_chart(fig, use_container_width=True)

# --- Streamlit UI ---
st.title("Stock Technical Analysis Dashboard")

# Sidebar for user inputs
st.sidebar.header("Stock Selection & Settings")
stock_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., sbin)", "DIVISLAB").upper()

timeframe_option = st.sidebar.radio(
    "Select Timeframe",
    ('Daily', 'Hourly')
)

period_map = {
    'Daily': '1y', # 1 year of daily data
    'Hourly': '60d' # 60 days of hourly data (yfinance max for 1h interval)
}
interval_map = {
    'Daily': '1d',
    'Hourly': '1h'
}

selected_period = period_map[timeframe_option]
selected_interval = interval_map[timeframe_option]

st.sidebar.subheader("Analysis Options")
show_patterns_toggle = st.sidebar.checkbox("Detect Chart Patterns", value=True)
show_sr_toggle = st.sidebar.checkbox("Detect Support & Resistance", value=True)
sr_window = st.sidebar.slider("S/R Detection Window (candles)", 5, 50, 10)


# Fetch data
if stock_ticker:
    st.subheader(f"Analyzing {stock_ticker}")
    data = get_stock_data(stock_ticker, selected_period, selected_interval)

    if data is not None and not data.empty:
        # Calculate Technical Indicators using pandas_ta
        data.ta.sma(length=20, append=True)
        data.ta.sma(length=50, append=True)
        data.ta.bbands(append=True) # Bollinger Bands
        data.ta.rsi(append=True)
        # data.ta.macd(append=True)

        # Detect Chart Patterns
        # if show_patterns_toggle:
        #     # Add various candlestick patterns. pandas_ta automatically adds columns like 'CDL_HAMMER'
        #     data.ta.cdl_all(append=True) # Detect all candlestick patterns

        # Detect Support and Resistance
        supports, resistances = [], []
        if show_sr_toggle:
            supports, resistances = detect_support_resistance(data, window=sr_window)

        # Plotting
        plot_candlestick_with_indicators(data, stock_ticker, timeframe_option, show_patterns_toggle, supports, resistances)
    elif data is None:
        st.info("Please enter a valid stock ticker and ensure data is available for the selected period/interval.")
else:
    st.info("Please enter a stock ticker in the sidebar to get started.")

st.markdown("---")
st.markdown("Developed by a Sai.")
