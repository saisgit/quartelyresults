import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Set the title and sidebar options
st.title("Stock Price Action Analysis Dashboard")
st.sidebar.header("User Input")

# User input: ticker and timeframe selection
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
timeframe = st.sidebar.selectbox("Select Timeframe", ["Daily", "Hourly"])

# Fetch data based on timeframe
if timeframe == "Daily":
    # 1-year history with daily candles
    data = yf.download(ticker, period="1y", interval="1d")
else:
    # 60-day history with hourly candles
    data = yf.download(ticker, period="60d", interval="60m")

# Ensure we don’t have any missing values
data.dropna(inplace=True)

#############################################
# Price Action Analysis Helper Functions
#############################################

def find_pivots(df, window=5):
    """
    Identify pivot points: local support (low) and resistance (high) levels.
    A point is considered a pivot if it is the minimum/maximum in a window.
    """
    supports = []
    resistances = []
    # Iterate over the data (skipping the first/last few points)
    for i in range(window, len(df) - window):
        current_low = df["Low"].iloc[i]
        current_high = df["High"].iloc[i]
        window_low = df["Low"].iloc[i - window : i + window + 1].min()
        window_high = df["High"].iloc[i - window : i + window + 1].max()
        if np.isclose(current_low, window_low):
            supports.append((df.index[i], current_low))
        if np.isclose(current_high, window_high):
            resistances.append((df.index[i], current_high))
    return supports, resistances

def cluster_levels(pivots, threshold):
    """
    Simplistic clustering of pivot levels.
    The idea is to group levels that are very close in price (within 'threshold').
    """
    levels = []
    for _, price in pivots:
        if not levels:
            levels.append(price)
        else:
            # Only add a new level if it differs significantly from existing levels
            if not any(abs(price - lvl) < threshold for lvl in levels):
                levels.append(price)
    return levels

def compute_trend_line(pivots, total_points):
    """
    Uses linear regression on the pivot points to compute a trend line.
    Here, we convert the datetime index of each pivot into its integer position in the dataframe.
    """
    if len(pivots) < 2:
        return None
    # Convert pivot timestamps into their positional indices within the dataframe.
    x = np.array([data.index.get_loc(time) for time, price in pivots])
    y = np.array([price for time, price in pivots])
    # Compute the best-fit line
    slope, intercept = np.polyfit(x, y, 1)
    # Create the trend line over the entire data range
    x_line = np.arange(total_points)
    y_line = slope * x_line + intercept
    return y_line

#############################################
# Compute Pivots, Clustering & Trend Lines
#############################################

# Look for pivot lows (supports) and highs (resistances)
supports, resistances = find_pivots(data, window=5)

# Cluster these levels to avoid drawing too many lines.
# Here, we use 1% of the total price range as our clustering threshold.
price_range = data["High"].max() - data["Low"].min()
threshold = 0.01 * price_range

support_levels = cluster_levels(supports, threshold)
resistance_levels = cluster_levels(resistances, threshold)

# Compute trend lines using linear regression on the pivot points.
support_trend = compute_trend_line(supports, len(data))
resistance_trend = compute_trend_line(resistances, len(data))

#############################################
# Plotting with Plotly
#############################################

fig = go.Figure()

# Create a candlestick chart for the stock data
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Price"
))

# Add horizontal lines for support levels (green dashed lines)
for level in support_levels:
    fig.add_hline(
        y=level,
        line_dash="dash",
        line_color="green",
        annotation_text="Support",
        annotation_position="bottom left"
    )

# Add horizontal lines for resistance levels (red dashed lines)
for level in resistance_levels:
    fig.add_hline(
        y=level,
        line_dash="dash",
        line_color="red",
        annotation_text="Resistance",
        annotation_position="top right"
    )

# Overlay the trend lines (if available)
if support_trend is not None:
    fig.add_trace(go.Scatter(
        x=data.index,
        y=support_trend,
        mode="lines",
        name="Support Trend",
        line=dict(color="blue")
    ))

if resistance_trend is not None:
    fig.add_trace(go.Scatter(
        x=data.index,
        y=resistance_trend,
        mode="lines",
        name="Resistance Trend",
        line=dict(color="orange")
    ))

# Update layout for clarity
fig.update_layout(
    title=f"{ticker} Price Action Analysis ({timeframe} timeframe)",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    template="plotly_white"
)

# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)

#############################################
# Additional Analysis & Future Enhancements
#############################################

st.write("""
### About this Analysis

This basic implementation identifies local pivot points within a sliding window on the historical data, clusters them to highlight significant support and resistance zones, and computes trend lines using linear regression on the detected pivots. 

There is ample room for additional refinements:
- **Enhanced Pivot Detection:** You might incorporate more advanced heuristics (or even machine learning models) to better identify meaningful swing highs and lows.
- **Dynamic Clustering:** Adjust the clustering threshold dynamically based on volatility or volume.
- **Additional Overlays:** Consider adding other technical indicators such as moving averages, Bollinger Bands, or Fibonacci retracements.
- **Interactivity:** Allow users to adjust analysis parameters (like window size or threshold) in real-time.
""")
