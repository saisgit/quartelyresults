import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta # Import pandas_ta for Bollinger Bands
from datetime import datetime, timedelta
import pytz # For timezone awareness

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Stock Screener (Prev Day BB Filters)")

# --- Helper Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_stock_data(ticker, period, interval):
    """Fetches historical stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker+".NS")
        # yfinance often returns data with timezone. Make it timezone-aware.
        data = stock.history(period=period, interval=interval)
        if data.empty:
            return None
        # Ensure index is timezone-aware for consistent comparison
        # Assuming Indian market for timezone 'Asia/Kolkata'
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert('Asia/Kolkata') 
        return data
    except Exception as e:
        # print(f"Error fetching data for {ticker}: {e}") # Uncomment for debugging in console
        return None

def calculate_bollinger_bands(data, window=50, num_std_dev=2):
    """
    Calculates Bollinger Bands (Middle Band, Upper Band, Lower Band) and Width Percentage
    using pandas_ta. Returns scalar values for the last candle's MB, UB, LB, and Width %.
    Returns None for any value if calculation fails or data is insufficient.
    """
    if data is None or data.empty or len(data) < window: # Ensure enough data for the window
        return None, None, None, None # MB, UB, LB, Width %
    
    bbands = ta.bbands(data['Close'], length=window, std=num_std_dev, append=False)
    
    if bbands.empty or len(bbands) < 1: # Check if bbands DataFrame is empty after calculation
        return None, None, None, None

    mb_col = f'BBM_{window}_{float(num_std_dev)}'
    ub_col = f'BBU_{window}_{float(num_std_dev)}'
    lb_col = f'BBL_{window}_{float(num_std_dev)}'
    bb_width_pct_col = f'BBB_{window}_{float(num_std_dev)}' 
    
    # Check if columns exist before accessing, which can happen if data is too short for the window
    if not all(col in bbands.columns for col in [mb_col, ub_col, lb_col, bb_width_pct_col]):
        return None, None, None, None

    # Access the last valid (non-NaN) value from each series
    # Using .iloc[-1] is generally safe for the last element, but check for NaN
    last_mb = bbands[mb_col].iloc[-1] if pd.notna(bbands[mb_col].iloc[-1]) else None
    last_ub = bbands[ub_col].iloc[-1] if pd.notna(bbands[ub_col].iloc[-1]) else None
    last_lb = bbands[lb_col].iloc[-1] if pd.notna(bbands[lb_col].iloc[-1]) else None
    last_bb_width_pct = bbands[bb_width_pct_col].iloc[-1] if pd.notna(bbands[bb_width_pct_col].iloc[-1]) else None

    return last_mb, last_ub, last_lb, last_bb_width_pct

def get_appropriate_period(interval):
    """Returns a suitable period for yfinance based on interval limitations."""
    # yfinance often provides max 60 days for <1h intervals, and 730 days for 1h/60m
    # For previous day's last candle, 7 days should be sufficient to get enough history
    if interval in ["1m", "5m", "15m", "30m", "90m"]: return "7d"
    if interval in ["60m", "1h"]: return "45d"
    return "1y" # Default for daily/weekly/monthly if used elsewhere

def get_previous_day_last_candle_bb(data, bb_period, bb_std_dev, current_tz):
    """
    Extracts Bollinger Bands (UB, LB) for the last complete candle of the previous trading day.
    Returns scalar UB, LB, and Width % values. Returns None if data is insufficient or previous day's candle not found.
    """
    if data is None or data.empty:
        return None, None, None # UB, LB, Width %

    if data.index.tz is None: # Ensure timezone is set for consistency
        data.index = data.index.tz_localize('UTC').tz_convert(current_tz)
    
    today_date = datetime.now(current_tz).date()
    #dates_in_data = data.index.date()
    data.reset_index(inplace=True)
    data_up_to_prev_day_end = data[data['Datetime'].dt.date<today_date]
    # Filter data to include only up to the end of the previous day
    # This ensures BB calculation is based on data *before* today's candles
    #data_up_to_prev_day_end = data[dates_in_data < today_date]
    #data_up_to_prev_day_end = data[data['Date'].dt.date<=today_date]
    #print(data_up_to_prev_day_end)
    
    if data_up_to_prev_day_end.empty:
        return None, None, None

    # Calculate BBs on the data up to the previous day's end
    # This function now returns scalar values directly
    mb, ub, lb, width_pct = calculate_bollinger_bands(data_up_to_prev_day_end, bb_period, bb_std_dev)
    print(ub,lb)
    
    # If calculation failed or no valid last value
    if ub is None or lb is None or width_pct is None:
        return None, None, None

    # The `calculate_bollinger_bands` function already returns the last scalar values.
    return ub, lb, width_pct


# --- Stock Screener Logic ---
def run_screener(all_tickers,filter_logic, #min_price, max_price, min_volume, 
                 enable_bb_5m, bb_5m_period, bb_5m_std_dev, bb_5m_price_type, bb_5m_condition, enable_bb_5m_width_display,
                 enable_bb_15m, bb_15m_period, bb_15m_std_dev, bb_15m_price_type, bb_15m_condition, enable_bb_15m_width_display,
                 enable_bb_1hr, bb_1hr_period, bb_1hr_std_dev, bb_1hr_price_type, bb_1hr_condition, enable_bb_1hr_width_display):
    """Applies screening criteria to a list of tickers."""
    screened_stocks = []
    
    progress_text = "Screening stocks. Please wait..."
    my_bar = st.progress(0, text=progress_text)
    
    current_tz = pytz.timezone('Asia/Kolkata') # Set the timezone for current day calculations

    total_tickers = len(all_tickers)
    for i, ticker in enumerate(all_tickers):
        my_bar.progress((i + 1) / total_tickers, text=f"Processing {ticker}...")

        if True: # Added try-except for robust error handling per stock
            # Fetch daily data for basic filters (price, volume)
            # daily_data = get_stock_data(ticker, period="1y", interval="1d")
            # if daily_data is None or daily_data.empty:
                # continue

            # latest_daily_close = daily_data['Close'].iloc[-1]
            # latest_daily_volume = daily_data['Volume'].iloc[-1]
            
            # --- Fetch 5m data for comparison prices (current day) ---
            # Need enough 5m data to ensure first candle of day is available.
            # Fetching for 2 days to handle market open edge cases.
            five_min_data_for_comparison = get_stock_data(ticker, period="2d", interval="5m") 
            
            first_5m_open = None
            first_5m_close = None
            latest_5m_close = None

            if five_min_data_for_comparison is not None and not five_min_data_for_comparison.empty:
                # Filter for today's data (assuming market is open)
                today_date_local = datetime.now(current_tz).date()
                #today_five_min_data = five_min_data_for_comparison[five_min_data_for_comparison.index.date() == today_date_local]
                five_min_data_for_comparison.reset_index(inplace=True)
                today_five_min_data = five_min_data_for_comparison[five_min_data_for_comparison['Datetime'].dt.date==today_date_local]
                
                if not today_five_min_data.empty:
                    first_5m_open = today_five_min_data['Open'].iloc[0]
                    first_5m_close = today_five_min_data['Close'].iloc[0]
                    latest_5m_close = today_five_min_data['Close'].iloc[-1]
                else: # If no data for today (e.g., weekend), use latest available 5m candle from yfinance for latest_5m_close
                    # This might be from previous day, but it's the 'latest' yfinance provides
                    latest_5m_close = five_min_data_for_comparison['Close'].iloc[-1]
                    # first_5m_open/close remain None if no data for today

            # --- Initialize BB Widths for results table ---
            bb_5m_width_pct_display = 'N/A'
            bb_15m_width_pct_display = 'N/A'
            bb_1hr_width_pct_display = 'N/A'

            # Collect results of all ENABLED filters
            # Basic Price and Volume filters are always considered in `enabled_filter_results`
            enabled_filter_results = []

            # 1. Basic Price Filter
            #price_filter_passed = (latest_daily_close >= min_price) and (latest_daily_close <= max_price)
            #enabled_filter_results.append(price_filter_passed)

            # 2. Basic Volume Filter
            # volume_filter_passed = (latest_daily_volume >= min_volume)
            # enabled_filter_results.append(volume_filter_passed)
            
            # 3. Bollinger Bands Filters
            bb_filters_config = {
                "5m": {"enabled": enable_bb_5m, "period": bb_5m_period, "std_dev": bb_5m_std_dev, "price_type": bb_5m_price_type, "condition": bb_5m_condition, "width_display_enabled": enable_bb_5m_width_display},
                "15m": {"enabled": enable_bb_15m, "period": bb_15m_period, "std_dev": bb_15m_std_dev, "price_type": bb_15m_price_type, "condition": bb_15m_condition, "width_display_enabled": enable_bb_15m_width_display},
                "1h": {"enabled": enable_bb_1hr, "period": bb_1hr_period, "std_dev": bb_1hr_std_dev, "price_type": bb_1hr_price_type, "condition": bb_1hr_condition, "width_display_enabled": enable_bb_1hr_width_display},
            }

            for interval, params in bb_filters_config.items():
                # Fetch data for BB calculation (need enough history for previous day's last candle)
                bb_calc_data = get_stock_data(ticker, period=get_appropriate_period(interval), interval=interval)
                
                prev_day_ub, prev_day_lb, latest_width_pct = get_previous_day_last_candle_bb(
                    bb_calc_data, params["period"], params["std_dev"], current_tz
                )
                
                # Update width for display in results table, if enabled
                if params["width_display_enabled"] and latest_width_pct is not None:
                    if interval == "5m": bb_5m_width_pct_display = round(latest_width_pct, 2)
                    elif interval == "15m": bb_15m_width_pct_display = round(latest_width_pct, 2)
                    elif interval == "1h": bb_1hr_width_pct_display = round(latest_width_pct, 2)

                # Only apply filter logic if the filter is enabled
                if params["enabled"]:
                    if prev_day_ub is None or prev_day_lb is None: # If previous day's BBs couldn't be calculated
                        enabled_filter_results.append(False)
                        continue
                    
                    # Determine which comparison price to use from 5m data
                    comparison_price = None
                    if params["price_type"] == "First 5m Candle Close":
                        comparison_price = first_5m_close
                    elif params["price_type"] == "First 5m Candle Open":
                        comparison_price = first_5m_open
                    elif params["price_type"] == "Latest 5m Candle Close":
                        comparison_price = latest_5m_close
                    
                    if comparison_price is None: # If 5m data or specific comparison candle not found
                        enabled_filter_results.append(False)
                        continue

                    bb_condition_passed = False
                    if params["condition"] == "Price > Upper Band":
                        bb_condition_passed = (comparison_price > prev_day_ub)
                    elif params["condition"] == "Price < Lower Band":
                        bb_condition_passed = (comparison_price < prev_day_lb)
                    elif params["condition"] == "Price within Bands":
                        bb_condition_passed = (comparison_price >= prev_day_lb and comparison_price <= prev_day_ub)
                    
                    enabled_filter_results.append(bb_condition_passed)

            # Apply global filter logic
            pass_stock = False
            
            # If no BB filters are enabled, only basic price/volume filters apply
            # This handles the case where only basic filters are active
            if not (enable_bb_5m or enable_bb_15m or enable_bb_1hr):
                pass_stock = (price_filter_passed and volume_filter_passed)
            elif filter_logic == "AND":
                pass_stock = all(enabled_filter_results)
            elif filter_logic == "OR":
                # For OR, we need to check if any of the *enabled* filters passed.
                # `enabled_filter_results` already contains results for basic filters and enabled BB filters.
                pass_stock = any(enabled_filter_results)

            if pass_stock:
                screened_stocks.append({
                    'Ticker': ticker,
                    #'Close': round(latest_daily_close, 2),
                    #'Volume': int(latest_daily_volume),
                    'BB 5m Width %': bb_5m_width_pct_display,
                    'BB 15m Width %': bb_15m_width_pct_display,
                    'BB 1hr Width %': bb_1hr_width_pct_display,
                })
        # except Exception as e:
            # st.warning(f"Skipping {ticker} due to an error during screening: {e}")
            # This will help identify which stock might be causing issues without crashing the whole app
            
    my_bar.empty() # Clear progress bar
    return pd.DataFrame(screened_stocks)

# --- Main App Layout ---
st.title("📊 Stock Screener (Previous Day BB Filters)")
st.markdown("Use the sidebar to define your screening criteria and find stocks that match!")

# Sidebar for Screener Inputs
st.sidebar.header("Stock Screener Criteria")

# Example Indian stock tickers (replace with your desired list)
default_tickers = ['COLPAL','BRITANNIA','HAL', 'LTIM', 'JINDALSTEL', 'INFY', 'OBEROIRLTY', 
                   'POLYCAB', 'ABB', 'JSWSTEEL',  'TCS', 'NAVINFLUOR', 'DEEPAKNTR', 'RELIANCE',
                   'SBILIFE', 'NESTLEIND', 'TITAN', 'PERSISTENT', 'ICICIBANK', 'HINDUNILVR',
                   'ALKEM', 'HDFCBANK', 'BHARTIARTL', 'COFORGE', 'PVRINOX', 'MRF', 'APOLLOHOSP',
                   'INDIGO', 'KOTAKBANK', 'BAJAJFINSV', 'ABBOTINDIA', 'SBIN', 'GRASIM', 'MARUTI',
                   'DIVISLAB',  'BAJFINANCE', 'MCX', 'EICHERMOT', 'SIEMENS', 'M&M', 'ACC', 'TVSMOTOR', 
                   'TORNTPHARM','DIXON', 'AUROPHARMA', 'GODREJPROP','TRENT', 'OFSS', 'SUNPHARMA',  
                   'HDFCAMC', 'BAJAJ-AUTO', 'ADANIPORTS', 'ULTRACEMCO', 'ADANIENT','UNITDSPR']
# Allow user to input tickers or use default
ticker_input = st.sidebar.text_area("Enter stock tickers (one per line, e.g., RELIANCE):", value="\n".join(default_tickers), height=200)
all_tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]

# Global Filter Logic
st.sidebar.subheader("Global Filter Logic")
filter_logic = st.sidebar.radio("Combine filters with:", ("AND", "OR"), key="filter_logic")

# Basic Filters (Always enabled, but their results are part of `enabled_filter_results` for global logic)
# st.sidebar.subheader("Basic Filters (Daily Data)")
# min_price = st.sidebar.number_input("Minimum Close Price:", value=100.0, min_value=0.0, key="min_price")
# max_price = st.sidebar.number_input("Maximum Close Price:", value=15000.0, min_value=0.0, key="max_price")
# min_volume = st.sidebar.number_input("Minimum Daily Volume (in shares):", value=100000.0, min_value=0.0, key="min_volume")

# Bollinger Bands Filters (using expanders for organization)
st.sidebar.subheader("Bollinger Bands Filters (Previous Day's Last Candle)")

bb_price_comparison_options = ["First 5m Candle Close", "First 5m Candle Open", "Latest 5m Candle Close"]

# 5m BB Filter
with st.sidebar:
    st.write("5-Minute Bollinger Bands")
    enable_bb_5m = st.checkbox("Enable 5m BB Filter", key="enable_bb_5m")
    bb_5m_period = st.number_input("5m BB Period:", value=50, min_value=1, key="bb_5m_period")
    bb_5m_std_dev = st.number_input("5m BB Std Dev:", value=2.0, min_value=0.1, key="bb_5m_std_dev")
    bb_5m_price_type = st.selectbox("Compare against:", bb_price_comparison_options, key="bb_5m_price_type_5m") # Unique key
    bb_5m_condition = st.selectbox("5m BB Condition:", ["Price within Bands", "Price > Upper Band", "Price < Lower Band"], key="bb_5m_condition")
    enable_bb_5m_width_display = st.checkbox("Show 5m BB Width %", value=True, key="enable_bb_5m_width_display")

# 15m BB Filter
    st.write("15-Minute Bollinger Bands")
    enable_bb_15m = st.checkbox("Enable 15m BB Filter", key="enable_bb_15m",value=True)
    bb_15m_period = st.number_input("15m BB Period:", value=50, min_value=1, key="bb_15m_period")
    bb_15m_std_dev = st.number_input("15m BB Std Dev:", value=2.0, min_value=0.1, key="bb_15m_std_dev")
    bb_15m_price_type = st.selectbox("Compare against:", bb_price_comparison_options, key="bb_15m_price_type_15m") # Unique key
    bb_15m_condition = st.selectbox("15m BB Condition:", ["Price within Bands", "Price > Upper Band", "Price < Lower Band"], key="bb_15m_condition")
    enable_bb_15m_width_display = st.checkbox("Show 15m BB Width %", value=True, key="enable_bb_15m_width_display")

# 1hr BB Filter
    st.write("1-Hour Bollinger Bands")
    enable_bb_1hr = st.checkbox("Enable 1hr BB Filter", key="enable_bb_1hr",value=True)
    bb_1hr_period = st.number_input("1hr BB Period:", value=50, min_value=1, key="bb_1hr_period")
    bb_1hr_std_dev = st.number_input("1hr BB Std Dev:", value=2.0, min_value=0.1, key="bb_1hr_std_dev")
    bb_1hr_price_type = st.selectbox("Compare against:", bb_price_comparison_options, key="bb_1hr_price_type_1hr") # Unique key
    bb_1hr_condition = st.selectbox("1hr BB Condition:", ["Price within Bands", "Price > Upper Band", "Price < Lower Band"], key="bb_1hr_condition")
    enable_bb_1hr_width_display = st.checkbox("Show 1hr BB Width %", value=True, key="enable_bb_1hr_width_display")

# Run Screener Button
if st.sidebar.button("Run Screener"):
    screened_df = run_screener(
        all_tickers,# min_price, max_price, min_volume,
        filter_logic,
        enable_bb_5m, bb_5m_period, bb_5m_std_dev, bb_5m_price_type, bb_5m_condition, enable_bb_5m_width_display,
        enable_bb_15m, bb_15m_period, bb_15m_std_dev, bb_15m_price_type, bb_15m_condition, enable_bb_15m_width_display,
        enable_bb_1hr, bb_1hr_period, bb_1hr_std_dev, bb_1hr_price_type, bb_1hr_condition, enable_bb_1hr_width_display
    )
    st.session_state['screened_df'] = screened_df # Store in session state

# Display Screener Results
if 'screened_df' in st.session_state and not st.session_state['screened_df'].empty:
    st.header("Screener Results")
    st.dataframe(st.session_state['screened_df'], use_container_width=True)
elif 'screened_df' in st.session_state and st.session_state['screened_df'].empty:
    st.info("No stocks matched your criteria. Try adjusting the filters.")
else:
    st.info("Run the screener to see results.")

st.markdown("---")
st.markdown("Disclaimer: This dashboard is for educational purposes only and should not be considered financial advice. Stock data from `yfinance` is typically delayed and has limitations on historical intraday data.")

