import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta # Import pandas_ta for Bollinger Bands
from datetime import datetime, timedelta
import pytz # For timezone awareness

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Buy/Sell Stock Screener (BB & Camarilla)")

# --- Timezone Setup (Assuming Indian Market) ---
INDIAN_TZ = pytz.timezone('Asia/Kolkata')

# --- Helper Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_stock_data(ticker, period, interval):
    """Fetches historical stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker+".NS")
        data = stock.history(period=period, interval=interval)
        if data.empty:
            return None
        # Ensure index is timezone-aware for consistent comparison
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert(INDIAN_TZ) 
        return data
    except Exception as e:
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
    last_mb = bbands[mb_col].iloc[-1] if pd.notna(bbands[mb_col].iloc[-1]) else None
    last_ub = bbands[ub_col].iloc[-1] if pd.notna(bbands[ub_col].iloc[-1]) else None
    last_lb = bbands[lb_col].iloc[-1] if pd.notna(bbands[lb_col].iloc[-1]) else None
    last_bb_width_pct = bbands[bb_width_pct_col].iloc[-1] if pd.notna(bbands[bb_width_pct_col].iloc[-1]) else None

    return last_mb, last_ub, last_lb, last_bb_width_pct

def get_appropriate_period(interval):
    """Returns a suitable period for yfinance based on interval limitations."""
    # For previous day's last candle, 7 days should be sufficient for intraday
    if interval in ["1m", "2m", "5m", "15m"]:
        return "7d"
    elif interval in ["30m", "60m", "90m", "1h"]:
        return "45d" # Enough for Camarilla pivots and daily close
    return "1y" # Default for other intervals (e.g., 1wk, 1mo if added later)

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
    
    # Filter data to include only up to the end of the previous day
    start_of_current_day_ts = datetime.now(current_tz).date() #pd.Timestamp(today_date, tz=data.index.tz)
    #data_up_to_prev_day_end = data[data.index < start_of_current_day_ts]
    data.reset_index(inplace=True)
    data_up_to_prev_day_end = data[data['Datetime'].dt.date<start_of_current_day_ts]
    
    if data_up_to_prev_day_end.empty:
        return None, None, None

    # Calculate BBs on the data up to the previous day's end
    mb, ub, lb, width_pct = calculate_bollinger_bands(data_up_to_prev_day_end, bb_period, bb_std_dev)
    
    # If calculation failed or no valid last value
    if ub is None or lb is None or width_pct is None:
        return None, None, None

    return ub, lb, width_pct

def calculate_camarilla_pivots(daily_data):
    """
    Calculates Camarilla Pivot Points (R4, S4) for the current day based on previous day's data.
    Returns R4, S4. Returns None if previous day's data is not available.
    """
    if daily_data is None or daily_data.empty or len(daily_data) < 2:
        return None, None

    # Get previous day's High, Low, Close
    prev_day_high = daily_data['High'].iloc[-2]
    prev_day_low = daily_data['Low'].iloc[-2]
    prev_day_close = daily_data['Close'].iloc[-2]

    # Calculate Range
    prev_day_range = prev_day_high - prev_day_low

    # Calculate Camarilla Pivots
    # R4 = C + (H - L) * 1.1 / 2
    # S4 = C - (H - L) * 1.1 / 2
    r4 = prev_day_close + (prev_day_range * 1.1 / 2)
    s4 = prev_day_close - (prev_day_range * 1.1 / 2)

    return r4, s4

# --- Stock Screener Logic ---
def run_screener(all_tickers, screener_type,
                 buy_filter1_enabled, buy_filter2_enabled,
                 sell_filter1_enabled, sell_filter2_enabled,
                 enable_bb_5m_width_display, enable_bb_15m_width_display, enable_bb_1hr_width_display):
    """Applies screening criteria to a list of tickers based on screener_type."""
    screened_stocks = []
    
    progress_text = f"Running {screener_type} Screener. Please wait..."
    my_bar = st.progress(0, text=progress_text)
    
    current_tz = INDIAN_TZ

    total_tickers = len(all_tickers)
    for i, ticker in enumerate(all_tickers):
        my_bar.progress((i + 1) / total_tickers, text=f"Processing {ticker}...")

        try:
            # --- Fetch necessary data ---
            daily_data = get_stock_data(ticker, period="2d", interval="1d") # Need 2 days for Camarilla
            if daily_data is None or daily_data.empty:
                continue
            latest_daily_close = daily_data['Close'].iloc[-1]
            latest_daily_volume = daily_data['Volume'].iloc[-1] # For display

            five_min_data_for_comparison = get_stock_data(ticker, period="2d", interval="5m") 
            if five_min_data_for_comparison is None or five_min_data_for_comparison.empty:
                continue

            # Get today's 5m candle data
            today_date_local = datetime.now(current_tz).date()
            #today_five_min_data = five_min_data_for_comparison[five_min_data_for_comparison.index.date() == today_date_local]
            five_min_data_for_comparison.reset_index(inplace=True)
            today_five_min_data = five_min_data_for_comparison[five_min_data_for_comparison['Datetime'].dt.date==today_date_local]
            
            first_5m_open = today_five_min_data['Open'].iloc[0] if not today_five_min_data.empty else None
            first_5m_close = today_five_min_data['Close'].iloc[0] if not today_five_min_data.empty else None
            latest_5m_close = today_five_min_data['Close'].iloc[-1] if not today_five_min_data.empty else None

            # --- Calculate Camarilla Pivots (for current day, based on previous day's daily data) ---
            camarilla_r4, camarilla_s4 = calculate_camarilla_pivots(daily_data)

            # --- Calculate Previous Day's Last Candle BBs for 15m and 1hr ---
            # For 15m BB (50,2)
            bb_15m_data = get_stock_data(ticker, period=get_appropriate_period("15m"), interval="15m")
            prev_day_15m_ub, prev_day_15m_lb, bb_15m_width_pct = get_previous_day_last_candle_bb(
                bb_15m_data, 50, 2, current_tz
            )
            
            # For 1hr BB (50,2)
            bb_1hr_data = get_stock_data(ticker, period=get_appropriate_period("1h"), interval="1h")
            prev_day_1hr_ub, prev_day_1hr_lb, bb_1hr_width_pct = get_previous_day_last_candle_bb(
                bb_1hr_data, 50, 2, current_tz
            )

            # For 5m BB Width (for display, not filter condition)
            bb_5m_data = get_stock_data(ticker, period=get_appropriate_period("5m"), interval="5m")
            _, _, _, bb_5m_width_pct = calculate_bollinger_bands(bb_5m_data, 50, 2) # Use 50,2 for consistency in width display
            bb_5m_width_pct = round(bb_5m_width_pct, 2) if bb_5m_width_pct is not None else 'N/A'


            # --- Initialize filter results ---
            filter1_passed = False
            filter2_passed = False
            
            # --- Apply Screener Specific Logic ---
            if screener_type == "BUY":
                # Filter 1: OR(first 5m open > prev_day_15m_ub, first 5m open > prev_day_1hr_ub)
                if buy_filter1_enabled:
                    cond_15m_bb = False
                    if first_5m_open is not None and prev_day_15m_ub is not None:
                        cond_15m_bb = (first_5m_open > prev_day_15m_ub)
                    
                    cond_1hr_bb = False
                    if first_5m_open is not None and prev_day_1hr_ub is not None:
                        cond_1hr_bb = (first_5m_open > prev_day_1hr_ub)
                    
                    filter1_passed = (cond_15m_bb or cond_1hr_bb)
                else:
                    filter1_passed = True # If disabled, this filter passes

                # Filter 2: latest 5m candle close > camarilla R4 pivot point
                if buy_filter2_enabled:
                    if latest_5m_close is not None and camarilla_r4 is not None:
                        filter2_passed = (latest_5m_close > camarilla_r4)
                    else:
                        filter2_passed = False # Cannot calculate if data is missing
                else:
                    filter2_passed = True # If disabled, this filter passes

                # Overall BUY condition: AND between filter1 and filter2
                if filter1_passed and filter2_passed:
                    screened_stocks.append({
                        'Ticker': ticker,
                        'Close': round(latest_daily_close, 2),
                        'Volume': int(latest_daily_volume),
                        'BB 5m Width %': bb_5m_width_pct if enable_bb_5m_width_display else 'N/A',
                        'BB 15m Width %': round(bb_15m_width_pct, 2) if enable_bb_15m_width_display and bb_15m_width_pct is not None else 'N/A',
                        'BB 1hr Width %': round(bb_1hr_width_pct, 2) if enable_bb_1hr_width_display and bb_1hr_width_pct is not None else 'N/A',
                        'Camarilla R4': round(camarilla_r4, 2) if camarilla_r4 is not None else 'N/A',
                    })

            elif screener_type == "SELL":
                # Filter 1: OR(first 5m open < prev_day_15m_lb, first 5m open < prev_day_1hr_lb)
                if sell_filter1_enabled:
                    cond_15m_bb = False
                    if first_5m_open is not None and prev_day_15m_lb is not None:
                        cond_15m_bb = (first_5m_open < prev_day_15m_lb)
                    
                    cond_1hr_bb = False
                    if first_5m_open is not None and prev_day_1hr_lb is not None:
                        cond_1hr_bb = (first_5m_open < prev_day_1hr_lb)
                    
                    filter1_passed = (cond_15m_bb or cond_1hr_bb)
                else:
                    filter1_passed = True # If disabled, this filter passes

                # Filter 2: latest 5m candle close < camarilla S4 pivot point
                if sell_filter2_enabled:
                    if latest_5m_close is not None and camarilla_s4 is not None:
                        filter2_passed = (latest_5m_close < camarilla_s4)
                    else:
                        filter2_passed = False # Cannot calculate if data is missing
                else:
                    filter2_passed = True # If disabled, this filter passes

                # Overall SELL condition: AND between filter1 and filter2
                if filter1_passed and filter2_passed:
                    screened_stocks.append({
                        'Ticker': ticker,
                        'Close': round(latest_daily_close, 2),
                        'Volume': int(latest_daily_volume),
                        'BB 5m Width %': bb_5m_width_pct if enable_bb_5m_width_display else 'N/A',
                        'BB 15m Width %': round(bb_15m_width_pct, 2) if enable_bb_15m_width_display and bb_15m_width_pct is not None else 'N/A',
                        'BB 1hr Width %': round(bb_1hr_width_pct, 2) if enable_bb_1hr_width_display and bb_1hr_width_pct is not None else 'N/A',
                        'Camarilla S4': round(camarilla_s4, 2) if camarilla_s4 is not None else 'N/A',
                    })

        except Exception as e:
            st.warning(f"Skipping {ticker} due to an error during screening: {e}")
            
    my_bar.empty() # Clear progress bar
    return pd.DataFrame(screened_stocks)

# --- Main App Layout ---
st.title("📈 Buy/Sell Stock Screener")
st.markdown("Select your stock universe and run the BUY or SELL screener based on predefined conditions.")

# --- Stock Selection ---
st.write("Stock Universe")
default_tickers = ['COLPAL','BRITANNIA','HAL', 'LTIM', 'JINDALSTEL', 'INFY', 'OBEROIRLTY', 
                   'POLYCAB', 'ABB', 'JSWSTEEL',  'TCS', 'NAVINFLUOR', 'DEEPAKNTR', 'RELIANCE',
                   'SBILIFE', 'NESTLEIND', 'TITAN', 'PERSISTENT', 'ICICIBANK', 'HINDUNILVR',
                   'ALKEM', 'HDFCBANK', 'BHARTIARTL', 'COFORGE', 'PVRINOX', 'MRF', 'APOLLOHOSP',
                   'INDIGO', 'KOTAKBANK', 'BAJAJFINSV', 'ABBOTINDIA', 'SBIN', 'GRASIM', 'MARUTI',
                   'DIVISLAB',  'BAJFINANCE', 'MCX', 'EICHERMOT', 'SIEMENS', 'M&M', 'ACC', 'TVSMOTOR', 
                   'TORNTPHARM','DIXON', 'AUROPHARMA', 'GODREJPROP','TRENT', 'OFSS', 'SUNPHARMA',  
                   'HDFCAMC', 'BAJAJ-AUTO', 'ADANIPORTS', 'ULTRACEMCO', 'ADANIENT','UNITDSPR']
all_tickers = [t.strip().upper() for t in st.text_area("Enter stock tickers (one per line, e.g., RELIANCE.NS):", value="\n".join(default_tickers), height=200).split('\n') if t.strip()]

# --- BB Width Display Options ---
# st.header("2. Bollinger Band Width Display Options")
# st.markdown("Select which Bollinger Band Width percentages to display in the results table:")
# col_bb_width1, col_bb_width2, col_bb_width3 = st.columns(3)
# with col_bb_width1:
    # enable_bb_5m_width_display = st.checkbox("Show 5m BB Width %", value=True, key="enable_bb_5m_width_display")
# with col_bb_width2:
    # enable_bb_15m_width_display = st.checkbox("Show 15m BB Width %", value=True, key="enable_bb_15m_width_display")
# with col_bb_width3:
    # enable_bb_1hr_width_display = st.checkbox("Show 1hr BB Width %", value=True, key="enable_bb_1hr_width_display")
enable_bb_5m_width_display = enable_bb_15m_width_display = enable_bb_1hr_width_display =  True
# --- BUY Screener Section ---
st.header("BUY Screener")
st.markdown("Stocks that meet all **enabled** BUY conditions (AND logic between the two main filters).")

with st.expander("Configure BUY Filters"):
    st.subheader("Filter 1: Open Price vs. Previous Day's Bollinger Bands (OR Condition)")
    buy_filter1_enabled = st.checkbox("Enable BUY Filter 1", value=True, key="buy_filter1_enabled")
    st.markdown("*(First 5m candle open > Previous Day's Last 15m Upper BB (50,2) OR First 5m candle open > Previous Day's Last 1hr Upper BB (50,2))*")

    st.subheader("Filter 2: Latest Close vs. Camarilla R4")
    buy_filter2_enabled = st.checkbox("Enable BUY Filter 2", value=True, key="buy_filter2_enabled")
    st.markdown("*(Latest 5m candle close > Camarilla R4 Pivot Point)*")

if st.button("▶️ Run BUY Screener", type="primary"):
    screened_df = run_screener(
        all_tickers, "BUY",
        buy_filter1_enabled, buy_filter2_enabled,
        False, False, # SELL filters disabled for BUY screener run
        enable_bb_5m_width_display, enable_bb_15m_width_display, enable_bb_1hr_width_display
    )
    st.session_state['buy_screened_df'] = screened_df
    st.session_state['last_screener_run'] = "BUY"

# Display BUY Screener Results
if 'last_screener_run' in st.session_state and st.session_state['last_screener_run'] == "BUY":
    st.subheader("BUY Screener Results")
    if 'buy_screened_df' in st.session_state and not st.session_state['buy_screened_df'].empty:
        st.dataframe(st.session_state['buy_screened_df'], use_container_width=True)
    else:
        st.info("No stocks matched the BUY criteria.")

st.markdown("---")

# --- SELL Screener Section ---
st.write("SELL Screener")
st.markdown("Stocks that meet all **enabled** SELL conditions (AND logic between the two main filters).")

with st.expander("Configure SELL Filters"):
    st.subheader("Filter 1: Open Price vs. Previous Day's Bollinger Bands (OR Condition)")
    sell_filter1_enabled = st.checkbox("Enable SELL Filter 1", value=True, key="sell_filter1_enabled")
    st.markdown("*(First 5m candle open < Previous Day's Last 15m Lower BB (50,2) OR First 5m candle open < Previous Day's Last 1hr Lower BB (50,2))*")

    st.subheader("Filter 2: Latest Close vs. Camarilla S4")
    sell_filter2_enabled = st.checkbox("Enable SELL Filter 2", value=True, key="sell_filter2_enabled")
    st.markdown("*(Latest 5m candle close < Camarilla S4 Pivot Point)*")

if st.button("▶️ Run SELL Screener", type="primary"):
    screened_df = run_screener(
        all_tickers, "SELL",
        False, False, # BUY filters disabled for SELL screener run
        sell_filter1_enabled, sell_filter2_enabled,
        enable_bb_5m_width_display, enable_bb_15m_width_display, enable_bb_1hr_width_display
    )
    st.session_state['sell_screened_df'] = screened_df
    st.session_state['last_screener_run'] = "SELL"

# Display SELL Screener Results
if 'last_screener_run' in st.session_state and st.session_state['last_screener_run'] == "SELL":
    st.subheader("SELL Screener Results")
    if 'sell_screened_df' in st.session_state and not st.session_state['sell_screened_df'].empty:
        st.dataframe(st.session_state['sell_screened_df'], use_container_width=True)
    else:
        st.info("No stocks matched the SELL criteria.")

st.markdown("---")
st.markdown("Disclaimer: This dashboard is for educational purposes only and should not be considered financial advice. Stock data from `yfinance` is typically delayed and has limitations on historical intraday data.")

