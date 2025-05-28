import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(layout="wide")

# --- Configuration ---
# A representative list of F&O stocks (replace with a dynamic source for production)
# This list is for demonstration purposes.
# For a real application, consider an API or regularly updated CSV for NSE F&O.
FUTURES_AND_OPTIONS_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "AXISBANK.NS",
    "MARUTI.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "HCLTECH.NS",
    "ULTRACEMCO.NS", "NESTLEIND.NS", "ADANIENT.NS", "TITAN.NS", "SUNPHARMA.NS",
    "TATAMOTORS.NS", "GRASIM.NS", "TECHM.NS", "M&M.NS", "NTPC.NS",
    "ONGC.NS", "POWERGRID.NS", "HINDALCO.NS", "JSWSTEEL.NS", "CIPLA.NS",
    "DRREDDY.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "NTPC.NS",
    "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "COALINDIA.NS", "WIPRO.NS",
    "INDUSINDBK.NS", "AXISBANK.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS",
    "NMDC.NS", "SAIL.NS", "TATAPOWER.NS", "BHARATFORG.NS", "FEDERALBNK.NS",
    "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "UNIONBANK.NS", "IDFCFIRSTB.NS","LICI.NS"
]

# Time window for earnings
DAYS_PRIOR = 2
DAYS_AHEAD = 2

# Number of threads for parallel execution
# Adjust based on your internet connection and yfinance's tolerance.
# Too many threads can lead to rate limiting.
MAX_WORKERS = 10 

# --- Helper Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour to avoid excessive API calls
def get_earnings_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        # get_earnings_dates tries to get future and historic.
        # limit=100 ensures we get enough data to cover future dates if available
        earnings_dates = ticker.get_earnings_dates(limit=100)
        return ticker_symbol, earnings_dates
    except Exception as e:
        # st.warning(f"Could not retrieve earnings for {ticker_symbol}: {e}") # Don't use st.warning in threads directly
        return ticker_symbol, pd.DataFrame()

def filter_earnings_by_date(earnings_df, start_date, end_date):
    if earnings_df.empty:
        return pd.DataFrame()
    
    # yfinance often returns earnings dates as index with timezone info
    # Ensure the index is a datetime object and timezone-naive for comparison
    if isinstance(earnings_df.index, pd.DatetimeIndex):
        # Convert to timezone-naive if it has a timezone, otherwise leave as is
        if earnings_df.index.tz is not None:
            earnings_df.index = earnings_df.index.tz_localize(None) 
    elif not earnings_df.empty and isinstance(earnings_df.index[0], str):
        earnings_df.index = pd.to_datetime(earnings_df.index)
        if earnings_df.index.tz is not None:
            earnings_df.index = earnings_df.index.tz_localize(None)

    # Filter by date range
    #filtered_df = earnings_df[(earnings_df.index.date >= start_date) & (earnings_df.index.date <= end_date)]
    earnings_df.reset_index(inplace=True)
    filtered_df = earnings_df[earnings_df['Earnings Date'].dt.date>=start_date]
    return filtered_df

# --- Streamlit App Layout ---

st.title("F&O Stock Quarterly Results Dashboard")
st.markdown("Displays F&O stocks with quarterly results in the last 2 days and next 2 days.")

# Current Date
today = datetime.now().date()
st.sidebar.subheader("Date Range")
st.sidebar.write(f"Today: {today.strftime('%Y-%m-%d')}")

# Define the date range
start_date = today - timedelta(days=DAYS_PRIOR)
end_date = today + timedelta(days=DAYS_AHEAD)

st.sidebar.write(f"Checking results from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

st.subheader("Processing Earnings Data (Parallel Fetching)...")

results_data = []
stock_symbols_to_process = FUTURES_AND_OPTIONS_STOCKS

# Progress bar
progress_text = "Fetching earnings data. Please wait..."
my_bar = st.progress(0, text=progress_text)

total_stocks = len(stock_symbols_to_process)
processed_count = 0

# Use ThreadPoolExecutor for parallel fetching
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all tasks
    future_to_stock = {executor.submit(get_earnings_data, stock): stock for stock in stock_symbols_to_process}

    for future in as_completed(future_to_stock):
        stock_symbol = future_to_stock[future]
        try:
            _, earnings_df = future.result() # Unpack the tuple from get_earnings_data
            
            if not earnings_df.empty:
                filtered_earnings = filter_earnings_by_date(earnings_df, start_date, end_date)
                st.write(filtered_earnings)
                
                for index, row in filtered_earnings.iterrows():
                    result_date = index#.strftime('%Y-%m-%d')
                    quarter = row.get('Quarter', 'N/A') # yfinance might not always provide 'Quarter'
                    eps_estimate = row.get('EPS Estimate', 'N/A')
                    reported_eps = row.get('Reported EPS', 'N/A')
                    
                    results_data.append({
                        "Symbol": stock_symbol,
                        "Earnings Date": result_date,
                        "Quarter": quarter,
                        "EPS Estimate": eps_estimate,
                        "Reported EPS": reported_eps
                    })
        except Exception as exc:
            st.warning(f"{stock_symbol} generated an exception: {exc}")
        
        processed_count += 1
        my_bar.progress(processed_count / total_stocks, text=f"Processing {stock_symbol} ({processed_count}/{total_stocks})")
    
my_bar.empty() # Clear the progress bar after completion

if results_data:
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values(by="Earnings Date").reset_index(drop=True)
    
    st.subheader("F&O Stocks with Quarterly Results")
    st.dataframe(results_df, use_container_width=True)
    
    st.download_button(
        label="Download Results as CSV",
        data=results_df.to_csv(index=False).encode('utf-8'),
        file_name="fno_quarterly_results.csv",
        mime="text/csv",
    )
else:
    st.info("No F&O stocks found with quarterly results in the specified date range.")

st.markdown("""
---
**Important Notes:**
- The list of F&O stocks is a static, representative sample. For real-time and exhaustive F&O data, consider using a dedicated financial data API or scraping from the NSE website (with caution and adherence to their terms of service).
- `yfinance`'s future earnings date data can sometimes be incomplete or delayed.
- The 'Quarter', 'EPS Estimate', and 'Reported EPS' columns might show 'N/A' if `yfinance` doesn't provide that specific data for a given entry.
- **Parallel processing with `ThreadPoolExecutor` significantly speeds up data fetching, but be mindful of API rate limits. Too many `MAX_WORKERS` can cause issues.**
""")
