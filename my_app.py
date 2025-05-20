import snscrape.modules.twitter as sntwitter
import pandas as pd
import streamlit as st

# Define Twitter handle and stock list
twitter_handle = "REDBOXINDIA"
stock_list = ["DIXON", "DIVISLAB"]
# earnings_dates = {"TSLA": "2025-05-15", "AAPL": "2025-05-20", "NVDA": "2025-05-25"}

# Scrape tweets
tweets = []
for tweet in sntwitter.TwitterUserScraper(twitter_handle).get_items():
    for stock in stock_list:
        if stock in tweet.content:
            #tweets.append([tweet.date, tweet.content, stock, earnings_dates.get(stock, "N/A")])
            tweets.append([tweet.date, tweet.content, stock])

# Convert to DataFrame
df = pd.DataFrame(tweets, columns=["Date", "Tweet", "Stock"])

# Streamlit Dashboard
st.title(f"Filtered Tweets from @{twitter_handle}")
st.dataframe(df)

# Visualization
st.bar_chart(df.groupby("Stock").count()["Tweet"])
