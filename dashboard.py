import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title="Crypto Data Dashboard", layout="wide")


@st.cache_data
def load_data(coin_name):
    """Load data based on the selected coin"""
    if coin_name == "Solana":
        df = pd.read_csv("Solana_cleaned.csv")
    elif coin_name == "Ethereum":
        df = pd.read_csv("ethereum_data_cleaned.csv")
    elif coin_name == "Bitcoin":
        df = pd.read_csv("Bitcoin_cleaned.csv")

    df["Time of Scraping"] = pd.to_datetime(df["Time of Scraping"])
    df = df.sort_values("Time of Scraping").reset_index(drop=True)
    return df


# Sidebar - Coin Selection
st.sidebar.header("🪙 Select Coin")
coin_choice = st.sidebar.selectbox(
    "Choose a cryptocurrency to visualize:",
    options=["Solana", "Ethereum", "Bitcoin"],
    index=0
)

# Load the data for the selected coin
df = load_data(coin_choice)

# Sidebar - Filters and Settings
st.sidebar.header("🔍 Filters and Settings")

# Date Range filter
min_date = df["Time of Scraping"].min().date()
max_date = df["Time of Scraping"].max().date()
date_range = st.sidebar.date_input(
    "📅 Select Date Range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date,
    help="Filter data by selecting a range of dates."
)

# Hour Range filter (0 to 23)
hour_range = st.sidebar.slider(
    "⏰ Select Hour Range",
    min_value=0, max_value=23, value=(0, 23),
    help="Filter data by selecting a range of hours (0 means midnight, 23 means 11 PM)."
)

# Moving Average Window Sizes (customizable)
st.sidebar.subheader("Moving Averages Settings")
ma_short = st.sidebar.slider("Short-Term MA Window (hours)", 3, 50, 7,
                             help="Select window size for short-term moving average.")
ma_long = st.sidebar.slider("Long-Term MA Window (hours)", 10, 200, 24,
                            help="Select window size for long-term moving average.")

# Filter the dataframe by selected date and hour range
df_filtered = df.copy()
if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df_filtered[(df_filtered["Time of Scraping"].dt.date >= start_date) &
                              (df_filtered["Time of Scraping"].dt.date <= end_date)]
df_filtered = df_filtered[(df_filtered["hour"] >= hour_range[0]) & (df_filtered["hour"] <= hour_range[1])]

# Sidebar - Download Filtered Data
csv = df_filtered.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name=f'{coin_choice.lower()}_filtered_data.csv',
    mime='text/csv'
)

# Main Dashboard Title and Intro
st.title(f"💡 {coin_choice} Data Dashboard")
st.write(f"""
This dashboard visualizes historical {coin_choice} data. Adjust the filters on the left to explore the data by date and hour.
Below, you'll see various graphs that provide insights into {coin_choice}'s price, market cap, volume, and volatility.
""")

# --------------------------
# Graph 1: Price Over Time with Annotations
st.subheader(f"📊 {coin_choice} Price Over Time")
st.write(
    f"This graph shows how {coin_choice}'s price has evolved over time. The highest and lowest points in the filtered range are annotated.")

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df_filtered["Time of Scraping"], df_filtered["Price"], color="blue", label="Price")
# Annotate highest and lowest price in the filtered data
max_price = df_filtered["Price"].max()
min_price = df_filtered["Price"].min()
max_time = df_filtered.loc[df_filtered["Price"].idxmax(), "Time of Scraping"]
min_time = df_filtered.loc[df_filtered["Price"].idxmin(), "Time of Scraping"]
ax1.annotate(f"Max: {max_price:.2f}", xy=(max_time, max_price), xytext=(max_time, max_price * 1.05),
             arrowprops=dict(facecolor='green', shrink=0.05), fontsize=10)
ax1.annotate(f"Min: {min_price:.2f}", xy=(min_time, min_price), xytext=(min_time, min_price * 0.95),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10)
ax1.set_title(f"{coin_choice} Price Over Time")
ax1.set_xlabel("Time")
ax1.set_ylabel("Price (USD)")
ax1.grid(True)
st.pyplot(fig1)

# --------------------------
# Graph 2: Market Cap Over Time
st.subheader(f"📊 {coin_choice} Market Cap Over Time")
st.write(f"This graph illustrates changes in {coin_choice}'s market capitalization over time.")

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(df_filtered["Time of Scraping"], df_filtered["Market Cap"], color="green")
ax2.set_title(f"{coin_choice} Market Cap Over Time")
ax2.set_xlabel("Time")
ax2.set_ylabel("Market Cap (USD)")
ax2.grid(True)
st.pyplot(fig2)

# --------------------------
# Graph 3: 24h Trading Volume Over Time
st.subheader(f"📊 24h Trading Volume Over Time")
st.write(f"This graph displays the 24-hour trading volume trend for {coin_choice}.")

fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(df_filtered["Time of Scraping"], df_filtered["24h Trading Volume"], color="purple")
ax3.set_title(f"{coin_choice} 24h Trading Volume Over Time")
ax3.set_xlabel("Time")
ax3.set_ylabel("Trading Volume")
ax3.grid(True)
st.pyplot(fig3)

# --------------------------
# Graph 4: Percentage Change Over Time (24h and 7d)
st.subheader(f"🔁 {coin_choice} Percentage Change Over Time")
st.write(
    f"This graph shows the 24-hour and 7-day percentage changes in {coin_choice}'s price, offering insight into market fluctuations.")

fig4, ax4 = plt.subplots(figsize=(12, 5))
ax4.plot(df_filtered["Time of Scraping"], df_filtered["24h Price Change %"], label="24h Change %", color="red")
ax4.plot(df_filtered["Time of Scraping"], df_filtered["7d Price Change %"], label="7d Change %", color="blue")
ax4.set_title(f"{coin_choice} Percentage Change Over Time")
ax4.set_xlabel("Time")
ax4.set_ylabel("% Change")
ax4.legend()
ax4.grid(True)
st.pyplot(fig4)

# --------------------------
# Graph 5: Price Distribution Histogram
st.subheader(f"📊 {coin_choice} Price Distribution")
st.write(f"This histogram shows the distribution of {coin_choice}'s price within the filtered period.")

fig5, ax5 = plt.subplots(figsize=(12, 5))
ax5.hist(df_filtered["Price"], bins=30, color="orange", edgecolor="black")
ax5.set_title(f"{coin_choice} Price Distribution")
ax5.set_xlabel("Price (USD)")
ax5.set_ylabel("Frequency")
ax5.grid(True)
st.pyplot(fig5)

# --------------------------
# Graph 6: Moving Averages (Customizable)
st.subheader(f"📊 {coin_choice} Moving Averages")
st.write(
    f"This graph overlays the actual price of {coin_choice} with short-term and long-term moving averages. Adjust the window sizes using the sliders on the sidebar.")

# Calculate moving averages on the filtered dataset
df_filtered["MA_Short"] = df_filtered["Price"].rolling(window=ma_short).mean()
df_filtered["MA_Long"] = df_filtered["Price"].rolling(window=ma_long).mean()

fig6, ax6 = plt.subplots(figsize=(12, 5))
ax6.plot(df_filtered["Time of Scraping"], df_filtered["Price"], label="Price", color="blue")
ax6.plot(df_filtered["Time of Scraping"], df_filtered["MA_Short"], label=f"{ma_short}-period MA", color="red")
ax6.plot(df_filtered["Time of Scraping"], df_filtered["MA_Long"], label=f"{ma_long}-period MA", color="green")
ax6.set_title(f"{coin_choice} Price with Moving Averages")
ax6.set_xlabel("Time")
ax6.set_ylabel("Price (USD)")
ax6.legend()
ax6.grid(True)
st.pyplot(fig6)

# --------------------------
# Graph 7: Volatility - Bollinger Bands
st.subheader(f"📊 {coin_choice} Volatility (Bollinger Bands)")
st.write(
    f"Bollinger Bands for {coin_choice} are computed using a 24-period moving average plus and minus two standard deviations to illustrate price volatility.")

price_series = df_filtered["Price"]
ma = price_series.rolling(window=24).mean()
std = price_series.rolling(window=24).std()
upper_band = ma + (2 * std)
lower_band = ma - (2 * std)

fig7, ax7 = plt.subplots(figsize=(12, 5))
ax7.plot(df_filtered["Time of Scraping"], price_series, label="Price", color="blue")
ax7.plot(df_filtered["Time of Scraping"], ma, label="24-Period MA", color="orange")
ax7.fill_between(df_filtered["Time of Scraping"], lower_band, upper_band, color="gray", alpha=0.3,
                 label="Bollinger Bands")
ax7.set_title(f"{coin_choice} Price Volatility with Bollinger Bands")
ax7.set_xlabel("Time")
ax7.set_ylabel("Price (USD)")
ax7.legend()
ax7.grid(True)
st.pyplot(fig7)

# --------------------------
# Graph 8: Bubble Chart (Price vs Market Cap)
st.subheader(f"📊 {coin_choice} Price vs Market Cap (Bubble Chart)")
st.write(
    f"This bubble chart plots {coin_choice} Price against Market Cap. The size of each bubble represents the Circulating Supply.")
fig8, ax8 = plt.subplots(figsize=(12, 5))
bubble_sizes = df_filtered["Circulating Supply"] / df_filtered["Circulating Supply"].max() * 200  # scale bubble sizes
scatter = ax8.scatter(df_filtered["Price"], df_filtered["Market Cap"], s=bubble_sizes, alpha=0.5, color="teal")
ax8.set_title(f"{coin_choice} Price vs Market Cap")
ax8.set_xlabel("Price (USD)")
ax8.set_ylabel("Market Cap (USD)")
st.pyplot(fig8)

# --------------------------
# Graph 9: Time Series Decomposition
st.subheader(f"📊 {coin_choice} Time Series Decomposition of Price")
st.write(
    f"The decomposition breaks the price time series of {coin_choice} into trend, seasonal, and residual components. (Requires regularly spaced data; if your data is irregular, consider resampling.)")
# Resample data to hourly means (if not already regular)
df_resampled = df_filtered.set_index("Time of Scraping").resample('H').mean().dropna()
try:
    decomp = seasonal_decompose(df_resampled["Price"], period=24)  # assuming daily seasonality
    fig9, (ax_trend, ax_seasonal, ax_resid) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax_trend.plot(decomp.trend, color="blue")
    ax_trend.set_title("Trend Component")
    ax_seasonal.plot(decomp.seasonal, color="orange")
    ax_seasonal.set_title("Seasonal Component")
    ax_resid.plot(decomp.resid, color="green")
    ax_resid.set_title("Residual Component")
    plt.xlabel("Time")
    st.pyplot(fig9)
except Exception as e:
    st.error(f"Error in time series decomposition: {e}")

# --------------------------
# Graph 10: Statistical Summary Panel
st.subheader(f"📊 {coin_choice} Statistical Summary")
st.write(
    f"Below is a summary of key statistics for {coin_choice}'s Price, Market Cap, and 24h Trading Volume based on the filtered data.")
summary_df = df_filtered[["Price", "Market Cap", "24h Trading Volume"]].describe().T
st.dataframe(summary_df)

# --------------------------
# Interactive Data Table
st.subheader(f"🔎 {coin_choice} Data Table")
st.write(
    f"The table below shows the filtered {coin_choice} data. You can sort and filter the table for specific insights.")
st.dataframe(df_filtered)

# End of the dashboard
st.write("📈 Explore and interact with the data by adjusting the filters in the sidebar!")
