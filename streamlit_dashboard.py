import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

st.set_page_config(page_title="Ethereum Dashboard", layout="wide")
st.title("🚀 Ethereum Forecast & Performance Dashboard")

# ----------------------
# 🧹 Data Cleaning Logic
# ----------------------
def run_data_cleaning():
    df = pd.read_csv('ethereum_data.csv')
    df['Time of Scraping'] = pd.to_datetime(df['Time of Scraping'])

    def clean_numeric(col):
        return col.replace({r'\$': '', ',': '', '%': ''}, regex=True).astype(float)

    for col in ['Price', 'Market Cap', '24h Trading Volume', 'Circulating Supply',
                '1h Price Change %', '24h Price Change %', '7d Price Change %']:
        df[col] = clean_numeric(df[col])

    range_split = df['24h Range (High/Low)'].str.extract(r'\$(?P<Low>[\d,.]+)\s*-\s*\$(?P<High>[\d,.]+)')
    df['24h Low'] = range_split['Low'].str.replace(',', '').astype(float)
    df['24h High'] = range_split['High'].str.replace(',', '').astype(float)
    df['24h Avg Price'] = df[['24h Low', '24h High']].mean(axis=1)
    df['Price Change Since Last'] = df['Price'].diff()

    df_cleaned = df[['Time of Scraping', 'Price', 'Price Change Since Last', '24h Low', '24h High', '24h Avg Price',
                     'Market Cap', '24h Trading Volume', 'Circulating Supply',
                     '1h Price Change %', '24h Price Change %', '7d Price Change %']]
    df_cleaned.to_csv('ethereum_data_cleaned.csv', index=False)
    print("✅ Data cleaned and saved!")

# --------------------------
# 🤖 Machine Learning Model
# --------------------------
def run_model_prediction():
    df = pd.read_csv('ethereum_data_cleaned.csv')
    df['Time of Scraping'] = pd.to_datetime(df['Time of Scraping'])
    df.dropna(inplace=True)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    features = ['Price', '24h Avg Price', 'Market Cap', '24h Trading Volume',
                '1h Price Change %', '24h Price Change %', '7d Price Change %']
    df['Target'] = df['Price'].shift(-5)
    df.dropna(inplace=True)

    kbin = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
    df['Target_Class'] = kbin.fit_transform(df[['Target']]).astype(int)

    X = df[features]
    y_class = df['Target_Class']
    y_reg = df['Target']

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_class_train, y_class_test = y_class.iloc[:split], y_class.iloc[split:]
    y_reg_train, y_reg_test = y_reg.iloc[:split], y_reg.iloc[split:]
    dates_test = df['Time of Scraping'].iloc[split:]

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_class_train)
    nb_pred = nb_model.predict(X_test)
    acc = (nb_pred == y_class_test).mean()

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_reg_train)
    rf_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_reg_test, rf_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, rf_pred))

    result = df.iloc[split:].copy()
    result['RF_Predicted'] = rf_pred
    result['NB_Class_Predicted'] = nb_pred
    result.to_csv('ethereum_predicted.csv', index=False)

    fig_pred, ax_pred = plt.subplots(figsize=(12, 5))
    ax_pred.plot(dates_test, y_reg_test.values, label="Actual", color='black')
    ax_pred.plot(dates_test, rf_pred, label="Predicted", linestyle='--', color='green')
    ax_pred.set_title("ETH Price Prediction (Random Forest)")
    ax_pred.legend()
    ax_pred.grid(True)
    fig_pred.savefig('eth_prediction_graph.png')

    combined_result_text = f"Naive Bayes Accuracy: {acc:.2f}\nRandom Forest MAE: {mae:.2f} | RMSE: {rmse:.2f}"
    print(combined_result_text)
    with open('ml_results.txt', 'w') as f:
        f.write(combined_result_text + '\n')

    print("✅ Prediction complete and saved, graph exported!")

# Auto-run cleaning and model
run_data_cleaning()
run_model_prediction()

# -------------------------
# 📊 Load Data & Display
# -------------------------
try:
    df_cleaned = pd.read_csv('data/ethereum_data_cleaned.csv')
    df_pred = pd.read_csv('data/ethereum_predicted.csv')
    df_cleaned['Time of Scraping'] = pd.to_datetime(df_cleaned['Time of Scraping'])
    df_pred['Time of Scraping'] = pd.to_datetime(df_pred['Time of Scraping'])

   # ========== FILTERS ========== #
    st.sidebar.header("🔍 Filters")

    reset_clicked = st.sidebar.button("🔄 Reset Filters", key="reset_button")
    show_all_clicked = st.sidebar.button("📊 Show All Data Without Filters", key="show_all_button")

    if reset_clicked:
        st.rerun()

    if show_all_clicked:
        df_cleaned = pd.read_csv('data/ethereum_data_cleaned.csv')
        df_cleaned['Time of Scraping'] = pd.to_datetime(df_cleaned['Time of Scraping'])

        st.subheader("📊 Price Over Time")
        fig_price, ax_price = plt.subplots(figsize=(12, 5))
        ax_price.plot(df_cleaned['Time of Scraping'], df_cleaned['Price'])
        ax_price.set_title("Price Over Time")
        ax_price.grid(True)
        st.pyplot(fig_price)

        st.subheader("📊 Market Cap Over Time")
        fig_mc, ax_mc = plt.subplots(figsize=(12, 5))
        ax_mc.plot(df_cleaned['Time of Scraping'], df_cleaned['Market Cap'])
        ax_mc.set_title("Market Cap Over Time")
        ax_mc.grid(True)
        st.pyplot(fig_mc)

        st.subheader("📊 24h Trading Volume Over Time")
        fig_vol, ax_vol = plt.subplots(figsize=(12, 5))
        ax_vol.plot(df_cleaned['Time of Scraping'], df_cleaned['24h Trading Volume'])
        ax_vol.set_title("24h Volume Over Time")
        ax_vol.grid(True)
        st.pyplot(fig_vol)

        st.subheader("🔁 % Change: 24h and 7d")
        fig_change, ax_change = plt.subplots(figsize=(12, 5))
        ax_change.plot(df_cleaned['Time of Scraping'], df_cleaned['24h Price Change %'], label='24h Change %', color='red')
        ax_change.plot(df_cleaned['Time of Scraping'], df_cleaned['7d Price Change %'], label='7d Change %', color='blue')
        ax_change.set_title("Price Change Over Time")
        ax_change.legend()
        ax_change.grid(True)
        st.pyplot(fig_change)

        st.subheader("📌 Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        corr_df = df_cleaned[['Price', 'Market Cap', '24h Trading Volume', 'Circulating Supply',
                            '1h Price Change %', '24h Price Change %', '7d Price Change %']]
        sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax_corr)
        ax_corr.set_title("Feature Correlation")
        st.pyplot(fig_corr)

        st.stop()  # prevent rest of app from running

    else:
        # Price range filter
        min_price = float(df_cleaned['Price'].min())
        max_price = float(df_cleaned['Price'].max())
        price_range = st.sidebar.slider("💵 Select Price Range (USD)", min_value=min_price, max_value=max_price, value=(min_price, max_price))
        df_cleaned = df_cleaned[(df_cleaned['Price'] >= price_range[0]) & (df_cleaned['Price'] <= price_range[1])]

        # Date filter
        min_date = df_cleaned['Time of Scraping'].min()
        max_date = df_cleaned['Time of Scraping'].max()
        date_range = st.sidebar.date_input("📅 Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
        if len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df_cleaned = df_cleaned[(df_cleaned['Time of Scraping'] >= start_date) & (df_cleaned['Time of Scraping'] <= end_date)]

    # Metric selection
    metric = st.sidebar.selectbox("📊 Select Metric to Visualize", ["Price", "Market Cap", "24h Trading Volume"])

    # Change % selector
    change_option = st.sidebar.radio("🔁 Show % Change", ["24h", "7d"])

    # Correlation toggle outside both conditions
    show_corr = st.sidebar.checkbox("📌 Show Correlation Heatmap", True)


    # ========== VISUALS ========== #

    st.subheader("📈 ETH Price vs Prediction")
    st.image("data/eth_prediction_graph.png", caption="Actual vs Predicted ETH Price (5-day forecast)", use_container_width=True)

    # Investment Recommendation
    if df_pred.empty:
        st.warning("⚠️ No prediction data available.")
    else:
        latest = df_pred.iloc[-1]
        current_price = latest['Price']
        predicted_price = latest['RF_Predicted']
        gain = ((predicted_price - current_price) / current_price) * 100

    st.subheader("💰 Investment Recommendation")
    st.write(f"Current: ${current_price:.2f} | Predicted: ${predicted_price:.2f} | Gain: {gain:.2f}%")
    if gain > 0:
        st.success("BUY")
    elif gain < 0:
        st.error("SELL")
    else:
        st.info("HOLD")

    # Volatility
    st.subheader("📉 Volatility")
    volatility = df_cleaned['Price'].std()
    st.write(f"Volatility (Std Dev): {volatility:.2f} USD")

    # Selected Metric Plot
    st.subheader(f"📊 {metric} Over Time")
    fig_metric, ax_metric = plt.subplots(figsize=(12, 5))
    ax_metric.plot(df_cleaned['Time of Scraping'], df_cleaned[metric])
    ax_metric.set_title(f"{metric} Over Time")
    ax_metric.grid(True)
    st.pyplot(fig_metric)

    # % Change Plot
    st.subheader("🔁 Price % Change")
    fig_change, ax_change = plt.subplots(figsize=(12, 5))

    if change_option == "Both":
        ax_change.plot(df_cleaned['Time of Scraping'], df_cleaned['24h Price Change %'], label='24h Change %', color='red')
        ax_change.plot(df_cleaned['Time of Scraping'], df_cleaned['7d Price Change %'], label='7d Change %', color='blue')
        ax_change.legend()
    else:
        column = '24h Price Change %' if change_option == '24h' else '7d Price Change %'
        color = 'red' if change_option == '24h' else 'blue'
        ax_change.plot(df_cleaned['Time of Scraping'], df_cleaned[column], color=color)

    ax_change.set_title("Price % Change")
    ax_change.grid(True)
    st.pyplot(fig_change)



    # Correlation Heatmap
    if show_corr:
        st.subheader("📌 Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        corr_df = df_cleaned[['Price', 'Market Cap', '24h Trading Volume', 'Circulating Supply',
                              '1h Price Change %', '24h Price Change %', '7d Price Change %']]
        sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax_corr)
        ax_corr.set_title("Feature Correlation")
        st.pyplot(fig_corr)

except FileNotFoundError:
    st.warning("⚠️ Please ensure the data files exist before running the dashboard.")
