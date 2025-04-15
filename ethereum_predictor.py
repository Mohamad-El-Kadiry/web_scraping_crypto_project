import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import timedelta

# Load and prepare Ethereum dataset
df = pd.read_csv("ethereum_data_cleaned.csv")
df["Time of Scraping"] = pd.to_datetime(df["Time of Scraping"])
df = df.sort_values("Time of Scraping").reset_index(drop=True)

# Extract time features
df["hour"] = df["Time of Scraping"].dt.hour
df["dayofweek"] = df["Time of Scraping"].dt.dayofweek
df["day"] = df["Time of Scraping"].dt.day
df["month"] = df["Time of Scraping"].dt.month

# Define features and target
features = ['Price Change Since Last', '24h Low', '24h High', '24h Avg Price',
            'Market Cap', '24h Trading Volume', 'Circulating Supply',
            '1h Price Change %', '24h Price Change %', '7d Price Change %',
            'hour', 'dayofweek', 'day', 'month']
target = 'Price'

# Train/test split (we're training on all available data)
X = df[features]
y = df[target]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_scaled, y)

# Forecast next 120 hours (5 days hourly forecast)
last_row = df.iloc[-1].copy()
future_prices = []
future_times = []
current_time = last_row["Time of Scraping"]

for i in range(120):  # 5 days * 24 hours
    next_time = current_time + timedelta(hours=1)
    future_times.append(next_time)

    # Time-based features
    hour = next_time.hour
    dayofweek = next_time.dayofweek
    day = next_time.day
    month = next_time.month

    # Prepare input row
    input_row = last_row.copy()
    input_row["hour"] = hour
    input_row["dayofweek"] = dayofweek
    input_row["day"] = day
    input_row["month"] = month

    # Simulated updates
    input_row["Price Change Since Last"] = input_row["Price"] - last_row["Price"]
    input_row["24h Low"] = last_row["24h Low"] * (1 + np.random.normal(0, 0.01))
    input_row["24h High"] = last_row["24h High"] * (1 + np.random.normal(0, 0.01))
    input_row["24h Avg Price"] = (input_row["24h Low"] + input_row["24h High"]) / 2
    input_row["Market Cap"] = last_row["Market Cap"] * (1 + np.random.normal(0, 0.01))
    input_row["24h Trading Volume"] = last_row["24h Trading Volume"] * (1 + np.random.normal(0, 0.01))
    input_row["Circulating Supply"] = last_row["Circulating Supply"] * (1 + np.random.normal(0, 0.005))

    # Prediction
    input_scaled = scaler.transform(pd.DataFrame([input_row[features].values], columns=features))
    predicted_price = model.predict(input_scaled)[0]
    future_prices.append(predicted_price)

    # Update for next iteration
    input_row["Price"] = predicted_price
    last_row = input_row
    current_time = next_time

# Plot the forecast
plt.figure(figsize=(14, 6))
plt.plot(df["Time of Scraping"], df["Price"], label="Historical ETH Price")
plt.plot(future_times, future_prices, label="Predicted ETH Price (Next 5 Days)", linestyle="--", color='red')
plt.xlabel("Time")
plt.ylabel("Ethereum Price")
plt.title("Ethereum Price Forecast: Hourly for Next 5 Days")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
