import sys
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Parameters to tweak
FORECAST_HORIZON = 10  # Forecast 10 minutes into the future
# Threshold for significant price change (this can be adjusted based on your data's scale)
PRICE_CHANGE_THRESHOLD = 0.5
STOP_LOSS = -10  # Stop loss threshold
TAKE_PROFIT = 10  # Take profit threshold


def backtest_hw_strategy(data, timestamps):
    # Fit the Holt-Winters model to the data
    model_hw = ExponentialSmoothing(
        data, trend="add", seasonal="add", seasonal_periods=24)
    model_hw_fit = model_hw.fit()

    # Forecast the next price for each time step
    hw_forecast = model_hw_fit.predict(
        start=FORECAST_HORIZON, end=len(data) + FORECAST_HORIZON - 1)

    # Initialize metrics and state
    holding = False
    buy_price = 0
    buy_time = None
    trades = []

    # Iterate through the data and implement the trading strategy
    for i in range(len(data) - FORECAST_HORIZON):
        current_price = data.iloc[i]
        forecasted_price = hw_forecast.iloc[i]
        price_difference = forecasted_price - current_price

        if not holding and price_difference > PRICE_CHANGE_THRESHOLD:
            # Buy Entry Signal
            buy_price = current_price
            buy_time = timestamps.iloc[i]
            holding = True
        elif holding:
            profit = current_price - buy_price
            exit_reason = "Forecast"  # Default exit reason
            if profit <= STOP_LOSS:
                exit_reason = "Stop Loss"
            elif profit >= TAKE_PROFIT:
                exit_reason = "Take Profit"

            # Sell when forecast is negative, or due to stop loss/take profit
            if exit_reason != "Forecast" or price_difference < -PRICE_CHANGE_THRESHOLD:
                duration_seconds = int(timestamps.iloc[i]) - int(buy_time)
                duration_minutes, duration_seconds = divmod(
                    duration_seconds, 60)
                trades.append({
                    "buy_time": buy_time,
                    "sell_time": timestamps.iloc[i],
                    "buy_price": buy_price,
                    "sell_price": current_price,
                    "profit": profit,
                    "duration": f"{duration_minutes}:{duration_seconds:02}",
                    "exit_reason": exit_reason
                })
                holding = False

    return trades


# Check if a CSV file path is provided as an argument
if len(sys.argv) < 2:
    print("Usage: python script_name.py path_to_csv_file")
    sys.exit()

# Load the BTC/USD dataset from the provided argument
csv_file_path = sys.argv[1]
data = pd.read_csv(csv_file_path)
prices = data['price']
timestamps = data['timestamp']

# Backtest the strategy on the entire BTC/USD dataset
trades = backtest_hw_strategy(prices, timestamps)

# Print the trade details
for trade in trades:
    print(trades[0])  # This will print the details of the first trade.
    print(f"Buy Time: {trade['buy_time']}, Buy Price: {trade['buy_price']}")
    print(
        f"Sell Time: {trade['sell_time']}, Sell Price: {trade['sell_price']}")
    print(
        f"Profit: {trade['profit']}, Duration: {trade['duration']}, Exit Reason: {trade['exit_reason']}")
    print('-' * 50)
