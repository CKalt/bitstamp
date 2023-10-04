import sys
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Parameters to tweak
FORECAST_HORIZON = 10  # Forecast 10 minutes into the future
PRICE_CHANGE_THRESHOLD = 0.5  # Threshold for significant price change (this can be adjusted based on your data's scale)
STOP_LOSS = -10  # Stop loss threshold
TAKE_PROFIT = 10  # Take profit threshold

def backtest_hw_strategy(data, timestamps):
    # ... (rest of the function remains unchanged)

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
    print(f"Buy Time: {trade['buy_time']}, Buy Price: {trade['buy_price']}")
    print(f"Sell Time: {trade['sell_time']}, Sell Price: {trade['sell_price']}")
    print(f"Profit: {trade['profit']}")
    print('-' * 50)
