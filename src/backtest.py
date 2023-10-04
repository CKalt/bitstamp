import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def backtest_hw_strategy(data, timestamps):
    # Fit the Holt-Winters model to the data
    model_hw = ExponentialSmoothing(data, trend="add", seasonal="add", seasonal_periods=24)
    model_hw_fit = model_hw.fit()
    
    # Forecast the next price for each time step
    hw_forecast = model_hw_fit.predict(start=1, end=len(data))
    
    # Initialize metrics and state
    holding = False
    buy_price = 0
    buy_time = None
    trades = []

    # Iterate through the data and implement the trading strategy
    for i in range(1, len(data)):
        current_price = data.iloc[i]
        forecasted_price = hw_forecast.iloc[i]
        if forecasted_price > current_price and not holding:
            # Buy
            buy_price = current_price
            buy_time = timestamps.iloc[i]
            holding = True
        elif forecasted_price < current_price and holding:
            # Sell
            sell_price = current_price
            sell_time = timestamps.iloc[i]
            profit = sell_price - buy_price
            trades.append({
                "buy_time": buy_time,
                "sell_time": sell_time,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "profit": profit
            })
            holding = False
            
    return trades

# Load the BTC/USD dataset
data = pd.read_csv('btcusd.csv')
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
