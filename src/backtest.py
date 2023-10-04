import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def backtest_hw_strategy(data):
    # Fit the Holt-Winters model to the data
    model_hw = ExponentialSmoothing(data, trend="add", seasonal="add", seasonal_periods=24)
    model_hw_fit = model_hw.fit()
    
    # Forecast the next price for each time step
    hw_forecast = model_hw_fit.predict(start=1, end=len(data))
    
    # Initialize metrics and state
    n_trades = 0
    n_winners = 0
    n_losers = 0
    total_profit = 0
    holding = False
    buy_price = 0

    # Iterate through the data and implement the trading strategy
    for i in range(1, len(data)):
        current_price = data.iloc[i]
        forecasted_price = hw_forecast.iloc[i]
        if forecasted_price > current_price and not holding:
            # Buy
            buy_price = current_price
            holding = True
        elif forecasted_price < current_price and holding:
            # Sell
            profit = current_price - buy_price
            total_profit += profit
            n_trades += 1
            if profit > 0:
                n_winners += 1
            else:
                n_losers += 1
            holding = False
            
    return n_trades, n_winners, n_losers, total_profit

# Load the BTC/USD dataset
data = pd.read_csv('btcusd.csv')
prices = data['price']

# Backtest the strategy on the entire BTC/USD dataset
n_trades, n_winners, n_losers, total_profit = backtest_hw_strategy(prices)
print(f"Number of trades: {n_trades}")
print(f"Number of winning trades: {n_winners}")
print(f"Number of losing trades: {n_losers}")
print(f"Total profit: {total_profit}")

