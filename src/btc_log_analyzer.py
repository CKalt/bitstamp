import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np

def parse_log_file(file_path):
    data = []
    total_lines = sum(1 for _ in open(file_path, 'r'))
    print(f"Total lines in log file: {total_lines}")
    with open(file_path, 'r') as file:
        for i, line in enumerate(file, 1):
            if i % 10000 == 0:
                print(f"Processing line {i}/{total_lines} ({i/total_lines*100:.2f}%)")
            try:
                json_data = json.loads(line)
                if json_data['event'] == 'trade':
                    trade_data = json_data['data']
                    data.append({
                        'timestamp': int(trade_data['timestamp']),
                        'price': float(trade_data['price']),
                        'amount': float(trade_data['amount']),
                        'type': int(trade_data['type'])
                    })
            except json.JSONDecodeError:
                continue
    print("Finished processing log file. Creating DataFrame...")
    return pd.DataFrame(data)

def analyze_data(df):
    print("Converting timestamp to datetime...")
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    print("Calculating basic statistics...")
    print(df.describe())
    print("Plotting price over time...")
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['price'])
    plt.title('Bitcoin Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.savefig('btc_price_over_time.png')
    plt.close()
    print("Calculating and plotting hourly trading volume...")
    df['volume'] = df['price'] * df['amount']
    hourly_volume = df.resample('H', on='datetime')['volume'].sum()
    plt.figure(figsize=(12, 6))
    plt.bar(hourly_volume.index, hourly_volume.values)
    plt.title('Hourly Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume (USD)')
    plt.savefig('btc_hourly_volume.png')
    plt.close()

# Trading System Functions
def add_moving_averages(df, short_window, long_window):
    df['Short_MA'] = df['price'].rolling(window=short_window).mean()
    df['Long_MA'] = df['price'].rolling(window=long_window).mean()
    return df

def generate_ma_signals(df):
    df['MA_Signal'] = 0
    df.loc[df['Short_MA'] > df['Long_MA'], 'MA_Signal'] = 1
    df.loc[df['Short_MA'] < df['Long_MA'], 'MA_Signal'] = -1
    return df

def calculate_rsi(df, window=14):
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def generate_rsi_signals(df, overbought=70, oversold=30):
    df['RSI_Signal'] = 0
    df.loc[df['RSI'] < oversold, 'RSI_Signal'] = 1
    df.loc[df['RSI'] > overbought, 'RSI_Signal'] = -1
    return df

def backtest(df, strategy, initial_balance=10000, position_size=0.1):
    df['Position'] = df[f'{strategy}_Signal'].shift(1)
    df['Returns'] = df['price'].pct_change()
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Balance'] = initial_balance * df['Cumulative_Returns']
    
    total_trades = np.sum(np.abs(df['Position'].diff()))
    profit_factor = np.sum(df['Strategy_Returns'][df['Strategy_Returns'] > 0]) / abs(np.sum(df['Strategy_Returns'][df['Strategy_Returns'] < 0]))
    sharpe_ratio = np.sqrt(252) * df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()
    
    return {
        'Final_Balance': df['Balance'].iloc[-1],
        'Total_Return': (df['Balance'].iloc[-1] - initial_balance) / initial_balance * 100,
        'Total_Trades': total_trades,
        'Profit_Factor': profit_factor,
        'Sharpe_Ratio': sharpe_ratio
    }

def optimize_ma_parameters(df, short_range, long_range):
    results = []
    for short_window in short_range:
        for long_window in long_range:
            if short_window >= long_window:
                continue
            df_test = add_moving_averages(df.copy(), short_window, long_window)
            df_test = generate_ma_signals(df_test)
            metrics = backtest(df_test, 'MA')
            results.append({
                'Strategy': 'MA',
                'Short_Window': short_window,
                'Long_Window': long_window,
                **metrics
            })
    return pd.DataFrame(results)

def optimize_rsi_parameters(df, window_range, overbought_range, oversold_range):
    results = []
    for window in window_range:
        for overbought in overbought_range:
            for oversold in oversold_range:
                if oversold >= overbought:
                    continue
                df_test = calculate_rsi(df.copy(), window)
                df_test = generate_rsi_signals(df_test, overbought, oversold)
                metrics = backtest(df_test, 'RSI')
                results.append({
                    'Strategy': 'RSI',
                    'RSI_Window': window,
                    'Overbought': overbought,
                    'Oversold': oversold,
                    **metrics
                })
    return pd.DataFrame(results)

def run_trading_system(df):
    # Resample data to hourly timeframe
    df_hourly = df.resample('1H', on='datetime').agg({
        'price': 'last',
        'amount': 'sum',
        'volume': 'sum'
    }).dropna()

    # MA Crossover Strategy
    print("Running MA Crossover Strategy...")
    ma_results = optimize_ma_parameters(df_hourly, range(4, 25, 2), range(26, 51, 2))
    best_ma = ma_results.loc[ma_results['Total_Return'].idxmax()]
    print("\nBest MA Crossover parameters:")
    print(best_ma)

    # RSI Strategy
    print("\nRunning RSI Strategy...")
    rsi_results = optimize_rsi_parameters(df_hourly, range(10, 21, 2), range(65, 81, 5), range(20, 36, 5))
    best_rsi = rsi_results.loc[rsi_results['Total_Return'].idxmax()]
    print("\nBest RSI parameters:")
    print(best_rsi)

    # Explicit Strategy Comparison
    print("\nStrategy Comparison:")
    comparison = pd.DataFrame({
        'MA Crossover': best_ma,
        'RSI': best_rsi
    }).T
    print(comparison[['Total_Return', 'Sharpe_Ratio', 'Profit_Factor', 'Total_Trades']])

    # Determine the best overall strategy
    best_strategy = 'MA Crossover' if best_ma['Total_Return'] > best_rsi['Total_Return'] else 'RSI'
    print(f"\nBest overall strategy: {best_strategy}")

    # Combine results
    all_results = pd.concat([ma_results, rsi_results])
    all_results.to_csv('optimization_results.csv', index=False)
    print("\nAll optimization results saved to 'optimization_results.csv'")

    return all_results, comparison

def main():
    file_path = 'btcusd.log'
    file_size = os.path.getsize(file_path) / (1024 * 1024) # Size in MB
    print(f"Log file size: {file_size:.2f} MB")
    print("Starting to parse log file...")
    df = parse_log_file(file_path)
    print(f"Parsed {len(df)} trade events.")
    print("Starting data analysis...")
    analyze_data(df)
    print("Analysis complete. Check the current directory for generated PNG files.")
    
    print("Running trading system...")
    optimization_results, strategy_comparison = run_trading_system(df)
    print("Trading system analysis complete.")
    
    # You can add more analysis or visualization of the results here if needed

if __name__ == "__main__":
    main()