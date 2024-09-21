import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

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

def calculate_bollinger_bands(df, window=20, num_std=2):
    df['BB_MA'] = df['price'].rolling(window=window).mean()
    df['BB_STD'] = df['price'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_MA'] + (df['BB_STD'] * num_std)
    df['BB_Lower'] = df['BB_MA'] - (df['BB_STD'] * num_std)
    return df

def generate_bollinger_band_signals(df):
    df['BB_Signal'] = 0
    df.loc[df['price'] < df['BB_Lower'], 'BB_Signal'] = 1  # Buy signal
    df.loc[df['price'] > df['BB_Upper'], 'BB_Signal'] = -1  # Sell signal
    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    df['MACD_Fast'] = df['price'].ewm(span=fast, adjust=False).mean()
    df['MACD_Slow'] = df['price'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['MACD_Fast'] - df['MACD_Slow']
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    return df

def generate_macd_signals(df):
    df['MACD_Signal'] = 0
    df.loc[df['MACD'] > df['MACD_Signal'], 'MACD_Signal'] = 1  # Buy signal
    df.loc[df['MACD'] < df['MACD_Signal'], 'MACD_Signal'] = -1  # Sell signal
    return df

def backtest(df, strategy, initial_balance=10000, position_size=0.1, transaction_cost=0.001, max_trades_per_day=10):
    df['Position'] = df[f'{strategy}_Signal'].shift(1)
    df['Returns'] = df['price'].pct_change()
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    
    # Apply transaction costs and limit trades per day
    df['Trade'] = df['Position'].diff().abs()
    df['Cumulative_Trades'] = df['Trade'].groupby(df.index.date).cumsum()
    df.loc[df['Cumulative_Trades'] > max_trades_per_day, 'Trade'] = 0
    df['Transaction_Costs'] = df['Trade'] * transaction_cost
    df['Strategy_Returns'] = df['Strategy_Returns'] - df['Transaction_Costs']
    
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Balance'] = initial_balance * df['Cumulative_Returns']
    
    total_trades = np.sum(df['Trade'])
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

def optimize_hft_parameters(df, strategy, **kwargs):
    results = []
    for params in kwargs['param_grid']:
        df_test = df.copy()
        if strategy == 'BB':
            df_test = calculate_bollinger_bands(df_test, window=params['window'], num_std=params['num_std'])
            df_test = generate_bollinger_band_signals(df_test)
        elif strategy == 'MACD':
            df_test = calculate_macd(df_test, fast=params['fast'], slow=params['slow'], signal=params['signal'])
            df_test = generate_macd_signals(df_test)
        
        metrics = backtest(df_test, strategy)
        results.append({
            'Strategy': strategy,
            **params,
            **metrics
        })
    return pd.DataFrame(results)

def generate_trade_list(df, strategy):
    trades = []
    position = 0
    entry_price = 0
    entry_time = None
    
    for index, row in df.iterrows():
        signal = row[f'{strategy}_Signal']
        
        if signal == 1 and position == 0:  # Enter long position
            position = 1
            entry_price = row['price']
            entry_time = index
        elif signal == -1 and position == 1:  # Exit long position
            exit_price = row['price']
            profit = (exit_price - entry_price) / entry_price
            trades.append({
                'Entry Time': entry_time,
                'Exit Time': index,
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Profit (%)': profit * 100
            })
            position = 0
    
    return pd.DataFrame(trades)

def run_trading_system(df):
    # Resample data to different timeframes
    df_hourly = df.resample('1H', on='datetime').agg({
        'price': 'last',
        'amount': 'sum',
        'volume': 'sum'
    }).dropna()
    
    df_15min = df.resample('15T', on='datetime').agg({
        'price': 'last',
        'amount': 'sum',
        'volume': 'sum'
    }).dropna()

    # Original strategies
    print("Running MA Crossover Strategy...")
    ma_results = optimize_ma_parameters(df_hourly, range(4, 25, 2), range(26, 51, 2))
    best_ma = ma_results.loc[ma_results['Total_Return'].idxmax()]
    print("\nBest MA Crossover parameters:")
    print(best_ma)

    # Generate trade list for best MA strategy
    best_ma_df = add_moving_averages(df_hourly.copy(), best_ma['Short_Window'], best_ma['Long_Window'])
    best_ma_df = generate_ma_signals(best_ma_df)
    ma_trades = generate_trade_list(best_ma_df, 'MA')
    ma_trades.to_csv('ma_trades.csv', index=False)
    print("MA Crossover trades saved to 'ma_trades.csv'")

    print("\nRunning RSI Strategy...")
    rsi_results = optimize_rsi_parameters(df_hourly, range(10, 21, 2), range(65, 81, 5), range(20, 36, 5))
    best_rsi = rsi_results.loc[rsi_results['Total_Return'].idxmax()]
    print("\nBest RSI parameters:")
    print(best_rsi)

    # Generate trade list for best RSI strategy
    best_rsi_df = calculate_rsi(df_hourly.copy(), best_rsi['RSI_Window'])
    best_rsi_df = generate_rsi_signals(best_rsi_df, best_rsi['Overbought'], best_rsi['Oversold'])
    rsi_trades = generate_trade_list(best_rsi_df, 'RSI')
    rsi_trades.to_csv('rsi_trades.csv', index=False)
    print("RSI trades saved to 'rsi_trades.csv'")

    # Higher Frequency Strategies
    print("\nRunning Bollinger Bands Strategy...")
    bb_param_grid = [{'window': w, 'num_std': s} for w in range(10, 31, 5) for s in [1.5, 2, 2.5]]
    bb_results = optimize_hft_parameters(df_15min, 'BB', param_grid=bb_param_grid)
    best_bb = bb_results.loc[bb_results['Total_Return'].idxmax()]
    print("\nBest Bollinger Bands parameters:")
    print(best_bb)

    # Generate trade list for best Bollinger Bands strategy
    best_bb_df = calculate_bollinger_bands(df_15min.copy(), window=best_bb['window'], num_std=best_bb['num_std'])
    best_bb_df = generate_bollinger_band_signals(best_bb_df)
    bb_trades = generate_trade_list(best_bb_df, 'BB')
    bb_trades.to_csv('bb_trades.csv', index=False)
    print("Bollinger Bands trades saved to 'bb_trades.csv'")

    print("\nRunning MACD Strategy...")
    macd_param_grid = [{'fast': f, 'slow': s, 'signal': sig} 
                       for f in [6, 12, 18] for s in [20, 26, 32] for sig in [7, 9, 11]]
    macd_results = optimize_hft_parameters(df_15min, 'MACD', param_grid=macd_param_grid)
    best_macd = macd_results.loc[macd_results['Total_Return'].idxmax()]
    print("\nBest MACD parameters:")
    print(best_macd)

    # Generate trade list for best MACD strategy
    best_macd_df = calculate_macd(df_15min.copy(), fast=best_macd['fast'], slow=best_macd['slow'], signal=best_macd['signal'])
    best_macd_df = generate_macd_signals(best_macd_df)
    macd_trades = generate_trade_list(best_macd_df, 'MACD')
    macd_trades.to_csv('macd_trades.csv', index=False)
    print("MACD trades saved to 'macd_trades.csv'")

    # Strategy Comparison
    print("\nStrategy Comparison:")
    comparison = pd.DataFrame({
        'MA Crossover': best_ma,
        'RSI': best_rsi,
        'Bollinger Bands': best_bb,
        'MACD': best_macd
    }).T
    print(comparison[['Total_Return', 'Sharpe_Ratio', 'Profit_Factor', 'Total_Trades']])

    # Determine the best overall strategy
    best_strategy = comparison['Total_Return'].idxmax()
    print(f"\nBest overall strategy: {best_strategy}")

    # Combine results
    all_results = pd.concat([ma_results, rsi_results, bb_results, macd_results])
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

if __name__ == "__main__":
    main()