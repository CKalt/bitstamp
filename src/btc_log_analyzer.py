import pandas as pd
import numpy as np
from datetime import timedelta

def add_moving_averages(df, short_window, long_window):
    df['Short_MA'] = df['price'].rolling(window=short_window).mean()
    df['Long_MA'] = df['price'].rolling(window=long_window).mean()
    return df

def generate_signals(df):
    df['Signal'] = 0
    df.loc[df['Short_MA'] > df['Long_MA'], 'Signal'] = 1
    df.loc[df['Short_MA'] < df['Long_MA'], 'Signal'] = -1
    return df

def backtest(df, initial_balance=10000, position_size=0.1):
    df['Position'] = df['Signal'].shift(1)
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

def optimize_parameters(df, short_range, long_range):
    results = []
    for short_window in short_range:
        for long_window in long_range:
            if short_window >= long_window:
                continue
            df_test = add_moving_averages(df.copy(), short_window, long_window)
            df_test = generate_signals(df_test)
            metrics = backtest(df_test)
            results.append({
                'Short_Window': short_window,
                'Long_Window': long_window,
                **metrics
            })
    return pd.DataFrame(results)

def main(df):
    # Resample data to hourly timeframe
    df_hourly = df.resample('1H', on='datetime').agg({
        'price': 'last',
        'amount': 'sum',
        'volume': 'sum'
    }).dropna()

    # Initial test with default parameters
    df_test = add_moving_averages(df_hourly.copy(), short_window=12, long_window=26)
    df_test = generate_signals(df_test)
    initial_results = backtest(df_test)
    print("Initial results:")
    print(initial_results)

    # Optimize parameters
    short_range = range(4, 25, 2)
    long_range = range(26, 51, 2)
    optimization_results = optimize_parameters(df_hourly, short_range, long_range)
    
    best_params = optimization_results.loc[optimization_results['Total_Return'].idxmax()]
    print("\nBest parameters:")
    print(best_params)

    return optimization_results

if __name__ == "__main__":
    # Assuming df is your parsed DataFrame from the previous script
    optimization_results = main(df)
    optimization_results.to_csv('optimization_results.csv', index=False)
    print("\nOptimization results saved to 'optimization_results.csv'")