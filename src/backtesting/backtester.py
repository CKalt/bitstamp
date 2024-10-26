# src/backtesting/backtester.py

import pandas as pd
import numpy as np

from indicators.technical_indicators import ensure_datetime_index

def backtest(df, strategy, initial_balance=10000, position_size=0.1, transaction_cost=0.001, max_trades_per_day=10):
    df = ensure_datetime_index(df)
    df['Position'] = df[f'{strategy}_Signal'].shift(1).fillna(0)
    df['Returns'] = df['price'].pct_change().fillna(0)
    df['Strategy_Returns'] = df['Position'] * df['Returns']

    # Calculate transaction costs
    df['Trade'] = df['Position'].diff().abs()
    df['Transaction_Costs'] = df['Trade'] * transaction_cost

    # Limit trades per day
    df['Daily_Trades'] = df['Trade'].groupby(df.index.date).cumsum()
    df.loc[df['Daily_Trades'] > max_trades_per_day, 'Strategy_Returns'] = 0

    df['Strategy_Returns'] -= df['Transaction_Costs']
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Balance'] = initial_balance * df['Cumulative_Returns']

    total_trades = df['Trade'].sum()

    # Calculate average trades per day
    trading_days = (df.index[-1].date() - df.index[0].date()).days + 1
    average_trades_per_day = total_trades / trading_days if trading_days > 0 else 0

    # Profit factor and Sharpe ratio
    positive_returns = df.loc[df['Strategy_Returns'] > 0, 'Strategy_Returns'].sum()
    negative_returns = -df.loc[df['Strategy_Returns'] < 0, 'Strategy_Returns'].sum()
    profit_factor = positive_returns / negative_returns if negative_returns != 0 else np.inf

    sharpe_ratio = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252) if df['Strategy_Returns'].std() != 0 else 0

    final_balance = df['Balance'].iloc[-1]
    total_return = (final_balance - initial_balance) / initial_balance * 100

    return {
        'Final_Balance': final_balance,
        'Total_Return': total_return,
        'Total_Trades': total_trades,
        'Average_Trades_Per_Day': average_trades_per_day,
        'Profit_Factor': profit_factor,
        'Sharpe_Ratio': sharpe_ratio
    }

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
