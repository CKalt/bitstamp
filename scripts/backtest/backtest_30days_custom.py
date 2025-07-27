#!/usr/bin/env python3
"""
Custom backtest script to achieve 1-2 trades per day average
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.loader import parse_log_file
from indicators.technical_indicators import (
    ensure_datetime_index, 
    add_moving_averages,
    generate_ma_signals,
    calculate_rsi,
    generate_rsi_signals
)

def run_backtest_with_config(config_params):
    """Run backtest with specific parameters"""
    
    # Load data for last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"Loading data from {start_date} to {end_date}")
    df = parse_log_file('btcusd.log', start_date, end_date)
    
    if df.empty:
        print("No data loaded!")
        return None
        
    # Ensure datetime index
    df = ensure_datetime_index(df)
    
    # Resample to specified frequency
    frequency = config_params.get('Frequency', '1H')
    df_resampled = df['price'].resample(frequency).ohlc()
    df_resampled['volume'] = df['amount'].resample(frequency).sum()
    df_resampled = df_resampled.dropna()
    
    # Add timestamp column for compatibility
    df_resampled['timestamp'] = df_resampled.index.astype(np.int64) // 10**9
    
    # Add moving averages
    short_window = config_params.get('Short_Window', 6)
    long_window = config_params.get('Long_Window', 34)
    df_resampled = add_moving_averages(df_resampled, short_window, long_window, 'close')
    
    # Generate signals
    df_resampled = generate_ma_signals(df_resampled)
    signals = df_resampled['MA_Signal']
    
    # Calculate trades
    position = 0
    trades = []
    initial_balance = 10000
    balance = initial_balance
    btc_balance = 0
    fee_rate = 0.0012  # Bitstamp fee
    
    for i in range(len(signals)):
        signal = signals.iloc[i]
        price = df_resampled['close'].iloc[i]
        
        if signal == 1 and position != 1:  # Buy signal
            # Buy BTC with all USD
            btc_amount = (balance * (1 - fee_rate)) / price
            btc_balance = btc_amount
            balance = 0
            position = 1
            trades.append({
                'date': df_resampled.index[i],
                'action': 'BUY',
                'price': price,
                'amount': btc_amount
            })
            
        elif signal == -1 and position != -1:  # Sell signal
            # Sell all BTC
            if btc_balance > 0:
                usd_amount = btc_balance * price * (1 - fee_rate)
                balance = usd_amount
                btc_balance = 0
                position = -1
                trades.append({
                    'date': df_resampled.index[i],
                    'action': 'SELL',
                    'price': price,
                    'amount': btc_balance
                })
    
    # Calculate final balance
    if position == 1 and btc_balance > 0:
        # Still holding BTC, convert to USD
        final_balance = btc_balance * df_resampled['close'].iloc[-1]
    else:
        final_balance = balance
        
    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    total_trades = len(trades)
    trading_days = (end_date - start_date).days
    avg_trades_per_day = total_trades / trading_days if trading_days > 0 else 0
    
    return {
        'config': config_params,
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_return': total_return,
        'total_trades': total_trades,
        'avg_trades_per_day': avg_trades_per_day,
        'trades': trades
    }

def find_optimal_parameters():
    """Find parameters that result in 1-2 trades per day"""
    
    # Test different parameter combinations
    test_configs = []
    
    # Vary the MA windows to find optimal trade frequency
    for short in range(4, 12, 2):  # 4, 6, 8, 10
        for long in range(20, 50, 5):  # 20, 25, 30, 35, 40, 45
            if long > short * 2:  # Ensure reasonable separation
                test_configs.append({
                    'Frequency': '1H',
                    'Short_Window': short,
                    'Long_Window': long,
                    'Strategy': 'MA'
                })
    
    # Also test some 2H and 4H frequencies for lower trade counts
    for freq in ['2H', '4H']:
        for short in [6, 8]:
            for long in [20, 30, 40]:
                test_configs.append({
                    'Frequency': freq,
                    'Short_Window': short,
                    'Long_Window': long,
                    'Strategy': 'MA'
                })
    
    results = []
    print(f"Testing {len(test_configs)} configurations...")
    
    for i, config in enumerate(test_configs):
        print(f"\rTesting config {i+1}/{len(test_configs)}", end='')
        result = run_backtest_with_config(config)
        if result and 0.5 <= result['avg_trades_per_day'] <= 2.5:
            results.append(result)
    
    print("\n\nConfigurations with 1-2 trades per day average:")
    print("-" * 80)
    
    # Sort by total return
    results.sort(key=lambda x: x['total_return'], reverse=True)
    
    for result in results[:10]:  # Show top 10
        print(f"Frequency: {result['config']['Frequency']}, "
              f"MA: {result['config']['Short_Window']}/{result['config']['Long_Window']}, "
              f"Return: {result['total_return']:.2f}%, "
              f"Trades/Day: {result['avg_trades_per_day']:.2f}, "
              f"Total Trades: {result['total_trades']}")
    
    # Save best result
    if results:
        best = results[0]
        with open('backtest_30days_optimal.json', 'w') as f:
            json.dump(best, f, indent=2, default=str)
        print(f"\nBest configuration saved to backtest_30days_optimal.json")
        return best
    
    return None

# Compare with current deployed strategy
print("Running backtest with deployed parameters...")
deployed_config = {
    "Frequency": "1H",
    "Strategy": "MA",
    "Short_Window": 6,
    "Long_Window": 34
}

deployed_result = run_backtest_with_config(deployed_config)
if deployed_result:
    print(f"\nDeployed strategy results (30 days):")
    print(f"Total Return: {deployed_result['total_return']:.2f}%")
    print(f"Trades per day: {deployed_result['avg_trades_per_day']:.2f}")
    print(f"Total trades: {deployed_result['total_trades']}")
    
    with open('backtest_30days_deployed.json', 'w') as f:
        json.dump(deployed_result, f, indent=2, default=str)

# Find optimal parameters
print("\n" + "="*80)
print("Searching for optimal parameters (1-2 trades/day)...")
optimal = find_optimal_parameters()

if optimal and deployed_result:
    print("\n" + "="*80)
    print("COMPARISON SUMMARY:")
    print(f"Deployed: {deployed_result['total_return']:.2f}% return, "
          f"{deployed_result['avg_trades_per_day']:.2f} trades/day")
    print(f"Optimal:  {optimal['total_return']:.2f}% return, "
          f"{optimal['avg_trades_per_day']:.2f} trades/day")
    
    if optimal['total_return'] > deployed_result['total_return']:
        print(f"\nOptimal configuration shows {optimal['total_return'] - deployed_result['total_return']:.2f}% "
              f"better return!")