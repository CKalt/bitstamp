#!/usr/bin/env python3
"""
Efficient backtest to find optimal 1-2 trades per day configuration
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
from indicators.technical_indicators import ensure_datetime_index

def backtest_ma_strategy(df_hourly, short_window, long_window, initial_balance=10000):
    """Run MA crossover backtest on pre-loaded hourly data"""
    
    # Calculate moving averages
    df = df_hourly.copy()
    df['MA_short'] = df['close'].rolling(window=short_window).mean()
    df['MA_long'] = df['close'].rolling(window=long_window).mean()
    
    # Generate signals
    df['signal'] = 0
    df.loc[df['MA_short'] > df['MA_long'], 'signal'] = 1
    df.loc[df['MA_short'] < df['MA_long'], 'signal'] = -1
    
    # Track position changes
    df['position'] = df['signal'].diff()
    
    # Simulate trades
    position = 0
    balance = initial_balance
    btc_balance = 0
    trades = []
    fee_rate = 0.0012
    
    for idx in df.index[long_window:]:  # Start after MA warmup
        signal = df.loc[idx, 'signal']
        price = df.loc[idx, 'close']
        
        if position != signal and signal != 0:
            if signal == 1:  # Buy
                btc_amount = (balance * (1 - fee_rate)) / price
                btc_balance = btc_amount
                balance = 0
                trades.append({'date': idx, 'action': 'BUY', 'price': price})
            elif signal == -1 and btc_balance > 0:  # Sell
                usd_amount = btc_balance * price * (1 - fee_rate)
                balance = usd_amount
                btc_balance = 0
                trades.append({'date': idx, 'action': 'SELL', 'price': price})
            position = signal
    
    # Final balance
    if btc_balance > 0:
        final_balance = btc_balance * df['close'].iloc[-1]
    else:
        final_balance = balance
        
    return {
        'final_balance': final_balance,
        'total_return': ((final_balance - initial_balance) / initial_balance) * 100,
        'num_trades': len(trades),
        'trades': trades
    }

# Load data once
print("Loading last 30 days of data...")
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
df = parse_log_file('btcusd.log', start_date, end_date)

# Convert to hourly data
print("Converting to hourly data...")
df = ensure_datetime_index(df)
df_hourly = df['price'].resample('1H').ohlc()
df_hourly['volume'] = df['amount'].resample('1H').sum()
df_hourly = df_hourly.dropna()

print(f"Loaded {len(df_hourly)} hours of data")
trading_days = 30
hours_per_day = 24

# Test deployed configuration
print("\nTesting deployed configuration (MA 6/34)...")
deployed_result = backtest_ma_strategy(df_hourly, 6, 34)
deployed_trades_per_day = deployed_result['num_trades'] / trading_days

print(f"Deployed: {deployed_result['total_return']:.2f}% return, "
      f"{deployed_trades_per_day:.2f} trades/day, "
      f"{deployed_result['num_trades']} total trades")

# Find optimal parameters
print("\nSearching for optimal 1-2 trades/day configuration...")
results = []

for short in range(4, 12):
    for long in range(20, 50, 2):
        if long > short * 2:
            result = backtest_ma_strategy(df_hourly, short, long)
            trades_per_day = result['num_trades'] / trading_days
            
            if 1.0 <= trades_per_day <= 2.0:
                results.append({
                    'short': short,
                    'long': long,
                    'return': result['total_return'],
                    'trades_per_day': trades_per_day,
                    'num_trades': result['num_trades']
                })

# Sort by return
results.sort(key=lambda x: x['return'], reverse=True)

print("\nTop configurations with 1-2 trades/day:")
print("-" * 60)
print("MA Windows | Return % | Trades/Day | Total Trades")
print("-" * 60)

for r in results[:10]:
    print(f"{r['short']:3d}/{r['long']:3d}    | {r['return']:8.2f} | {r['trades_per_day']:10.2f} | {r['num_trades']:12}")

# Save best result
if results:
    best = results[0]
    print(f"\n🎯 Best configuration: MA {best['short']}/{best['long']}")
    print(f"   Return: {best['return']:.2f}%")
    print(f"   Trades per day: {best['trades_per_day']:.2f}")
    
    # Create recommendation
    recommendation = {
        "analysis_date": datetime.now().isoformat(),
        "period": "30 days",
        "deployed_strategy": {
            "short_ma": 6,
            "long_ma": 34,
            "return_pct": deployed_result['total_return'],
            "trades_per_day": deployed_trades_per_day
        },
        "optimal_strategy": {
            "short_ma": best['short'],
            "long_ma": best['long'],
            "return_pct": best['return'],
            "trades_per_day": best['trades_per_day']
        },
        "improvement": {
            "return_gain": best['return'] - deployed_result['total_return'],
            "trade_frequency_match": abs(best['trades_per_day'] - 1.5) < abs(deployed_trades_per_day - 1.5)
        }
    }
    
    with open('backtest_recommendation.json', 'w') as f:
        json.dump(recommendation, f, indent=2)
    
    print("\n✅ Recommendation saved to backtest_recommendation.json")
else:
    print("\n⚠️  No configurations found with 1-2 trades per day")