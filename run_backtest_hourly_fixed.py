#!/usr/bin/env python3
"""
FIXED BACKTEST: Uses hourly bars like the live system
"""
import subprocess
import json
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import sys

# Add src to path
sys.path.append('src')
from data.loader import parse_log_file


def prepare_hourly_data(days_back=30):
    """Load and resample data to hourly bars"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"Loading {days_back} days of data...")
    df = parse_log_file('btcusd.log', start_date=start_date, end_date=end_date)
    
    if df is None or len(df) == 0:
        raise ValueError("No data loaded")
    
    # Ensure datetime index
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
    
    print(f"Loaded {len(df):,} ticks")
    
    # CRITICAL: Resample to hourly bars
    print("Resampling to hourly bars...")
    df_hourly = df.resample('1H').agg({
        'price': ['first', 'max', 'min', 'last'],
        'amount': 'sum'
    })
    df_hourly.columns = ['open', 'high', 'low', 'close', 'volume']
    df_hourly = df_hourly.dropna()
    
    print(f"Created {len(df_hourly):,} hourly bars")
    
    # Save to temp file for backtest
    hourly_file = 'btcusd_hourly_temp.csv'
    df_hourly.to_csv(hourly_file)
    
    return hourly_file, df_hourly


def run_backtest_on_hourly_data(df_hourly, ma_short, ma_long):
    """Run simple backtest on hourly data"""
    # Add MAs
    df = df_hourly.copy()
    df['MA_short'] = df['close'].rolling(window=ma_short).mean()
    df['MA_long'] = df['close'].rolling(window=ma_long).mean()
    
    # Skip NaN values
    df = df.dropna()
    
    # Generate signals
    df['signal'] = 0
    df.loc[df['MA_short'] > df['MA_long'], 'signal'] = 1
    df.loc[df['MA_short'] < df['MA_long'], 'signal'] = -1
    
    # Simulate trading
    position = 0
    balance = 10000
    btc = 0
    trades = []
    fee_rate = 0.0012
    
    for i in range(1, len(df)):
        prev_signal = df.iloc[i-1]['signal']
        curr_signal = df.iloc[i]['signal']
        
        if prev_signal != curr_signal and curr_signal != 0:
            price = df.iloc[i]['close']
            timestamp = df.index[i]
            
            if curr_signal == 1 and position <= 0:
                # BUY
                if balance > 0:
                    fee = balance * fee_rate
                    btc = (balance - fee) / price
                    balance = 0
                    position = 1
                    trades.append({
                        'time': timestamp,
                        'type': 'BUY',
                        'price': price,
                        'btc': btc
                    })
            elif curr_signal == -1 and position >= 0:
                # SELL
                if btc > 0:
                    usd = btc * price
                    fee = usd * fee_rate
                    balance = usd - fee
                    btc = 0
                    position = -1
                    trades.append({
                        'time': timestamp,
                        'type': 'SELL',
                        'price': price,
                        'usd': balance
                    })
    
    # Final value
    final_value = balance if btc == 0 else btc * df.iloc[-1]['close']
    total_return = ((final_value - 10000) / 10000) * 100
    
    return {
        'ma_short': ma_short,
        'ma_long': ma_long,
        'trades': len(trades),
        'return': total_return,
        'final_value': final_value,
        'trade_list': trades
    }


def main():
    print("=" * 60)
    print("FIXED BACKTEST - USING HOURLY BARS")
    print("=" * 60)
    print(f"Period: Last 30 days")
    print(f"Note: This matches how the live system processes data!")
    print("=" * 60)
    
    # Load and prepare hourly data once
    hourly_file, df_hourly = prepare_hourly_data(days_back=30)
    
    # Test configurations
    test_configs = [
        (6, 34),   # Current
        (5, 15),   # Very fast
        (8, 21),   # Fibonacci
        (10, 20),  # Balanced
        (10, 30),  # Medium
        (12, 26),  # MACD-like
        (15, 30),  # Medium-slow
        (20, 50),  # Classic
    ]
    
    results = []
    
    print("\nRunning backtests on hourly data...")
    print("-" * 60)
    
    for ma_short, ma_long in test_configs:
        print(f"\nTesting MA {ma_short}/{ma_long}...", end='', flush=True)
        
        start_time = time.time()
        result = run_backtest_on_hourly_data(df_hourly, ma_short, ma_long)
        elapsed = time.time() - start_time
        
        results.append(result)
        print(f" Done in {elapsed:.1f}s")
        print(f"   Trades: {result['trades']} | Return: {result['return']:+.2f}%")
    
    # Sort by return
    results.sort(key=lambda x: x['return'], reverse=True)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS (HOURLY BARS - MATCHING LIVE SYSTEM)")
    print("=" * 60)
    print(f"{'Rank':<5} {'MA Config':>12} {'Return':>10} {'Trades':>8} {'Trade/Day':>10}")
    print("-" * 45)
    
    for i, r in enumerate(results, 1):
        trades_per_day = r['trades'] / 30.0
        print(f"{i:<5} {r['ma_short']:>5}/{r['ma_long']:<5} {r['return']:>9.2f}% {r['trades']:>8} {trades_per_day:>10.2f}")
    
    # Show best result details
    if results:
        best = results[0]
        print(f"\n🏆 WINNER: MA {best['ma_short']}/{best['ma_long']}")
        print(f"   Return: {best['return']:.2f}%")
        print(f"   Total trades: {best['trades']}")
        print(f"   Avg trades per day: {best['trades']/30:.2f}")
        
        # Show last few trades
        if best['trade_list']:
            print(f"\n   Last 5 trades:")
            for trade in best['trade_list'][-5:]:
                print(f"   {trade['time'].strftime('%Y-%m-%d %H:%M')} - {trade['type']} @ ${trade['price']:,.0f}")
    
    # Clean up
    if os.path.exists(hourly_file):
        os.remove(hourly_file)
    
    print("\n" + "=" * 60)
    print("NOTE: These results now match what the live system would do!")
    print("The previous tick-based backtest was completely wrong.")
    print("=" * 60)


if __name__ == "__main__":
    main()