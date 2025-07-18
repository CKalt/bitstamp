#!/usr/bin/env python3
"""
Quick test of backtesting with 1 day of data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bktst import EnhancedBacktester
from src.data.loader import parse_log_file
import json
import pandas as pd
from datetime import datetime

# Load configuration
config_path = 'best_strategy.json'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    config = {}

# Set defaults
config.setdefault('initial_balance', 10000)
config.setdefault('fee_rate', 0.0012)
config.setdefault('slippage_rate', 0.0005)

# Add some constraints to reduce excessive trading
config['min_trade_gap_minutes'] = 30  # Increase from 15
config['signal_confirmation_bars'] = 3  # Increase from 2

print("Loading data...")
# Load just 1 day of data
start_date = datetime(2025, 6, 29)
end_date = datetime(2025, 6, 30)

df = parse_log_file('btcusd.log', start_date=start_date, end_date=end_date)

if df is None or len(df) == 0:
    print("No data loaded")
    sys.exit(1)

# Convert timestamp to datetime
if 'timestamp' in df.columns:
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
if 'close' not in df.columns and 'price' in df.columns:
    df['close'] = df['price']

print(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")

# Run backtest
print("\nRunning backtest...")
backtester = EnhancedBacktester(config)

# Suppress excessive logging
import logging
backtester.logger.setLevel(logging.WARNING)

try:
    results = backtester.run_backtest(df)
    
    # Print summary
    print("\n" + "="*60)
    print("QUICK BACKTEST RESULTS (1 DAY)")
    print("="*60)
    
    print(f"\nPerformance:")
    print(f"  Initial: ${results['initial_balance']:,.2f}")
    print(f"  Final:   ${results['final_equity']:,.2f}")
    print(f"  Return:  {results['total_return_pct']:.2f}%")
    
    print(f"\nTrading:")
    print(f"  Trades:     {results['total_trades']}")
    print(f"  Win Rate:   {results['win_rate']:.1f}%")
    print(f"  Pivot Trades: {results['pivot_trades']}")
    
    print(f"\nCosts:")
    print(f"  Fees:       ${results['total_fees']:.2f}")
    print(f"  Slippage:   ${results['total_slippage']:.2f}")
    
    # Save results
    with open('test_1day_results.json', 'w') as f:
        # Convert trades to serializable format
        results_copy = results.copy()
        results_copy['trades'] = len(results['trades'])  # Just count
        results_copy['equity_curve'] = len(results['equity_curve'])  # Just count
        json.dump(results_copy, f, indent=2)
    
    print("\nResults saved to test_1day_results.json")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()