#\!/usr/bin/env python3
"""
Backtest different proximity threshold values to see which performs best
Tests on 1-minute bars to match the test server configuration
"""

import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from src.tdr_core.strategies import MACrossoverStrategy
from src.data.loader import parse_log_file

def run_backtest_with_threshold(df, proximity_threshold, initial_usd=10000):
    """
    Run backtest with specific proximity threshold
    Returns final P&L and trade statistics
    """
    strategy = MACrossoverStrategy(
        short_window=4,
        long_window=20,
        proximity_threshold=proximity_threshold,
        do_live_trades=False  # Paper trading
    )
    
    # Initialize with USD (SHORT position to start)
    balance_usd = initial_usd
    balance_btc = 0
    position = -1  # Start SHORT (holding USD)
    entry_price = 0
    
    trades = []
    signals_blocked = 0
    
    # Process each 1-minute candle
    for i in range(20, len(df)):  # Need 20 bars for MA20
        current_price = df.iloc[i]['close']
        timestamp = df.index[i]
        
        # Get subset of data up to current point
        data_slice = df.iloc[:i+1]
        
        # Evaluate signal
        signal_result = strategy.evaluate_signal_v2(
            data_slice, 
            position, 
            timestamp
        )
        
        action = signal_result.get('action', 'NO_TRADE')
        
        if action == 'NO_TRADE_PROXIMITY':
            signals_blocked += 1
        elif action == 'BUY' and position == -1:
            # Buy BTC with USD
            balance_btc = (balance_usd * 0.9975) / current_price  # 0.25% fee
            balance_usd = 0
            position = 1
            entry_price = current_price
            trades.append({
                'time': timestamp,
                'type': 'BUY',
                'price': current_price,
                'btc_amount': balance_btc
            })
        elif action == 'SELL' and position == 1:
            # Sell BTC for USD
            balance_usd = balance_btc * current_price * 0.9975  # 0.25% fee
            balance_btc = 0
            position = -1
            trades.append({
                'time': timestamp,
                'type': 'SELL',
                'price': current_price,
                'usd_amount': balance_usd
            })
    
    # Calculate final value
    final_price = df.iloc[-1]['close']
    if position == 1:
        final_value = balance_btc * final_price
    else:
        final_value = balance_usd
    
    pnl = final_value - initial_usd
    pnl_percent = (pnl / initial_usd) * 100
    
    return {
        'proximity_threshold': proximity_threshold,
        'final_value': final_value,
        'pnl': pnl,
        'pnl_percent': pnl_percent,
        'num_trades': len(trades),
        'signals_blocked': signals_blocked,
        'final_position': 'LONG' if position == 1 else 'SHORT',
        'trades': trades
    }

def main():
    print("Loading data from btcusd.log...")
    
    # Load last 30 days of data for faster processing
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"Loading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    df = parse_log_file('btcusd.log', start_date=start_date, end_date=end_date)
    
    if df.empty:
        print("Error: No data loaded")
        return
    
    # Convert timestamp to datetime index
    df.index = pd.to_datetime(df.index, unit='s')
    
    # Rename price column to close if needed
    if 'price' in df.columns and 'close' not in df.columns:
        df['close'] = df['price']
    
    # Resample to 1-minute bars
    print(f"Resampling {len(df)} trades to 1-minute bars...")
    df_1min = df.resample('1min').agg({
        'close': 'last',
        'amount': 'sum'  # Use amount column for volume
    }).dropna()
    df_1min['volume'] = df_1min['amount']  # Rename for consistency
    
    print(f"Testing on {len(df_1min)} 1-minute candles")
    print(f"Date range: {df_1min.index[0]} to {df_1min.index[-1]}")
    print("")
    
    # Test different proximity thresholds
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    results = []
    
    print("Running backtests...")
    print("-" * 70)
    
    for threshold in thresholds:
        print(f"Testing proximity threshold: {threshold*100:.1f}%")
        result = run_backtest_with_threshold(df_1min, threshold)
        results.append(result)
        
        print(f"  Final P&L: ${result['pnl']:,.2f} ({result['pnl_percent']:.2f}%)")
        print(f"  Trades: {result['num_trades']}, Blocked: {result['signals_blocked']}")
        print("")
    
    # Summary
    print("=" * 70)
    print("SUMMARY - Proximity Threshold Performance")
    print("=" * 70)
    print(f"{'Threshold':<12} {'P&L':<12} {'P&L %':<10} {'Trades':<10} {'Blocked':<10}")
    print("-" * 70)
    
    for r in sorted(results, key=lambda x: x['pnl'], reverse=True):
        print(f"{r['proximity_threshold']*100:>8.1f}%   "
              f"${r['pnl']:>10,.2f}  "
              f"{r['pnl_percent']:>8.2f}%  "
              f"{r['num_trades']:>8}  "
              f"{r['signals_blocked']:>9}")
    
    # Find optimal threshold
    best = max(results, key=lambda x: x['pnl'])
    print("")
    print(f"BEST THRESHOLD: {best['proximity_threshold']*100:.1f}% "
          f"with P&L of ${best['pnl']:,.2f}")

if __name__ == '__main__':
    main()
