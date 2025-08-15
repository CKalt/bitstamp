#!/usr/bin/env python3
"""
Simple backtest for proximity threshold - tests the core logic without full system dependencies
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import parse_log_file

def calculate_ma(data, window):
    """Calculate simple moving average"""
    return data.rolling(window=window).mean()

def run_backtest_with_threshold(df, proximity_threshold=0.3, initial_usd=10000):
    """
    Run backtest with specific proximity threshold
    Uses MA4 and MA20 crossover strategy
    """
    # Calculate moving averages
    df['MA4'] = calculate_ma(df['close'], 4)
    df['MA20'] = calculate_ma(df['close'], 20)
    
    # Initialize
    balance_usd = initial_usd
    balance_btc = 0
    position = -1  # Start SHORT (holding USD)
    trades = []
    signals_blocked = 0
    
    # Track for analysis
    proximity_values = []
    
    # Skip first 20 rows (need MA20)
    for i in range(20, len(df)):
        current_price = df.iloc[i]['close']
        ma4 = df.iloc[i]['MA4']
        ma20 = df.iloc[i]['MA20']
        timestamp = df.index[i]
        
        # Calculate proximity
        if ma20 > 0:
            proximity = abs(ma4 - ma20) / ma20
        else:
            proximity = 0
            
        proximity_values.append(proximity)
        
        # Determine signal (1 = BUY signal, -1 = SELL signal)
        if ma4 > ma20:
            signal = 1  # MA4 above MA20 - bullish
        else:
            signal = -1  # MA4 below MA20 - bearish
            
        # Check if we should trade
        should_trade = False
        trade_type = None
        
        if signal == 1 and position == -1:  # Buy signal and we're SHORT
            if proximity >= proximity_threshold:
                should_trade = True
                trade_type = 'BUY'
            else:
                signals_blocked += 1
                
        elif signal == -1 and position == 1:  # Sell signal and we're LONG  
            if proximity >= proximity_threshold:
                should_trade = True
                trade_type = 'SELL'
            else:
                signals_blocked += 1
        
        # Execute trade if conditions met
        if should_trade:
            if trade_type == 'BUY':
                # Buy BTC with USD
                fee = balance_usd * 0.0025  # 0.25% fee
                balance_btc = (balance_usd - fee) / current_price
                balance_usd = 0
                position = 1
                trades.append({
                    'time': timestamp,
                    'type': 'BUY',
                    'price': current_price,
                    'btc_amount': balance_btc,
                    'ma4': ma4,
                    'ma20': ma20,
                    'proximity': proximity
                })
            elif trade_type == 'SELL':
                # Sell BTC for USD
                gross_usd = balance_btc * current_price
                fee = gross_usd * 0.0025  # 0.25% fee
                balance_usd = gross_usd - fee
                balance_btc = 0
                position = -1
                trades.append({
                    'time': timestamp,
                    'type': 'SELL',
                    'price': current_price,
                    'usd_amount': balance_usd,
                    'ma4': ma4,
                    'ma20': ma20,
                    'proximity': proximity
                })
    
    # Calculate final value
    final_price = df.iloc[-1]['close']
    if position == 1:
        final_value = balance_btc * final_price
    else:
        final_value = balance_usd
    
    pnl = final_value - initial_usd
    pnl_percent = (pnl / initial_usd) * 100
    
    # Calculate average proximity when trades happened vs when blocked
    avg_prox = np.mean(proximity_values) if proximity_values else 0
    
    return {
        'proximity_threshold': proximity_threshold,
        'final_value': final_value,
        'pnl': pnl,
        'pnl_percent': pnl_percent,
        'num_trades': len(trades),
        'signals_blocked': signals_blocked,
        'final_position': 'LONG' if position == 1 else 'SHORT',
        'avg_proximity': avg_prox,
        'trades': trades
    }

def main():
    print("Simple Proximity Threshold Backtest")
    print("=" * 50)
    
    # Load last 7 days for faster testing
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"Loading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    df = parse_log_file('btcusd.log', start_date=start_date, end_date=end_date)
    
    if df.empty:
        print("Error: No data loaded")
        return
    
    # Convert to proper format
    df.index = pd.to_datetime(df.index, unit='s')
    if 'price' in df.columns:
        df['close'] = df['price']
    
    # Resample to 1-minute bars
    print(f"Resampling {len(df)} trades to 1-minute bars...")
    df_1min = df.resample('1min').agg({
        'close': 'last',
        'amount': 'sum'
    }).dropna()
    
    print(f"Testing on {len(df_1min)} 1-minute candles")
    print(f"Date range: {df_1min.index[0]} to {df_1min.index[-1]}")
    print("")
    
    # Test different proximity thresholds
    thresholds = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01]
    results = []
    
    print("Running backtests...")
    print("-" * 50)
    
    for threshold in thresholds:
        print(f"Testing proximity threshold: {threshold*100:.1f}%", end=" ")
        result = run_backtest_with_threshold(df_1min, threshold)
        results.append(result)
        print(f"→ P&L: ${result['pnl']:,.2f} ({result['num_trades']} trades)")
    
    # Summary table
    print("\n" + "=" * 70)
    print("PROXIMITY THRESHOLD BACKTEST RESULTS")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Final Value':<12} {'P&L':<12} {'P&L %':<10} {'Trades':<8} {'Blocked':<8}")
    print("-" * 70)
    
    for r in sorted(results, key=lambda x: x['pnl'], reverse=True):
        print(f"{r['proximity_threshold']*100:>8.1f}%   "
              f"${r['final_value']:>10,.2f}  "
              f"${r['pnl']:>10,.2f}  "
              f"{r['pnl_percent']:>8.2f}%  "
              f"{r['num_trades']:>6}  "
              f"{r['signals_blocked']:>7}")
    
    # Find optimal threshold
    best = max(results, key=lambda x: x['pnl'])
    worst = min(results, key=lambda x: x['pnl'])
    
    print("\n" + "=" * 70)
    print(f"BEST:  {best['proximity_threshold']*100:.1f}% threshold → "
          f"${best['pnl']:,.2f} P&L ({best['num_trades']} trades)")
    print(f"WORST: {worst['proximity_threshold']*100:.1f}% threshold → "
          f"${worst['pnl']:,.2f} P&L ({worst['num_trades']} trades)")
    
    # Show impact
    no_threshold = next((r for r in results if r['proximity_threshold'] == 0), None)
    current_threshold = next((r for r in results if r['proximity_threshold'] == 0.003), None)
    
    if no_threshold and current_threshold:
        improvement = current_threshold['pnl'] - no_threshold['pnl']
        trade_reduction = no_threshold['num_trades'] - current_threshold['num_trades']
        print(f"\nIMPACT OF 0.3% THRESHOLD:")
        print(f"  P&L improvement: ${improvement:,.2f}")
        print(f"  Trade reduction: {trade_reduction} fewer trades")
        print(f"  Signals blocked: {current_threshold['signals_blocked']}")

if __name__ == '__main__':
    main()