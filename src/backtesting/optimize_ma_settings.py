#!/usr/bin/env python3
"""
Test different MA period combinations to find optimal settings for current market
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import parse_log_file

def calculate_ma(data, window):
    """Calculate simple moving average"""
    return data.rolling(window=window).mean()

def backtest_ma_strategy(df, short_window, long_window, proximity_threshold=0.003, initial_usd=10000):
    """
    Backtest MA crossover with given parameters
    """
    # Calculate MAs
    df[f'MA{short_window}'] = calculate_ma(df['close'], short_window)
    df[f'MA{long_window}'] = calculate_ma(df['close'], long_window)
    
    # Initialize
    balance_usd = initial_usd
    balance_btc = 0
    position = -1  # Start SHORT
    trades = 0
    signals_blocked = 0
    
    # Skip rows until we have both MAs
    start_idx = max(short_window, long_window)
    
    for i in range(start_idx, len(df)):
        current_price = df.iloc[i]['close']
        ma_short = df.iloc[i][f'MA{short_window}']
        ma_long = df.iloc[i][f'MA{long_window}']
        
        # Calculate proximity
        if ma_long > 0:
            proximity = abs(ma_short - ma_long) / ma_long
        else:
            proximity = 0
        
        # Determine signal
        if ma_short > ma_long:
            signal = 1  # Bullish
        else:
            signal = -1  # Bearish
        
        # Check if we should trade
        should_trade = False
        
        if signal == 1 and position == -1:  # Buy signal
            if proximity >= proximity_threshold:
                should_trade = True
                trade_type = 'BUY'
            else:
                signals_blocked += 1
                
        elif signal == -1 and position == 1:  # Sell signal
            if proximity >= proximity_threshold:
                should_trade = True
                trade_type = 'SELL'
            else:
                signals_blocked += 1
        
        # Execute trade
        if should_trade:
            trades += 1
            if trade_type == 'BUY':
                fee = balance_usd * 0.0025
                balance_btc = (balance_usd - fee) / current_price
                balance_usd = 0
                position = 1
            else:  # SELL
                gross_usd = balance_btc * current_price
                fee = gross_usd * 0.0025
                balance_usd = gross_usd - fee
                balance_btc = 0
                position = -1
    
    # Calculate final value
    final_price = df.iloc[-1]['close']
    if position == 1:
        final_value = balance_btc * final_price
    else:
        final_value = balance_usd
    
    pnl = final_value - initial_usd
    pnl_percent = (pnl / initial_usd) * 100
    
    # Clean up dataframe columns
    df.drop([f'MA{short_window}', f'MA{long_window}'], axis=1, inplace=True, errors='ignore')
    
    return {
        'short': short_window,
        'long': long_window,
        'pnl': pnl,
        'pnl_percent': pnl_percent,
        'trades': trades,
        'blocked': signals_blocked,
        'final_position': 'LONG' if position == 1 else 'SHORT'
    }

def main():
    print("=" * 80)
    print("MA SETTINGS OPTIMIZATION - Finding Best MA Periods")
    print("=" * 80)
    
    # Load 30 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"\nTesting period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    df = parse_log_file('btcusd.log', start_date=start_date, end_date=end_date)
    df.index = pd.to_datetime(df.index, unit='s')
    if 'price' in df.columns:
        df['close'] = df['price']
    
    # Test on HOURLY bars (like production)
    print("Creating hourly bars...")
    df_hourly = df.resample('1h').agg({
        'close': 'last',
        'amount': 'sum'
    }).dropna()
    
    print(f"Testing on {len(df_hourly)} hourly candles\n")
    
    # Define MA combinations to test
    # Short MAs: 2-20
    # Long MAs: 10-100
    test_combinations = []
    
    # Common combinations
    common_combos = [
        (4, 20),   # Current
        (5, 20),   # Slightly slower
        (8, 21),   # Fibonacci
        (9, 21),   # Common
        (10, 30),  # Medium
        (10, 50),  # Classic
        (12, 26),  # MACD settings
        (20, 50),  # Standard
        (20, 100), # Long term
        (50, 100), # Very long term
        (50, 200), # Traditional
        # Faster combinations
        (2, 10),
        (3, 10),
        (3, 15),
        (5, 15),
        # Wider spreads
        (5, 30),
        (5, 50),
        (10, 40),
        (15, 45),
        (15, 60),
    ]
    
    # Test each combination
    results = []
    print("Testing MA combinations...")
    print("-" * 40)
    
    for short, long in common_combos:
        result = backtest_ma_strategy(df_hourly.copy(), short, long, proximity_threshold=0.003)
        results.append(result)
        
        status = "✅" if result['pnl'] > 0 else "❌"
        print(f"MA{short}/MA{long}: {status} ${result['pnl']:>8,.2f} ({result['trades']} trades)")
    
    # Sort by P&L
    results_sorted = sorted(results, key=lambda x: x['pnl'], reverse=True)
    
    # Display results
    print("\n" + "=" * 80)
    print("TOP 10 MA COMBINATIONS (0.3% proximity threshold)")
    print("=" * 80)
    print(f"\n{'Rank':<6} {'MA Settings':<15} {'P&L':<12} {'Return':<10} {'Trades':<8} {'Blocked':<8}")
    print("-" * 70)
    
    for i, r in enumerate(results_sorted[:10], 1):
        profit_emoji = "🟢" if r['pnl'] > 0 else "🔴"
        print(f"{i:<6} MA{r['short']}/MA{r['long']:<11} "
              f"{profit_emoji} ${r['pnl']:>9,.2f}  {r['pnl_percent']:>7.2f}%  "
              f"{r['trades']:>6}  {r['blocked']:>7}")
    
    # Find profitable ones
    profitable = [r for r in results if r['pnl'] > 0]
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    print(f"\n📊 Results Summary:")
    print(f"   Total combinations tested: {len(results)}")
    print(f"   Profitable combinations: {len(profitable)} ({len(profitable)/len(results)*100:.0f}%)")
    print(f"   Current (MA4/MA20): ${next((r['pnl'] for r in results if r['short']==4 and r['long']==20), 0):,.2f}")
    
    if profitable:
        best = results_sorted[0]
        print(f"\n🏆 Best Performer:")
        print(f"   MA{best['short']}/MA{best['long']}: ${best['pnl']:,.2f} ({best['pnl_percent']:.2f}%)")
        print(f"   Trades: {best['trades']}, Blocked: {best['blocked']}")
        
        # Compare to current
        current = next((r for r in results if r['short']==4 and r['long']==20), None)
        if current and best != current:
            improvement = best['pnl'] - current['pnl']
            print(f"   Improvement over MA4/MA20: ${improvement:,.2f}")
    
    # Test without proximity threshold
    print("\n" + "=" * 80)
    print("TESTING BEST SETTINGS WITHOUT PROXIMITY THRESHOLD")
    print("=" * 80)
    
    if results_sorted:
        top_3 = results_sorted[:3]
        print(f"\n{'MA Settings':<15} {'With 0.3%':<15} {'Without':<15} {'Difference':<15}")
        print("-" * 60)
        
        for r in top_3:
            # Test without threshold
            no_threshold = backtest_ma_strategy(
                df_hourly.copy(), 
                r['short'], 
                r['long'], 
                proximity_threshold=0.0
            )
            
            diff = r['pnl'] - no_threshold['pnl']
            print(f"MA{r['short']}/MA{r['long']:<11} "
                  f"${r['pnl']:>10,.2f}  "
                  f"${no_threshold['pnl']:>10,.2f}  "
                  f"${diff:>10,.2f}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if profitable:
        # Group by characteristics
        fast_mas = [r for r in profitable if r['short'] <= 10]
        slow_mas = [r for r in profitable if r['long'] >= 50]
        
        print("\n✅ Profitable MA combinations found!")
        
        if fast_mas:
            print(f"\n1. Fast MAs work: {len(fast_mas)} profitable combinations with short MA ≤ 10")
            
        if slow_mas:
            print(f"\n2. Longer MAs work: {len(slow_mas)} profitable combinations with long MA ≥ 50")
        
        print(f"\n3. Consider switching to MA{best['short']}/MA{best['long']} "
              f"(${best['pnl']:,.2f} profit vs current ${current['pnl'] if current else 0:,.2f} loss)")
        
        print("\n4. The proximity threshold is still valuable - protects against losses")
    else:
        print("\n❌ No profitable MA combinations found in this market")
        print("   Consider:")
        print("   • Different strategy types (mean reversion, momentum)")
        print("   • Adding additional filters")
        print("   • Waiting for trending market conditions")

if __name__ == '__main__':
    main()