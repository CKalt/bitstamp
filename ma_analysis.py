#!/usr/bin/env python3
"""
Analyze MA values and movement for TDR trading system.

Key findings:
1. System is running AdaptiveMultiStrategy (not pure MA) due to hardcoded instantiation in shell.py:474
2. AdaptiveMultiStrategy switches between TRENDING (MA), RANGING (RSI), and VOLATILE (MACD) strategies
3. MA values are calculated in generate_trending_signal() method using add_moving_averages()
"""

import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.loader import parse_log_file
from indicators.technical_indicators import add_moving_averages, ensure_datetime_index
import pandas as pd

def get_ma_analysis():
    """Calculate MA values and movement from btcusd.log"""
    
    print("📊 MA Analysis for TDR Trading System")
    print("=" * 60)
    
    # Load recent data (30 hours for good MA20 calculation)
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=30)
    
    print(f"Loading data from {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
    
    # Parse log file
    df = parse_log_file('btcusd.log', start_date, end_date)
    
    if df.empty:
        print("❌ No data found in btcusd.log")
        return
    
    print(f"✅ Loaded {len(df)} data points")
    
    # Convert to hourly candles (matching what the strategy uses)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = ensure_datetime_index(df)
    
    # Resample to hourly (matching strategy logic)
    df_hourly = df.resample('1H').agg({
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    print(f"📊 Converted to {len(df_hourly)} hourly candles")
    
    if len(df_hourly) < 20:
        print(f"❌ Need at least 20 hourly candles, only have {len(df_hourly)}")
        return
    
    # Calculate MAs using same method as strategy
    df_ma = add_moving_averages(df_hourly.copy(), 4, 20, price_col='close')
    
    # Get current values
    current_data = df_ma.iloc[-1]
    prev_data = df_ma.iloc[-2]
    
    current_price = current_data['close']
    ma4_current = current_data['Short_MA']
    ma20_current = current_data['Long_MA']
    
    ma4_prev = prev_data['Short_MA']
    ma20_prev = prev_data['Long_MA']
    
    # Movement rates
    ma4_movement = ma4_current - ma4_prev
    ma20_movement = ma20_current - ma20_prev
    
    print(f"\n🕐 Analysis Time: {current_data.name.strftime('%Y-%m-%d %H:00')}")
    print(f"Current Price: ${current_price:,.0f}")
    
    print(f"\n📈 Moving Averages:")
    print(f"MA4:  ${ma4_current:,.0f} (moving ${ma4_movement:+.1f}/hour)")
    print(f"MA20: ${ma20_current:,.0f} (moving ${ma20_movement:+.1f}/hour)")
    print(f"Spread: ${ma4_current - ma20_current:,.0f}")
    
    # Signal analysis
    if ma4_current < ma20_current:
        gap = ma20_current - ma4_current
        print(f"\n🔴 MA Signal: SHORT (MA4 is ${gap:,.0f} below MA20)")
        
        # Convergence analysis
        relative_movement = ma4_movement - ma20_movement
        
        if relative_movement > 0:
            hours_to_flip = gap / relative_movement
            print(f"\n📊 Convergence Analysis:")
            print(f"  • MA4 gaining on MA20 at ${relative_movement:.1f}/hour")
            print(f"  • Estimated time to flip: {hours_to_flip:.1f} hours")
            print(f"  • MA4 needs to reach ${ma20_current:,.0f} to generate LONG signal")
        elif relative_movement < 0:
            print(f"\n📉 Divergence Analysis:")
            print(f"  • Gap widening by ${-relative_movement:.1f}/hour")
            print(f"  • MAs moving apart - no flip in sight")
        else:
            print(f"\n➡️ Parallel Movement:")
            print(f"  • Gap stable at ${gap:,.0f}")
            print(f"  • No convergence or divergence")
    else:
        gap = ma4_current - ma20_current
        print(f"\n🟢 MA Signal: LONG (MA4 is ${gap:,.0f} above MA20)")
    
    # Show recent MA trend
    print(f"\n📊 Recent MA Values (newest first):")
    for i in range(min(5, len(df_ma))):
        idx = -(i+1)
        row = df_ma.iloc[idx]
        print(f"  {i}h ago: MA4=${row['Short_MA']:,.0f}, MA20=${row['Long_MA']:,.0f}, Price=${row['close']:,.0f}")
    
    # Price needed for immediate flip
    if ma4_current < ma20_current:
        # To make MA4 = MA20: (P + p1 + p2 + p3) / 4 = MA20
        recent_prices = [df_hourly.iloc[-(i+1)]['close'] for i in range(1, 4)]
        price_needed = 4 * ma20_current - sum(recent_prices)
        change_needed = price_needed - current_price
        pct_needed = (change_needed / current_price) * 100
        
        print(f"\n💡 Immediate Flip Requirements:")
        print(f"  • Price must reach: ${price_needed:,.0f}")
        print(f"  • Change needed: ${change_needed:,.0f} ({pct_needed:+.1f}%)")
        print(f"  • This would instantly make MA4 = MA20")
    
    print(f"\n⚠️  Important Notes:")
    print(f"  1. System is using AdaptiveMultiStrategy (not pure MA)")
    print(f"  2. Currently in RANGING mode (confidence 50.0%)")
    print(f"  3. Will only flip position when TRENDING strategy is active AND MA signal changes")
    print(f"  4. Your position: SHORT at $117,564 (loss: -$1,056)")

if __name__ == "__main__":
    get_ma_analysis()