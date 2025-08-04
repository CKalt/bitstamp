#!/usr/bin/env python3
"""
MA Strategy Preview - Shows what each MA strategy would do RIGHT NOW
Helps you choose the right strategy before starting the server
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def get_recent_prices(hours=24):
    """Get recent price data from server."""
    print("📊 Fetching recent price data from server...")
    
    try:
        # Get last N hours of data from server
        cmd = f'ssh ck "tail -10000 ~/projects/bitstamp/btcusd.log | grep trade"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if not result.stdout:
            print("❌ Could not fetch price data from server")
            return None
            
        # Parse the log data
        prices = []
        timestamps = []
        
        for line in result.stdout.strip().split('\n'):
            try:
                data = json.loads(line)
                if 'data' in data and 'price' in data['data']:
                    price = float(data['data']['price'])
                    timestamp = float(data['data']['timestamp'])
                    prices.append(price)
                    timestamps.append(datetime.fromtimestamp(timestamp))
            except:
                continue
        
        if not prices:
            print("❌ No valid price data found")
            return None
            
        # Create dataframe
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices
        })
        
        # Filter to recent data
        cutoff_time = datetime.now() - timedelta(hours=hours)
        df = df[df['timestamp'] > cutoff_time]
        
        print(f"✅ Loaded {len(df)} price points from last {hours} hours")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def calculate_ma_signals(df, short_window, long_window):
    """Calculate MA values and signals."""
    # Resample to hourly for MA calculation
    df_hourly = df.set_index('timestamp').resample('1H').agg({
        'price': 'last'
    }).dropna()
    
    # Calculate MAs
    df_hourly[f'MA{short_window}'] = df_hourly['price'].rolling(window=short_window).mean()
    df_hourly[f'MA{long_window}'] = df_hourly['price'].rolling(window=long_window).mean()
    
    # Generate signals
    df_hourly['signal'] = 0
    df_hourly.loc[df_hourly[f'MA{short_window}'] > df_hourly[f'MA{long_window}'], 'signal'] = 1
    df_hourly.loc[df_hourly[f'MA{short_window}'] < df_hourly[f'MA{long_window}'], 'signal'] = -1
    
    return df_hourly

def preview_strategy(short_window, long_window, current_position, current_price):
    """Preview what a specific MA strategy would do."""
    
    print(f"\n🔍 MA {short_window}/{long_window} Strategy Preview")
    print("=" * 50)
    
    # Get price data
    df = get_recent_prices(max(short_window, long_window) + 5)
    if df is None:
        return None
    
    # Calculate signals
    df_ma = calculate_ma_signals(df, short_window, long_window)
    
    # Get latest values
    if len(df_ma) < max(short_window, long_window):
        print(f"❌ Not enough data for MA {short_window}/{long_window}")
        return None
        
    latest = df_ma.iloc[-1]
    ma_short = latest[f'MA{short_window}']
    ma_long = latest[f'MA{long_window}']
    signal = int(latest['signal'])
    
    # Calculate metrics
    ma_diff = ma_short - ma_long
    ma_diff_pct = (ma_diff / ma_long) * 100
    
    print(f"📈 Current Values:")
    print(f"   MA{short_window}: ${ma_short:.2f}")
    print(f"   MA{long_window}: ${ma_long:.2f}")
    print(f"   Difference: ${ma_diff:.2f} ({ma_diff_pct:.2f}%)")
    print(f"   Signal: {signal} ({'LONG' if signal == 1 else 'SHORT' if signal == -1 else 'NEUTRAL'})")
    
    # Determine action
    will_trade = False
    action = "HOLD"
    reason = ""
    
    if signal == 1 and current_position <= 0:
        will_trade = True
        action = "BUY"
        reason = "Signal is LONG but position is SHORT/NEUTRAL"
    elif signal == -1 and current_position >= 0:
        will_trade = True
        action = "SELL"
        reason = "Signal is SHORT but position is LONG/NEUTRAL"
    else:
        reason = f"Signal ({signal}) matches position ({current_position})"
    
    print(f"\n🎯 Action: {action}")
    print(f"   Reason: {reason}")
    
    if will_trade:
        print(f"\n⚠️  WARNING: This strategy would trade IMMEDIATELY upon resume!")
        if action == "BUY":
            print(f"   Would flip from SHORT to LONG at ~${current_price}")
        else:
            print(f"   Would flip from LONG to SHORT at ~${current_price}")
    else:
        print(f"\n✅ SAFE: This strategy matches your current position")
        print(f"   No immediate trade would occur")
    
    # Show recent crossovers
    print(f"\n📊 Recent Crossovers:")
    crossovers = []
    for i in range(max(10, len(df_ma)-10), len(df_ma)):
        if i > 0 and df_ma.iloc[i]['signal'] != df_ma.iloc[i-1]['signal']:
            crossovers.append({
                'time': df_ma.index[i],
                'from': int(df_ma.iloc[i-1]['signal']),
                'to': int(df_ma.iloc[i]['signal']),
                'price': df_ma.iloc[i]['price']
            })
    
    if crossovers:
        for cross in crossovers[-3:]:  # Show last 3
            from_pos = 'LONG' if cross['from'] == 1 else 'SHORT' if cross['from'] == -1 else 'NEUTRAL'
            to_pos = 'LONG' if cross['to'] == 1 else 'SHORT' if cross['to'] == -1 else 'NEUTRAL'
            print(f"   {cross['time'].strftime('%Y-%m-%d %H:%M')}: {from_pos} → {to_pos} at ${cross['price']:.2f}")
    else:
        print(f"   No crossovers in recent history")
    
    return {
        'ma_short': ma_short,
        'ma_long': ma_long,
        'signal': signal,
        'will_trade': will_trade,
        'action': action
    }

def main():
    """Main preview function."""
    
    print("🔮 MA Strategy Preview Tool")
    print("This shows what each strategy would do RIGHT NOW")
    print("=" * 60)
    
    # Get current status
    try:
        import requests
        response = requests.get("http://localhost:4000/api/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            current_price = data.get('last_price', 115000)
            position_data = data.get('position', {})
            current_position = position_data.get('position', 0)
            
            print(f"\n📍 Current Status:")
            print(f"   Price: ${current_price:.2f}")
            print(f"   Position: {current_position} ({'LONG' if current_position > 0 else 'SHORT' if current_position < 0 else 'NEUTRAL'})")
            
            if position_data.get('position_size'):
                print(f"   Size: {abs(position_data['position_size']):.8f} BTC")
                print(f"   Entry: ${position_data.get('entry_price', 0):.2f}")
        else:
            print("⚠️  Could not fetch current status from API")
            print("   Using defaults: Position = 1 (LONG), Price = $115,000")
            current_price = 115000
            current_position = 1
    except Exception as e:
        print(f"⚠️  API Error: {e}")
        print("   Using defaults: Position = 1 (LONG), Price = $115,000")
        current_price = 115000
        current_position = 1
    
    # Preview different strategies
    strategies = [
        (4, 20, "Aggressive - More trades, faster response"),
        (6, 34, "Moderate - Balanced approach"),
        (12, 48, "Conservative - Fewer trades, more stable"),
        (8, 21, "Alternative - Different perspective")
    ]
    
    results = []
    
    for short, long, description in strategies:
        print(f"\n{'='*60}")
        result = preview_strategy(short, long, current_position, current_price)
        if result:
            result['short'] = short
            result['long'] = long
            result['description'] = description
            results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 SUMMARY - Which Strategy To Choose?")
    print(f"{'='*60}")
    
    safe_strategies = [r for r in results if not r['will_trade']]
    trade_strategies = [r for r in results if r['will_trade']]
    
    if safe_strategies:
        print("\n✅ SAFE CHOICES (No immediate trade):")
        for r in safe_strategies:
            print(f"   • MA {r['short']}/{r['long']} - {r['description']}")
            print(f"     Signal: {r['signal']}, matches your position")
    
    if trade_strategies:
        print("\n⚠️  WILL TRADE IMMEDIATELY:")
        for r in trade_strategies:
            print(f"   • MA {r['short']}/{r['long']} - {r['description']}")
            print(f"     Action: {r['action']} (Signal: {r['signal']})")
    
    print("\n💡 RECOMMENDATIONS:")
    print("1. Choose a SAFE strategy if you want to maintain current position")
    print("2. Choose a TRADE strategy only if you want to flip positions")
    print("3. You can wait for market to move before starting any strategy")
    print("4. Always verify server config matches your choice before starting")
    
    print("\n📝 TO APPLY YOUR CHOICE:")
    print("ssh ck")
    print("gg btc  # For live, or gg tst for test")
    print("nano best_strategy.json")
    print("# Set your Short_Window and Long_Window")
    print("# Save and exit")
    print("# Then start server and resume trading")

if __name__ == "__main__":
    main()