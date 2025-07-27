#!/usr/bin/env python3
"""
Analyze MA status and flip conditions
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.loader import parse_log_file
from datetime import datetime, timedelta
import pandas as pd

print("📊 MA Flip Analysis")
print("=" * 50)

# Load recent data
end_date = datetime.now()
start_date = end_date - timedelta(hours=24)

print("Loading recent price data...")
df = parse_log_file('btcusd.log', start_date, end_date)

# Resample to hourly
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df.set_index('datetime', inplace=True)
hourly = df['price'].resample('1H').last().dropna()

print(f"Loaded {len(hourly)} hourly candles")

# Calculate current MAs
if len(hourly) >= 20:
    # Get last 20 values
    recent_prices = hourly.tail(20).values
    
    # Calculate MAs
    ma4 = recent_prices[-4:].mean()
    ma20 = recent_prices.mean()
    current_price = recent_prices[-1]
    
    print(f"\n📈 Current Status:")
    print(f"   Current Price: ${current_price:,.2f}")
    print(f"   MA4 (fast):    ${ma4:,.2f}")
    print(f"   MA20 (slow):   ${ma20:,.2f}")
    
    spread = ma4 - ma20
    spread_pct = (spread / ma20) * 100
    
    print(f"\n📊 MA Analysis:")
    print(f"   MA Spread: ${spread:,.2f} ({spread_pct:.2f}%)")
    
    if ma4 > ma20:
        print(f"   Signal: LONG ✅ (MA4 > MA20)")
        print(f"\n🎯 Flip Conditions:")
        print(f"   MA4 must cross below MA20 (${ma20:,.2f})")
        print(f"   Current gap: ${spread:,.2f}")
        
        # Calculate how much price needs to move
        # Assuming price affects MA4 more than MA20
        price_move_needed = spread * 4  # Rough approximation
        price_target = current_price - price_move_needed
        move_pct = (price_move_needed / current_price) * 100
        
        print(f"\n💡 Rough Estimate:")
        print(f"   Price needs to drop to ~${price_target:,.2f}")
        print(f"   That's a {move_pct:.1f}% drop from current")
        
        # Check recent trend
        ma4_1h_ago = recent_prices[-5:-1].mean()
        ma4_trend = ma4 - ma4_1h_ago
        
        if ma4_trend < 0:
            print(f"\n📉 MA4 is falling (${ma4_trend:,.2f}/hour)")
            hours_to_flip = abs(spread / ma4_trend) if ma4_trend < 0 else 999
            print(f"   At this rate: ~{hours_to_flip:.1f} hours to flip")
        else:
            print(f"\n📈 MA4 is rising (${ma4_trend:,.2f}/hour)")
            print(f"   Flip is moving further away")
    else:
        print(f"   Signal: SHORT 🔴 (MA4 < MA20)")
        print(f"   MA4 must cross above MA20 to flip LONG")
else:
    print("Not enough data for MA20 calculation")