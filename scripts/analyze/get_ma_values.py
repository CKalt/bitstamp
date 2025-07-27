#!/usr/bin/env python3
"""Get current MA values and movement from server"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.loader import parse_log_file
from datetime import datetime, timedelta
import pandas as pd

# Load recent data
end_date = datetime.now()
start_date = end_date - timedelta(hours=25)  # Get 25 hours for MA20 + history

print("Loading recent price data...")
df = parse_log_file('btcusd.log', start_date, end_date)

# Convert to hourly
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df.set_index('datetime', inplace=True)
hourly = df['price'].resample('1H').last().dropna()

if len(hourly) >= 20:
    # Get the last 20 hourly closes
    recent_prices = hourly.tail(20).values
    
    # Current values
    current_price = recent_prices[-1]
    ma4_current = recent_prices[-4:].mean()
    ma20_current = recent_prices.mean()
    
    # Previous values (1 hour ago)
    ma4_prev = recent_prices[-5:-1].mean()
    ma20_prev = recent_prices[:-1].mean()
    
    # Movement per hour
    ma4_movement = ma4_current - ma4_prev
    ma20_movement = ma20_current - ma20_prev
    
    print(f"\n📊 Current MA Status:")
    print(f"Current Price: ${current_price:,.0f}")
    print(f"MA4:  ${ma4_current:,.0f} (moving ${ma4_movement:+.0f}/hour)")
    print(f"MA20: ${ma20_current:,.0f} (moving ${ma20_movement:+.0f}/hour)")
    print(f"Spread: ${ma4_current - ma20_current:,.0f}")
    
    if ma4_current < ma20_current:
        gap = ma20_current - ma4_current
        print(f"\n🔴 Signal: SHORT (MA4 is ${gap:,.0f} below MA20)")
        
        # Calculate time to flip at current rates
        if ma4_movement > ma20_movement:
            closing_rate = ma4_movement - ma20_movement
            hours_to_flip = gap / closing_rate
            print(f"\n📈 Gap Analysis:")
            print(f"MA4 gaining on MA20 at: ${closing_rate:.0f}/hour")
            print(f"Estimated time to flip: {hours_to_flip:.1f} hours")
        elif ma4_movement < ma20_movement:
            print(f"\n📉 Gap Analysis:")
            print(f"MAs diverging - gap is widening!")
            print(f"MA4 needs price to rise significantly")
        else:
            print(f"\n➡️ Gap Analysis:")
            print(f"MAs moving in parallel - gap unchanged")
    else:
        print(f"\n🟢 Signal: LONG (MA4 > MA20)")
    
    # Show recent price action
    print(f"\n📈 Recent Price Movement:")
    for i in range(5):
        idx = -(i+1)
        print(f"{i}h ago: ${recent_prices[idx]:,.0f}")
else:
    print("Not enough data for MA20")