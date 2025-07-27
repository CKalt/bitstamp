#!/usr/bin/env python3
"""
Check current MA status and predict next flip
"""
import requests
import json
from datetime import datetime

SERVER_URL = "http://localhost:4000"

print("📊 Current MA Status Check")
print("=" * 50)

# Get current status
response = requests.post(
    f"{SERVER_URL}/api/command",
    json={"command": "strategy_diagnostics"},
    headers={'Content-Type': 'application/json'}
)

if response.ok:
    output = response.json().get('output', '')
    
    # Extract key info from output
    lines = output.split('\n')
    for line in lines:
        if 'Current Price:' in line:
            print(line.strip())
        elif 'Entry:' in line and 'Current:' in line:
            print(line.strip())
        elif 'P&L:' in line:
            print(line.strip())
        elif 'MA Crossover:' in line:
            print(f"\n🔍 {line.strip()}")

# Get detailed MA data
print("\n📈 Requesting detailed MA analysis...")
response = requests.post(
    f"{SERVER_URL}/api/command",
    json={"command": "candles btcusd 1H 24"},
    headers={'Content-Type': 'application/json'}
)

if response.ok:
    output = response.json().get('output', '')
    
    # Parse the last few candles to see MA trend
    lines = output.split('\n')
    
    # Look for current MA values
    print("\n📊 Recent 1H Candle Data:")
    candle_count = 0
    for i, line in enumerate(lines):
        if 'Close:' in line and candle_count < 5:
            print(line.strip())
            candle_count += 1

# Calculate MA values manually
print("\n🧮 Calculating current MA values...")
response = requests.get(f"{SERVER_URL}/api/data/btcusd?limit=20")
if response.ok:
    data = response.json()
    candles = data.get('candles', [])
    
    if len(candles) >= 20:
        # Get close prices
        closes = [float(c['close']) for c in candles]
        
        # Calculate MAs
        ma4 = sum(closes[:4]) / 4
        ma20 = sum(closes[:20]) / 20
        
        current_price = closes[0]
        
        print(f"\n📊 Moving Average Analysis:")
        print(f"   Current Price: ${current_price:,.2f}")
        print(f"   MA4 (fast):    ${ma4:,.2f}")
        print(f"   MA20 (slow):   ${ma20:,.2f}")
        print(f"   Difference:    ${ma4 - ma20:,.2f}")
        
        if ma4 > ma20:
            print(f"\n✅ Signal: LONG (MA4 > MA20)")
            flip_distance = ma4 - ma20
            flip_pct = (flip_distance / ma20) * 100
            print(f"   Distance to flip: ${flip_distance:,.2f} ({flip_pct:.2f}%)")
            print(f"   MA4 needs to drop below ${ma20:,.2f} to trigger SELL")
            
            # Estimate when flip might occur
            ma4_change_rate = (closes[0] - closes[3]) / 4  # Average change per hour
            if ma4_change_rate < 0:  # MA4 is falling
                hours_to_flip = flip_distance / abs(ma4_change_rate)
                print(f"\n⏰ At current rate, flip in ~{hours_to_flip:.1f} hours")
            else:
                print(f"\n⏰ MA4 is rising, flip not imminent")
        else:
            print(f"\n🔴 Signal: SHORT (MA4 < MA20)")
            flip_distance = ma20 - ma4
            flip_pct = (flip_distance / ma20) * 100
            print(f"   Distance to flip: ${flip_distance:,.2f} ({flip_pct:.2f}%)")
            print(f"   MA4 needs to rise above ${ma20:,.2f} to trigger BUY")