#!/usr/bin/env python3
"""Quick check of MA values and distance to flip"""
import requests

# Get recent candle data
response = requests.get("http://localhost:4000/api/data/btcusd?limit=20")
if response.ok:
    data = response.json()
    candles = data.get('candles', [])
    
    if len(candles) >= 20:
        # Calculate MAs
        closes = [float(c['close']) for c in candles]
        ma4 = sum(closes[:4]) / 4
        ma20 = sum(closes[:20]) / 20
        current_price = closes[0]
        
        print(f"📊 MA Analysis:")
        print(f"Current Price: ${current_price:,.0f}")
        print(f"MA4:  ${ma4:,.0f}")
        print(f"MA20: ${ma20:,.0f}")
        print(f"Spread: ${ma4 - ma20:,.0f}")
        
        if ma4 < ma20:
            print(f"\n🔴 Signal: SHORT (MA4 < MA20)")
            print(f"Need MA4 to rise above ${ma20:,.0f} to flip LONG")
            gap = ma20 - ma4
            print(f"Gap to close: ${gap:,.0f}")
        else:
            print(f"\n🟢 Signal: LONG (MA4 > MA20)")