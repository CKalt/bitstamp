#!/usr/bin/env python3
"""Get current MA values and movement rates from server API"""
import urllib.request
import json
from datetime import datetime, timedelta

def get_ma_status():
    """Get MA values and calculate movement rates"""
    
    # Get recent hourly candles from server
    url = "http://localhost:4000/api/data/btcusd?interval=1h&limit=25"
    
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())
    except Exception as e:
        print(f"❌ Failed to get data from server: {e}")
        print("\nMake sure:")
        print("1. SSH tunnel is active: ssh -L 4000:localhost:4000 chriskoin")
        print("2. TDR server is running on the remote machine")
        return
    
    candles = data.get('candles', [])
    
    if len(candles) < 20:
        print(f"❌ Not enough data. Got {len(candles)} candles, need at least 20")
        return
    
    # Extract hourly closing prices (newest first)
    hourly_closes = [float(c['close']) for c in candles]
    
    # Current MA values (using most recent data)
    ma4_current = sum(hourly_closes[:4]) / 4
    ma20_current = sum(hourly_closes[:20]) / 20
    current_price = hourly_closes[0]
    
    # Previous MA values (1 hour ago)
    ma4_prev = sum(hourly_closes[1:5]) / 4
    ma20_prev = sum(hourly_closes[1:21]) / 20
    
    # Movement per hour
    ma4_movement = ma4_current - ma4_prev
    ma20_movement = ma20_current - ma20_prev
    
    # Display results
    print(f"\n📊 Current MA Status @ {datetime.now().strftime('%H:%M')}")
    print("=" * 50)
    print(f"Current Price: ${current_price:,.0f}")
    print(f"\nMA4:  ${ma4_current:,.0f} (moving ${ma4_movement:+.1f}/hour)")
    print(f"MA20: ${ma20_current:,.0f} (moving ${ma20_movement:+.1f}/hour)")
    print(f"\nSpread: ${ma4_current - ma20_current:,.0f}")
    
    # Signal analysis
    if ma4_current < ma20_current:
        gap = ma20_current - ma4_current
        print(f"\n🔴 Signal: SHORT (MA4 is ${gap:,.0f} below MA20)")
        
        # Calculate convergence/divergence
        relative_movement = ma4_movement - ma20_movement
        
        if relative_movement > 0:
            # MAs converging
            hours_to_flip = gap / relative_movement
            print(f"\n📈 MAs Converging:")
            print(f"   - MA4 gaining on MA20 at ${relative_movement:.1f}/hour")
            print(f"   - Estimated time to flip: {hours_to_flip:.1f} hours")
            print(f"   - MA4 needs to reach ${ma20_current:,.0f} to flip")
        elif relative_movement < 0:
            # MAs diverging
            print(f"\n📉 MAs Diverging:")
            print(f"   - Gap widening by ${-relative_movement:.1f}/hour")
            print(f"   - Price needs significant upward movement")
        else:
            print(f"\n➡️ MAs Moving in Parallel")
            print(f"   - Gap remains constant at ${gap:,.0f}")
    else:
        gap = ma4_current - ma20_current
        print(f"\n🟢 Signal: LONG (MA4 is ${gap:,.0f} above MA20)")
    
    # Recent price trend
    print(f"\n📈 Recent Price Movement:")
    for i in range(min(5, len(hourly_closes))):
        timestamp = datetime.now() - timedelta(hours=i)
        print(f"   {i}h ago: ${hourly_closes[i]:,.0f} ({timestamp.strftime('%H:00')})")
    
    # Price needed for MA4 to flip
    if ma4_current < ma20_current:
        # Calculate price needed for MA4 to equal MA20
        # MA4_new = (P + closes[1] + closes[2] + closes[3]) / 4 = MA20
        # P = 4 * MA20 - (closes[1] + closes[2] + closes[3])
        price_needed = 4 * ma20_current - sum(hourly_closes[1:4])
        price_change_needed = price_needed - current_price
        
        print(f"\n💡 Flip Analysis:")
        print(f"   - Price needed for immediate flip: ${price_needed:,.0f}")
        print(f"   - Required price increase: ${price_change_needed:,.0f} ({price_change_needed/current_price*100:.1f}%)")

if __name__ == "__main__":
    try:
        get_ma_status()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("1. SSH tunnel is active: ssh -L 4000:localhost:4000 chriskoin")
        print("2. TDR server is running on the remote machine")