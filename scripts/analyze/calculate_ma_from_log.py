#!/usr/bin/env python3
"""Calculate MA values directly from btcusd.log"""
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def parse_recent_prices(filename='btcusd.log', hours_back=25):
    """Parse recent hourly prices from log"""
    hourly_prices = {}
    cutoff_time = datetime.now() - timedelta(hours=hours_back)
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    timestamp = int(parts[0])
                    price = float(parts[1])
                    dt = datetime.fromtimestamp(timestamp)
                    
                    if dt >= cutoff_time:
                        # Round to hour
                        hour_key = dt.replace(minute=0, second=0, microsecond=0)
                        hourly_prices[hour_key] = price
                except:
                    continue
    
    # Get last 25 hours in order
    sorted_hours = sorted(hourly_prices.keys(), reverse=True)[:25]
    return [(h, hourly_prices[h]) for h in sorted_hours]

def main():
    print("📊 Calculating MA values from btcusd.log...")
    print("=" * 60)
    
    # Get hourly prices
    hourly_data = parse_recent_prices()
    
    if len(hourly_data) < 20:
        print(f"❌ Not enough data. Got {len(hourly_data)} hourly prices, need at least 20")
        return
    
    # Extract just prices (newest first)
    prices = [p for _, p in hourly_data]
    
    # Current values
    current_price = prices[0]
    ma4 = sum(prices[:4]) / 4
    ma20 = sum(prices[:20]) / 20
    
    # Previous values (1 hour ago)
    ma4_prev = sum(prices[1:5]) / 4
    ma20_prev = sum(prices[1:21]) / 20
    
    # Movement
    ma4_move = ma4 - ma4_prev
    ma20_move = ma20 - ma20_prev
    
    print(f"\n🕐 Current Time: {hourly_data[0][0].strftime('%Y-%m-%d %H:00')}")
    print(f"Current Price: ${current_price:,.0f}")
    
    print(f"\n📈 Moving Averages:")
    print(f"MA4:  ${ma4:,.0f} (moving ${ma4_move:+.1f}/hour)")
    print(f"MA20: ${ma20:,.0f} (moving ${ma20_move:+.1f}/hour)")
    print(f"Spread: ${ma4 - ma20:,.0f}")
    
    # Signal
    if ma4 < ma20:
        gap = ma20 - ma4
        print(f"\n🔴 Signal: SHORT (MA4 is ${gap:,.0f} below MA20)")
        
        # Convergence analysis
        relative_move = ma4_move - ma20_move
        if relative_move > 0:
            hours_to_flip = gap / relative_move
            print(f"\n📊 Convergence Analysis:")
            print(f"  • MA4 gaining at ${relative_move:.1f}/hour")
            print(f"  • Time to flip: ~{hours_to_flip:.1f} hours")
            print(f"  • MA4 needs to reach ${ma20:,.0f}")
        elif relative_move < 0:
            print(f"\n📉 Divergence Analysis:")
            print(f"  • Gap widening by ${-relative_move:.1f}/hour")
            print(f"  • MAs moving apart")
        else:
            print(f"\n➡️ Parallel Movement:")
            print(f"  • Gap stable at ${gap:,.0f}")
    else:
        gap = ma4 - ma20
        print(f"\n🟢 Signal: LONG (MA4 is ${gap:,.0f} above MA20)")
    
    # Recent price action
    print(f"\n📊 Recent Hourly Closes:")
    for i in range(min(5, len(hourly_data))):
        hour, price = hourly_data[i]
        print(f"  {hour.strftime('%H:00')}: ${price:,.0f}")
    
    # What price needed for flip
    if ma4 < ma20:
        # To make MA4 = MA20: (P + p1 + p2 + p3) / 4 = MA20
        price_needed = 4 * ma20 - sum(prices[1:4])
        change_needed = price_needed - current_price
        pct_needed = (change_needed / current_price) * 100
        
        print(f"\n💡 Flip Requirements:")
        print(f"  • Price needed: ${price_needed:,.0f}")
        print(f"  • Change needed: ${change_needed:,.0f} ({pct_needed:+.1f}%)")

if __name__ == "__main__":
    main()