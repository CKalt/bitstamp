#!/usr/bin/env python3
"""
Simple MA analysis without external dependencies.
Parses btcusd.log directly to calculate MA values.
"""

import json
from datetime import datetime, timedelta
from collections import defaultdict

def parse_btcusd_log(hours_back=48):
    """Parse recent price data from btcusd.log"""
    hourly_prices = defaultdict(list)
    cutoff_time = datetime.now() - timedelta(hours=hours_back)
    
    try:
        with open('btcusd.log', 'r') as f:
            for line in f:
                try:
                    # Parse JSON trade data
                    trade = json.loads(line.strip())
                    if 'data' in trade and 'timestamp' in trade['data'] and trade.get('event') == 'trade':
                        timestamp = int(trade['data']['timestamp'])
                        price = float(trade['data']['price'])
                        # Skip if timestamp is way in the future (test data)
                        if timestamp > 2000000000:  # Year 2033+
                            continue
                        dt = datetime.fromtimestamp(timestamp)
                        
                        # Group by hour regardless of cutoff for debugging
                        hour_key = dt.replace(minute=0, second=0, microsecond=0)
                        hourly_prices[hour_key].append(price)
                except:
                    continue
    except FileNotFoundError:
        print("❌ btcusd.log not found")
        return []
    
    # Get hourly closing prices (last price of each hour)
    hourly_closes = []
    for hour in sorted(hourly_prices.keys(), reverse=True):
        if hourly_prices[hour]:
            hourly_closes.append((hour, hourly_prices[hour][-1]))  # Last price = close
    
    # Debug info
    if hourly_closes:
        newest = hourly_closes[0][0]
        oldest = hourly_closes[-1][0] if len(hourly_closes) > 1 else newest
        print(f"Found data from {oldest} to {newest}")
        print(f"Total hours with data: {len(hourly_closes)}")
    
    return hourly_closes[:25]  # Return newest 25 hours

def calculate_ma(prices, period):
    """Calculate simple moving average"""
    if len(prices) < period:
        return None
    return sum(prices[:period]) / period

def main():
    print("📊 Simple MA Analysis")
    print("=" * 60)
    
    # Get hourly prices
    hourly_data = parse_btcusd_log()
    
    if len(hourly_data) < 20:
        print(f"❌ Not enough data. Got {len(hourly_data)} hours, need at least 20")
        return
    
    # Extract just prices (newest first)
    prices = [price for _, price in hourly_data]
    
    # Calculate current MAs
    ma4 = calculate_ma(prices, 4)
    ma20 = calculate_ma(prices, 20)
    
    # Calculate previous MAs (1 hour ago)
    ma4_prev = calculate_ma(prices[1:], 4)
    ma20_prev = calculate_ma(prices[1:], 20)
    
    if not all([ma4, ma20, ma4_prev, ma20_prev]):
        print("❌ Could not calculate all MA values")
        return
    
    # Current values
    current_time = hourly_data[0][0]
    current_price = prices[0]
    
    # Movement rates
    ma4_movement = ma4 - ma4_prev
    ma20_movement = ma20 - ma20_prev
    
    print(f"\n🕐 Current Hour: {current_time.strftime('%Y-%m-%d %H:00')}")
    print(f"Current Price: ${current_price:,.0f}")
    
    print(f"\n📈 Moving Averages:")
    print(f"MA4:  ${ma4:,.0f} (moving ${ma4_movement:+.1f}/hour)")
    print(f"MA20: ${ma20:,.0f} (moving ${ma20_movement:+.1f}/hour)")
    print(f"Spread: ${ma4 - ma20:,.0f}")
    
    # Signal analysis
    if ma4 < ma20:
        gap = ma20 - ma4
        print(f"\n🔴 MA Signal: SHORT (MA4 is ${gap:,.0f} below MA20)")
        
        # Convergence analysis
        relative_movement = ma4_movement - ma20_movement
        
        if relative_movement > 0:
            hours_to_flip = gap / relative_movement
            print(f"\n📊 Convergence Analysis:")
            print(f"  • MA4 gaining on MA20 at ${relative_movement:.1f}/hour")
            print(f"  • Estimated time to flip: {hours_to_flip:.1f} hours")
            print(f"  • Flip will occur when MA4 reaches ${ma20:,.0f}")
        elif relative_movement < 0:
            print(f"\n📉 Divergence Analysis:")
            print(f"  • Gap widening by ${-relative_movement:.1f}/hour")
            print(f"  • MAs moving apart")
        else:
            print(f"\n➡️ Parallel Movement:")
            print(f"  • Gap stable at ${gap:,.0f}")
    else:
        gap = ma4 - ma20
        print(f"\n🟢 MA Signal: LONG (MA4 is ${gap:,.0f} above MA20)")
    
    # Recent price trend
    print(f"\n📊 Recent Hourly Prices:")
    for i in range(min(5, len(hourly_data))):
        hour, price = hourly_data[i]
        print(f"  {hour.strftime('%H:00')}: ${price:,.0f}")
    
    # What price needed for immediate flip
    if ma4 < ma20:
        # To make MA4 = MA20 immediately: (P + p1 + p2 + p3) / 4 = MA20
        price_needed = 4 * ma20 - sum(prices[1:4])
        change_needed = price_needed - current_price
        pct_needed = (change_needed / current_price) * 100
        
        print(f"\n💡 Immediate Flip Requirements:")
        print(f"  • Price must jump to: ${price_needed:,.0f}")
        print(f"  • Change needed: ${change_needed:,.0f} ({pct_needed:+.1f}%)")
    
    print(f"\n📍 Current Trading Status:")
    print(f"  • Position: SHORT at $117,564")
    print(f"  • Current Price: ${current_price:,.0f}")
    print(f"  • Unrealized Loss: ~${(current_price - 117564) * 1.45:.0f}")
    print(f"  • Strategy: AdaptiveMultiStrategy in RANGING mode")
    print(f"  • Will flip to LONG when MA4 > MA20 AND in TRENDING mode")

if __name__ == "__main__":
    main()