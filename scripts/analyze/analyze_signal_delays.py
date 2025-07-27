#!/usr/bin/env python3
"""
Analyze signal timing delays in the current system.
This will help us understand how much we could improve.
"""

import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def analyze_signal_delays():
    """Analyze how late our signals are compared to ideal timing."""
    
    print("Signal Delay Analysis")
    print("=" * 50)
    
    # Load recent trades to see signal vs execution timing
    try:
        with open('trades.json', 'r') as f:
            trades = json.load(f)
    except:
        print("No trades.json found")
        trades = []
    
    # Analyze each trade
    delays = []
    for trade in trades[-20:]:  # Last 20 trades
        if 'signal_timestamp' in trade and 'timestamp' in trade:
            signal_time = datetime.fromisoformat(trade['signal_timestamp'].replace('T', ' '))
            trade_time = datetime.fromisoformat(trade['timestamp'].replace('T', ' '))
            delay = (trade_time - signal_time).total_seconds()
            
            # Signal timestamps are always on the hour
            # Real crossover could have happened up to 59 minutes earlier
            potential_delay = delay + (59 * 60)  # Maximum possible delay
            
            delays.append({
                'signal_time': signal_time,
                'trade_time': trade_time,
                'execution_delay': delay,
                'max_signal_delay': potential_delay,
                'price': trade.get('price', 0),
                'type': trade.get('type', 'unknown')
            })
    
    if delays:
        df = pd.DataFrame(delays)
        
        print(f"\nAnalyzed {len(delays)} recent trades:")
        print(f"Average execution delay: {df['execution_delay'].mean():.1f} seconds")
        print(f"Max possible signal delay: {df['max_signal_delay'].mean()/60:.1f} minutes")
        
        print("\nRecent trades:")
        for d in delays[-5:]:
            print(f"  {d['signal_time']} -> {d['trade_time']} "
                  f"({d['execution_delay']:.0f}s delay) "
                  f"${d['price']:,.0f} {d['type']}")
    
    # Theoretical analysis
    print("\n" + "="*50)
    print("Theoretical Timing Analysis:")
    print("- Current: Check every 30 seconds, but only see hourly candles")
    print("- Average detection delay: 30 minutes (half of hour)")
    print("- Worst case delay: 59 minutes")
    print("- With 5-min candles: Average delay would be 2.5 minutes")
    print("- Improvement potential: 27.5 minutes faster on average")
    
    # Estimate impact
    print("\nPotential Impact:")
    print("- BTC typical hourly movement: 0.5-1.0%")
    print("- 30-minute delay could mean 0.25-0.5% worse entry")
    print("- On $170k position: $425-850 per trade")
    print("- With ~40 trades/month: $17,000-34,000 potential improvement")

def simulate_faster_signals():
    """Simulate what would happen with faster signal detection."""
    
    print("\n" + "="*50)
    print("Faster Signal Simulation")
    print("="*50)
    
    # Simulate crossover scenarios
    scenarios = [
        {"name": "Quick spike", "crossover_minute": 15, "reversal_minute": 45},
        {"name": "Early cross", "crossover_minute": 5, "reversal_minute": None},
        {"name": "Late cross", "crossover_minute": 55, "reversal_minute": None},
        {"name": "Multiple crosses", "crossover_minute": 20, "reversal_minute": 40},
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"  Crossover at minute: {scenario['crossover_minute']}")
        
        # Current system (hourly)
        hourly_detection = 60 - scenario['crossover_minute']
        
        # 5-minute system
        five_min_detection = 5 - (scenario['crossover_minute'] % 5)
        
        print(f"  Current system detects in: {hourly_detection} minutes")
        print(f"  5-min system detects in: {five_min_detection} minutes")
        print(f"  Improvement: {hourly_detection - five_min_detection} minutes faster")
        
        if scenario['reversal_minute']:
            if scenario['reversal_minute'] < 60:
                print(f"  ⚠️  Reversal at minute {scenario['reversal_minute']} - current system MISSES this!")

def check_current_ma_calculation():
    """Show how MAs are currently calculated."""
    
    print("\n" + "="*50)
    print("Current MA Calculation Method")
    print("="*50)
    
    config = {
        "timeframe": "1H",
        "MA_short": 6,
        "MA_long": 34,
        "check_frequency": "30 seconds",
        "data_source": "hourly candle closes"
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nImplications:")
    print("- MA6 = Average of last 6 hourly closes")
    print("- MA34 = Average of last 34 hourly closes")
    print("- New values only available once per hour")
    print("- Checking every 30 seconds is redundant 99% of the time")
    
    print("\nProposed improvement:")
    print("- Use 5-minute candles instead")
    print("- MA6 on 5-min = 30 minutes of data (same as 0.5 hours)")
    print("- MA34 on 5-min = 170 minutes of data (same as 2.83 hours)")
    print("- Would need to adjust MA periods to maintain similar behavior")

if __name__ == "__main__":
    analyze_signal_delays()
    simulate_faster_signals()
    check_current_ma_calculation()
    
    print("\n" + "="*50)
    print("RECOMMENDATION: Start by implementing parallel 5-minute MA calculation")
    print("Run both systems side-by-side to compare signals before making changes")
    print("="*50)