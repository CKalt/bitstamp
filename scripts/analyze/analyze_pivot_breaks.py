#!/usr/bin/env python3
"""
Analyze pivot break strategy performance
Based on today's live trading results
"""
import json
from datetime import datetime, timedelta

def analyze_todays_pivot_breaks():
    """Analyze today's pivot break trades"""
    
    # Today's pivot break trades
    pivot_trades = [
        {
            "time": "16:15:15",
            "action": "SELL",
            "reason": "below support $118348",
            "signal_price": 118048,
            "actual_price": 117990,  # Estimated from fills
            "result": "Good - captured profit"
        },
        {
            "time": "17:42:47", 
            "action": "BUY",
            "reason": "above resistance $116744",
            "signal_price": 117506,
            "actual_price": 117506,
            "result": "Whipsaw - sold 1.5 hours later"
        },
        {
            "time": "19:31:44",
            "action": "SELL", 
            "reason": "below support $117050",
            "signal_price": 117044,
            "actual_price": 117044,
            "result": "Whipsaw - bought back 30 min later"
        },
        {
            "time": "20:01:45",
            "action": "BUY",
            "reason": "above resistance $116715", 
            "signal_price": 116729,
            "actual_price": 116759,  # Avg of multi-part
            "result": "Currently in position"
        }
    ]
    
    print("TODAY'S PIVOT BREAK ANALYSIS")
    print("=" * 60)
    
    # Calculate time between trades
    prev_time = None
    rapid_trades = 0
    
    for i, trade in enumerate(pivot_trades):
        print(f"\n{i+1}. {trade['time']} - {trade['action']}")
        print(f"   Reason: {trade['reason']}")
        print(f"   Slippage: ${abs(trade['signal_price'] - trade['actual_price'])}")
        print(f"   Result: {trade['result']}")
        
        # Time analysis
        curr_time = datetime.strptime(f"2025-07-21 {trade['time']}", "%Y-%m-%d %H:%M:%S")
        if prev_time:
            time_diff = (curr_time - prev_time).total_seconds() / 3600
            print(f"   Time since last: {time_diff:.1f} hours")
            if time_diff < 2:
                rapid_trades += 1
        prev_time = curr_time
    
    print(f"\n⚠️  WHIPSAW DETECTION:")
    print(f"   Rapid trades (<2hr): {rapid_trades}")
    print(f"   Position flips: 4 in 4 hours!")
    
    # Calculate costs
    print(f"\nTRADING COSTS:")
    total_slippage = sum(abs(t['signal_price'] - t['actual_price']) for t in pivot_trades)
    avg_slippage = total_slippage / len(pivot_trades)
    print(f"   Total slippage: ${total_slippage:.2f}")
    print(f"   Avg per trade: ${avg_slippage:.2f}")
    
    # Fees (approximate)
    position_value = 117000 * 1.36
    fee_per_trade = position_value * 0.0025
    total_fees = fee_per_trade * len(pivot_trades)
    print(f"   Est. fees: ${total_fees:.2f}")
    print(f"   Total friction: ${total_slippage + total_fees:.2f}")
    
    return {
        'trades': len(pivot_trades),
        'whipsaws': rapid_trades,
        'avg_slippage': avg_slippage,
        'total_costs': total_slippage + total_fees
    }

def suggest_pivot_improvements():
    """Suggest improvements to pivot break strategy"""
    
    print("\n\nPIVOT BREAK STRATEGY IMPROVEMENTS")
    print("=" * 60)
    
    print("\n1. ADD TIME FILTER:")
    print("   if time_since_last_trade < 2 hours:")
    print("       skip_signal = True  # Prevent whipsaws")
    
    print("\n2. ADD VOLATILITY FILTER:")
    print("   atr = calculate_ATR(14)")
    print("   if atr / price > 0.015:  # High volatility")
    print("       pivot_buffer *= 1.5  # Wider pivots")
    
    print("\n3. ADD CONFIRMATION BARS:")
    print("   if pivot_break_detected:")
    print("       wait_for_n_bars = 2  # Confirm break")
    print("       if still_beyond_pivot:")
    print("           execute_trade()")
    
    print("\n4. DYNAMIC PIVOT CALCULATION:")
    print("   Instead of fixed pivots:")
    print("   - Use Volume Weighted Average Price (VWAP)")
    print("   - Adjust for time of day volatility")
    print("   - Consider order book imbalance")
    
    print("\n5. POSITION SIZING BASED ON CONFIDENCE:")
    print("   if consecutive_whipsaws > 1:")
    print("       position_size = base_size * 0.5")
    print("   if strong_trend:")
    print("       position_size = base_size * 1.2")

def create_enhanced_backtest_config():
    """Create configuration for realistic backtesting"""
    
    config = {
        "realistic_fills": {
            "enabled": True,
            "base_slippage": 0.0005,
            "multi_part_slippage": 0.0002,
            "max_order_size": 0.9
        },
        "whipsaw_filter": {
            "enabled": True,
            "min_hours_between_trades": 2,
            "max_daily_trades": 5
        },
        "adaptive_pivots": {
            "enabled": True,
            "base_buffer": 0.002,  # 0.2%
            "volatility_multiplier": True,
            "confirmation_bars": 2
        },
        "fees": {
            "maker": 0.0025,
            "taker": 0.0025,
            "multi_part_multiplier": 2.5
        },
        "risk_management": {
            "max_consecutive_losses": 3,
            "reduce_size_after_losses": True,
            "daily_loss_limit": -1000
        }
    }
    
    with open('realistic_backtest_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n\nSAVED: realistic_backtest_config.json")
    print("Use this configuration for accurate backtesting!")
    
    return config

if __name__ == "__main__":
    # Analyze today's trading
    stats = analyze_todays_pivot_breaks()
    
    # Suggest improvements
    suggest_pivot_improvements()
    
    # Create config
    config = create_enhanced_backtest_config()
    
    print("\n\nKEY TAKEAWAY:")
    print("=" * 60)
    print("Your live trading shows the pivot break strategy is too sensitive.")
    print("4 trades in 4 hours with high whipsaw rate means:")
    print("1. Pivots are too tight for current volatility")
    print("2. Need time-based filters to prevent overtrading")
    print("3. Multi-part orders add significant friction")
    print("\nBacktest with these realistic parameters to match live results!")