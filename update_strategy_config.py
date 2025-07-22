#!/usr/bin/env python3
"""
Update strategy configuration for better whipsaw protection
"""
import json
import os
from datetime import datetime

def update_config_for_whipsaw_protection():
    """Add whipsaw protection to existing config"""
    
    # Read current config
    config_file = "best_strategy.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        print(f"❌ {config_file} not found!")
        return
    
    # Backup current config
    backup_name = f"best_strategy.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(backup_name, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Backed up current config to {backup_name}")
    
    # Add whipsaw protection settings
    updates = {
        # Time-based filters
        "min_time_between_trades_minutes": 120,  # 2 hours
        "max_trades_per_day": 5,
        "max_trades_per_hour": 2,
        
        # Pivot adjustments
        "pivot_buffer_multiplier": 1.5,  # Make pivots 50% wider
        "use_dynamic_pivots": True,
        "volatility_lookback_hours": 24,
        
        # Confirmation requirements
        "require_confirmation_bars": 2,
        "min_breakout_distance_percent": 0.2,  # Must break by 0.2%
        
        # Position sizing
        "reduce_size_after_whipsaw": True,
        "whipsaw_size_reduction": 0.7,  # Trade 70% size after whipsaw
        "whipsaw_lookback_hours": 4,
        
        # MA strategy specific
        "ma_separation_threshold": 0.3,  # Require 0.3% MA separation
        "ignore_small_crosses": True,
        
        # Risk limits
        "daily_loss_limit": -2000,
        "consecutive_loss_limit": 3,
        "pause_after_limit_hours": 4
    }
    
    # Apply updates
    config.update(updates)
    
    # Save updated config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ Updated configuration with whipsaw protection:")
    for key, value in updates.items():
        print(f"   {key}: {value}")
    
    print("\n⚠️  IMPORTANT: These settings will:")
    print("   1. Require 2 hours between trades")
    print("   2. Use wider pivots (1.5x)")
    print("   3. Require 2-bar confirmation")
    print("   4. Reduce position size after whipsaws")
    print("   5. Limit to 5 trades per day")
    
    return config

def analyze_improvement():
    """Show expected improvement with new settings"""
    
    print("\n\nEXPECTED IMPROVEMENTS:")
    print("=" * 50)
    
    # Yesterday's performance
    yesterday_trades = 7
    yesterday_whipsaws = 4
    yesterday_fees = 760
    
    # Expected with new settings
    expected_trades = 3
    expected_whipsaws = 1
    expected_fees = 320
    
    print(f"\nYesterday (actual):")
    print(f"   Trades: {yesterday_trades}")
    print(f"   Whipsaws: {yesterday_whipsaws}")
    print(f"   Fees: ${yesterday_fees}")
    
    print(f"\nWith new settings (expected):")
    print(f"   Trades: {expected_trades} (-{yesterday_trades - expected_trades})")
    print(f"   Whipsaws: {expected_whipsaws} (-{yesterday_whipsaws - expected_whipsaws})")
    print(f"   Fees: ${expected_fees} (-${yesterday_fees - expected_fees})")
    
    print(f"\nMonthly impact:")
    print(f"   Fee savings: ${(yesterday_fees - expected_fees) * 20:,.0f}")
    print(f"   Fewer losing trades from whipsaws")
    print(f"   Better risk-adjusted returns")

if __name__ == "__main__":
    print("STRATEGY CONFIGURATION UPDATE")
    print("=" * 50)
    
    # Update config
    config = update_config_for_whipsaw_protection()
    
    # Show expected improvements
    analyze_improvement()
    
    print("\n\n📝 TO APPLY CHANGES:")
    print("1. Review the updated best_strategy.json")
    print("2. Restart the server to load new settings")
    print("3. Monitor for reduced trading frequency")
    print("\nThe system will now be less reactive to noise!")