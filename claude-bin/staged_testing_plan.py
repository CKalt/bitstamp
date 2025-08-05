#!/usr/bin/env python3
"""
Staged Testing Plan - Start with 5-minute bars for rapid debugging
"""

import json
from datetime import datetime

def create_test_configs():
    """Create test configurations for each stage"""
    
    # Stage 1: 5-minute bars (rapid testing)
    stage1_config = {
        "Short_Window": 4,
        "Long_Window": 20,
        "do_live_trades": False,  # PAPER TRADING ONLY
        "strategy_type": "MA",
        "enable_adaptive_strategy": False,
        "max_trades_per_day": 100,  # Higher limit for 5-min testing
        "proximity_threshold": 0.5,
        "candle_interval": "5min",  # 5-MINUTE BARS
        "_comment": "STAGE 1: 5-minute paper trading for rapid bug detection"
    }
    
    # Stage 2: Hourly bars (production-like)
    stage2_config = {
        "Short_Window": 4,
        "Long_Window": 20,
        "do_live_trades": False,  # STILL PAPER TRADING
        "strategy_type": "MA",
        "enable_adaptive_strategy": False,
        "max_trades_per_day": 10,
        "proximity_threshold": 0.5,
        "candle_interval": "1h",  # HOURLY BARS
        "_comment": "STAGE 2: Hourly paper trading for production validation"
    }
    
    # Save configs
    with open('/tmp/stage1_5min_test.json', 'w') as f:
        json.dump(stage1_config, f, indent=2)
    
    with open('/tmp/stage2_hourly_test.json', 'w') as f:
        json.dump(stage2_config, f, indent=2)
    
    print("📋 STAGED TESTING PLAN")
    print("=" * 60)
    print()
    print("🔬 STAGE 1: 5-Minute Bar Testing (1-2 days)")
    print("Purpose: Rapid bug detection and signal validation")
    print("- Evaluates every 5 minutes instead of hourly")
    print("- See 12x more signals per day")
    print("- Catch bugs quickly")
    print("- Validate proximity threshold logic")
    print("- ALL PAPER TRADES - NO RISK")
    print()
    print("📊 STAGE 2: Hourly Bar Testing (2-3 days)")
    print("Purpose: Production-like validation")
    print("- Same as live system but paper trading")
    print("- Validate daily trade limits")
    print("- Check overnight behavior")
    print("- Confirm proximity threshold effectiveness")
    print("- STILL PAPER TRADES - NO RISK")
    print()
    print("✅ STAGE 3: Live Trading")
    print("Only after both stages pass successfully")
    print()
    print("Files created:")
    print("- /tmp/stage1_5min_test.json")
    print("- /tmp/stage2_hourly_test.json")

if __name__ == "__main__":
    create_test_configs()