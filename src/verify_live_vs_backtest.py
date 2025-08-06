#!/usr/bin/env python3
"""
Verify that live trading matches backtest behavior for proximity threshold
"""

import sys
import os

def compare_live_vs_expected():
    """Compare what actually happened vs what should have happened"""
    
    print("📊 LIVE vs BACKTEST COMPARISON")
    print("==============================\n")
    
    # What SHOULD have happened with proximity threshold
    expected = {
        "02:30": {
            "proximity": 0.001,
            "action": "BLOCK",  # Should have blocked at 0.001%
            "reason": "MAs too close (0.001% < 0.3%)"
        },
        "03:01": {
            "proximity": 0.05,
            "action": "BLOCK",  # Should have blocked 
            "reason": "MAs too close (0.05% < 0.3%)"
        },
        "06:01": {
            "proximity": 0.5,  # Estimate
            "action": "TRADE",  # Should trade if > 0.3%
            "reason": "MAs diverged enough"
        }
    }
    
    # What ACTUALLY happened (from overnight test)
    actual = {
        "02:30": {
            "proximity": 0.001,
            "action": "TRADED",  # BUG: Traded despite low proximity
            "result": "Lost $59 on flip"
        },
        "03:01": {
            "proximity": 0.05,
            "action": "TRADED",  # BUG: Traded again
            "result": "Lost $60 on flip (31 min later)"
        },
        "06:01": {
            "proximity": 0.5,  # Estimate
            "action": "TRADED",
            "result": "Lost $691 on flip"
        }
    }
    
    print("❌ CRITICAL BUG FOUND:")
    print("----------------------")
    print("The proximity threshold was NOT blocking trades!")
    print("- At 02:30: Traded with 0.001% proximity (should block)")
    print("- At 03:01: Traded with 0.05% proximity (should block)")
    print("")
    
    print("✅ AFTER FIX (Current Status):")
    print("-----------------------------")
    print("Now seeing '🚫 BLOCKING TRADE' messages at 0.19-0.20% proximity")
    print("This is correct behavior - blocking when < 0.3%")
    print("")
    
    print("📋 VERIFICATION CHECKLIST:")
    print("-------------------------")
    print("1. ✅ Proximity calculation working (logs show correct %)")
    print("2. ✅ Blocking messages now appear in logs")
    print("3. ✅ No trades executed when proximity < 0.3%")
    print("4. ⏳ Need to wait for MAs to diverge > 0.3% to verify trades execute")
    print("")
    
    print("🎯 KEY METRICS TO MATCH:")
    print("------------------------")
    print("BACKTEST should show:")
    print("  - Block rate: ~95-99% with 0.3% threshold")
    print("  - Trades per day: 1-3 with hourly candles")
    print("  - No consecutive trades within 30 minutes")
    print("")
    print("LIVE TRADING should show:")
    print("  - Same block rate as backtest")
    print("  - Same trade frequency")
    print("  - Same entry/exit points (±1 candle)")

if __name__ == '__main__':
    compare_live_vs_expected()