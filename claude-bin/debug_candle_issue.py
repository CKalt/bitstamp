#!/usr/bin/env python3
"""
Debug why 1-minute candles aren't triggering
"""
from datetime import datetime

# Simulate the candle logic
print("🔍 DEBUGGING 1-MINUTE CANDLE LOGIC")
print("=" * 40)

# Initial setup (what the code does)
signal_time = datetime(2025, 8, 6, 0, 0, 0)  # Midnight
current_candle = signal_time.replace(second=0, microsecond=0)
_last_candle_check = current_candle

print(f"Initial setup:")
print(f"  signal_time: {signal_time}")
print(f"  current_candle: {current_candle}")
print(f"  _last_candle_check: {_last_candle_check}")
print()

# Problem: signal_time is stuck at midnight
# But should_evaluate checks if current > last
# Since both are midnight, it's never > so never evaluates!

print("Problem identified:")
print("  should_evaluate = current_candle > _last_candle_check")
print(f"  {current_candle} > {_last_candle_check} = {current_candle > _last_candle_check}")
print()
print("❌ BUG: Signal time is stuck at midnight (00:00:00)")
print("   This prevents minute transitions from being detected")
print()
print("The system needs real-time data to update signal_time")
print("Or it needs to use datetime.now() for candle checks")