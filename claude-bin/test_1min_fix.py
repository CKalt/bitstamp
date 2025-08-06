#!/usr/bin/env python3
"""
Test that 1-minute candle fix works
"""
from datetime import datetime
import time

print("🧪 TESTING 1-MINUTE CANDLE FIX")
print("=" * 40)

# Simulate the fixed logic
_last_candle_check = None

for i in range(150):  # Test for 2.5 minutes
    current_time = datetime.now()
    current_candle = current_time.replace(second=0, microsecond=0)
    
    if _last_candle_check is None:
        _last_candle_check = current_candle
        print(f"Initial candle: {current_candle}")
    
    should_evaluate = current_candle > _last_candle_check
    
    if should_evaluate:
        print(f"✅ NEW CANDLE at {current_candle} (was {_last_candle_check})")
        _last_candle_check = current_candle
    else:
        if i % 10 == 0:  # Print every 10 seconds
            seconds_left = 60 - current_time.second
            print(f"   Waiting... {seconds_left}s until next candle")
    
    time.sleep(1)

print("\nTest complete! Should have seen 2-3 candle transitions.")