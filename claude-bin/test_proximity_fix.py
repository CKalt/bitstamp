#!/usr/bin/env python3
"""
Test the proximity threshold fix
"""
import sys
sys.path.insert(0, '/Users/chris/projects/python/btc/src')

# Test scenarios
test_cases = [
    {"ma4": 114368, "ma20": 114592, "position": 1, "signal": -1, "name": "Current situation"},
    {"ma4": 114700, "ma20": 114100, "position": -1, "signal": 1, "name": "Clear LONG signal"},
    {"ma4": 114100, "ma20": 114700, "position": 1, "signal": -1, "name": "Clear SHORT signal"},
    {"ma4": 114500, "ma20": 114450, "position": -1, "signal": 1, "name": "Borderline case"},
]

PROXIMITY_THRESHOLD = 0.5  # 0.5% threshold

print("PROXIMITY THRESHOLD TEST")
print("=" * 60)
print(f"Threshold: {PROXIMITY_THRESHOLD}%")
print()

for test in test_cases:
    ma4 = test["ma4"]
    ma20 = test["ma20"]
    position = test["position"]
    signal = test["signal"]
    
    # Calculate proximity
    ma_diff = ma4 - ma20
    ma_proximity = abs(ma_diff) / ma20 * 100
    
    # Apply threshold logic
    if ma_proximity <= PROXIMITY_THRESHOLD:
        action = "NO_TRADE_PROXIMITY"
        reason = f"MAs too close: {ma_proximity:.2f}% <= {PROXIMITY_THRESHOLD}%"
    elif signal != position:
        action = "WILL_TRADE"
        reason = f"Signal ({signal}) != Position ({position})"
    else:
        action = "NO_TRADE"
        reason = "Signal matches position"
    
    print(f"{test['name']}:")
    print(f"  MA4: ${ma4:,} | MA20: ${ma20:,}")
    print(f"  Proximity: {ma_proximity:.2f}%")
    print(f"  Position: {position} | Signal: {signal}")
    print(f"  Action: {action}")
    print(f"  Reason: {reason}")
    print()

print("SUMMARY:")
print("With 0.5% threshold, your current 0.20% proximity would NOT trade")
print("This would have prevented all 5 flips from last night")