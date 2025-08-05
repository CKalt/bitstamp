#!/usr/bin/env python3
"""
Test different proximity thresholds to reduce flipping
"""
import json

# Current situation
ma4 = 114368
ma20 = 114592
diff = ma4 - ma20
proximity_pct = abs(diff) / ma20 * 100

print("CURRENT MARKET SITUATION")
print(f"MA4:  ${ma4:,}")
print(f"MA20: ${ma20:,}")
print(f"Diff: ${diff:,} ({proximity_pct:.2f}%)")
print(f"Signal: {'LONG' if ma4 > ma20 else 'SHORT'}")
print()

# Test different thresholds
thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

print("THRESHOLD ANALYSIS")
print("-" * 50)
for threshold in thresholds:
    would_trade = proximity_pct > threshold
    print(f"{threshold:4.2f}% threshold: {'WOULD TRADE' if would_trade else 'NO TRADE (too close)'}")

print()
print("RECOMMENDATION:")
print("Use 0.5% threshold to filter out noise while still catching real moves")
print()

# Show implementation
print("IMPLEMENTATION CODE:")
print("-" * 50)
print("""
# In strategies.py, modify the signal evaluation:

def evaluate_signal(self, df, current_position):
    ma_short = df[f'MA_{self.short_window}'].iloc[-1]
    ma_long = df[f'MA_{self.long_window}'].iloc[-1]
    
    # Calculate proximity
    diff = ma_short - ma_long
    proximity_pct = abs(diff) / ma_long * 100
    
    # Original signal
    if ma_short > ma_long:
        signal = 1  # LONG
    else:
        signal = -1  # SHORT
    
    # Apply proximity threshold
    PROXIMITY_THRESHOLD = 0.5  # Only trade if MAs differ by >0.5%
    
    if proximity_pct <= PROXIMITY_THRESHOLD:
        # Too close to flip - maintain current position
        self.logger.info(f"📍 MAs too close ({proximity_pct:.2f}% < {PROXIMITY_THRESHOLD}%) - holding position")
        return current_position
    
    return signal
""")