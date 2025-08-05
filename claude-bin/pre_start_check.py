#!/usr/bin/env python3
"""Pre-start safety check - runs on server before starting"""

import json
import sys

print("🛡️  PRE-START SAFETY CHECK")
print("=" * 40)

# Check current config
try:
    with open('best_strategy.json', 'r') as f:
        config = json.load(f)
    print('📋 Server Config:')
    print(f'   MA Strategy: {config["Short_Window"]}/{config["Long_Window"]}')
    print(f'   Live Trading: {config["do_live_trades"]}')
    print()
except Exception as e:
    print(f"❌ Could not read config: {e}")
    sys.exit(1)

# Calculate current signal
lines = []
try:
    with open('btcusd.log', 'rb') as f:
        f.seek(-2000000, 2)  # Last 2MB
        for line in f:
            try:
                data = json.loads(line)
                if 'data' in data and 'price' in data['data']:
                    lines.append(float(data['data']['price']))
            except:
                pass
except Exception as e:
    print(f"❌ Could not read price log: {e}")
    sys.exit(1)

short_win = config['Short_Window']
long_win = config['Long_Window']

# Need at least long_window * 120 prices (roughly 120 per hour)
needed = long_win * 120
if len(lines) < needed:
    print(f"❌ Not enough data: have {len(lines)}, need {needed}")
    sys.exit(1)

# Calculate MAs
ma_short = sum(lines[-short_win*120:]) / (short_win*120)
ma_long = sum(lines[-long_win*120:]) / (long_win*120)
signal = 1 if ma_short > ma_long else -1
current_price = lines[-1]

print(f'📊 Current Market:')
print(f'   Price: ${current_price:.0f}')
print(f'   MA{short_win}: ${ma_short:.0f}')
print(f'   MA{long_win}: ${ma_long:.0f}')
print(f'   Difference: ${ma_short - ma_long:.0f}')
print(f'   Signal: {signal} ({"LONG" if signal == 1 else "SHORT"})')
print()

# Check against known position
# YOU ARE CURRENTLY LONG
current_pos = 1
print(f'🎯 Prediction for your LONG position:')
if signal == current_pos:
    print('   ✅ SAFE TO START - Signal matches your position')
    print('   No immediate trade will occur')
    print()
    print('   Safe to proceed with:')
    print('   screen -S server')
    print('   python src/tdr.py --server')
else:
    print('   ⚠️  WARNING - WOULD TRADE IMMEDIATELY!')
    print('   Signal wants SHORT but you are LONG')
    print('   System would sell your BTC on resume')
    print()
    print('   Options:')
    print('   1. Change config to different MA')
    print('   2. Accept the trade')
    print('   3. Wait for signal to change')

print()
print("=" * 40)