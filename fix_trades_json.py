#!/usr/bin/env python3
"""
Append a fake BUY trade to trades.json to fix position tracking
"""
import json
import os
from datetime import datetime

trades_file = "trades.json"

# Create initial structure if file doesn't exist
if not os.path.exists(trades_file):
    trades_data = []
else:
    # Read existing trades
    with open(trades_file, 'r') as f:
        trades_data = json.load(f)
        
# Handle both formats - array or object with trades array
if isinstance(trades_data, dict) and 'trades' in trades_data:
    trades_list = trades_data['trades']
elif isinstance(trades_data, list):
    trades_list = trades_data
else:
    print(f"❌ Unknown trades.json format")
    exit(1)

# Create a fake BUY trade with the correct entry price
fake_trade = {
    "timestamp": datetime.now().isoformat(),
    "type": "BUY",
    "amount": 1.36,
    "price": 117545.0,
    "cost": 159861.2,  # 1.36 * 117545
    "balance_btc": 1.36,
    "balance_usd": 0.0,
    "signal": "MANUAL_RESUME",
    "strategy": "trending",
    "regime": "trending",
    "confidence": 100.0,
    "is_manual_resume": True
}

# Append the fake trade
trades_list.append(fake_trade)

# Save back to file - preserve original format
if isinstance(trades_data, dict):
    trades_data['trades'] = trades_list
    save_data = trades_data
else:
    save_data = trades_list

with open(trades_file, 'w') as f:
    json.dump(save_data, f, indent=2)

print(f"✅ Added fake BUY trade to {trades_file} (appended at end)")
print(f"   Amount: 1.36 BTC")
print(f"   Price: $117,545")
print(f"   Cost: $159,861.20")
print(f"   Timestamp: {fake_trade['timestamp']}")
print("")
print("This should fix the position tracking to show:")
print("- Position: LONG")
print("- Entry price: $117,545")
print("- Amount: 1.36 BTC")