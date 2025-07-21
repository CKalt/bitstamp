#!/usr/bin/env python3
"""
Script to add the missing SELL trade to trades.json
Run this on the server to properly record the SHORT entry
"""
import json
import os
from datetime import datetime

# Path to trades.json
TRADES_FILE = "/home/chris/projects/bitstamp/trades.json"

# The SELL trade that flipped us from LONG to SHORT
sell_trade = {
    "type": "sell",
    "symbol": "btcusd",
    "amount": 1.45378686,
    "price": 116970.0,
    "timestamp": "2025-07-21 00:28:27",
    "signal_timestamp": "2025-07-21 00:28:00",
    "data_source": "live",
    "live_trading": True,
    "reason": "Manual sell to flip from LONG to SHORT - MA signal",
    "order_result": {
        "id": "1901456315228164",
        "market": "BTC/USD",
        "datetime": "2025-07-21 00:28:27.867000",
        "type": "1",  # 1 = sell order
        "amount": "1.45378686",
        "price": "116970"
    },
    "trade_group_id": "SELL_20250721002827",
    "multi_part_sequence": 1,
    "multi_part_total": 1
}

# Load existing trades
if os.path.exists(TRADES_FILE):
    with open(TRADES_FILE, 'r') as f:
        trades = json.load(f)
else:
    trades = []

# Check if this trade already exists
trade_exists = False
for trade in trades:
    if trade.get('order_result', {}).get('id') == sell_trade['order_result']['id']:
        trade_exists = True
        print(f"Trade {sell_trade['order_result']['id']} already exists in trades.json")
        break

if not trade_exists:
    # Add the new trade
    trades.append(sell_trade)
    
    # Sort trades by timestamp
    trades.sort(key=lambda x: x.get('timestamp', ''))
    
    # Save back to file
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=2)
    
    print(f"✅ Added SELL trade to trades.json:")
    print(f"   Amount: {sell_trade['amount']} BTC")
    print(f"   Price: ${sell_trade['price']:,.2f}")
    print(f"   Total: ${sell_trade['amount'] * sell_trade['price']:,.2f}")
    print(f"   Timestamp: {sell_trade['timestamp']}")
else:
    print("Trade already exists, not adding duplicate")

print(f"\nTotal trades in file: {len(trades)}")