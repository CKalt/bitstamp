#!/usr/bin/env python3
"""
Script to fix the SHORT position tracking
Run this AFTER adding the sell trade to trades.json
"""
import json
import os

# Calculate correct values for SHORT position
btc_sold = 1.45378686
sell_price = 116970.0
usd_received = btc_sold * sell_price  # $170,090.79

print("SHORT Position Fix Values:")
print(f"  BTC Sold: {btc_sold}")
print(f"  Sell Price: ${sell_price:,.2f}")
print(f"  USD Received: ${usd_received:,.2f}")
print(f"  Entry Price: ${sell_price:,.2f} (SHORT entry is the SELL price)")

# For SHORT positions in the system:
# - position = -1
# - position_size = -btc_sold (NEGATIVE!)
# - position_cost_basis = usd_received
# - Entry price calculation: position_cost_basis / abs(position_size)

print(f"\nPosition tracking values needed:")
print(f"  position: -1 (SHORT)")
print(f"  position_size: {-btc_sold} (negative for SHORT)")
print(f"  position_cost_basis: ${usd_received:.2f}")
print(f"  Calculated entry: ${usd_received / btc_sold:.2f}")

# Update resume-auto-trade.json
resume_file = "/home/chris/projects/bitstamp/resume-auto-trade.json"
resume_data = {
    "timestamp": "2025-07-21T00:28:27.000000",
    "position": "SHORT",
    "amount": usd_received,
    "unit": "usd",
    "entry_price": sell_price,
    "current_price": 117491.0,  # Update with current price
    "unrealized_pnl": (sell_price - 117491.0) * btc_sold,  # Negative = loss
    "command": f"resume_auto_trade {usd_received:.2f}usd short {sell_price:.0f}",
    "strategy": {
        "type": "AdaptiveMultiStrategy",
        "short_window": 10,
        "long_window": 46,
        "current_regime": "trending",
        "active_strategy": "trending"
    },
    "balances": {
        "btc": 0.0,
        "usd": usd_received
    },
    "trades_executed": 1,
    "last_trade_time": "2025-07-21T00:28:27",
    "trade_references": [
        {
            "timestamp": "2025-07-18 15:00:05",
            "type": "buy",
            "amount": 1.30952246,
            "price": 118198.0,
            "trade_group_id": "BUY_20250718150005"
        },
        {
            "timestamp": "2025-07-18 15:00:05",
            "type": "buy",
            "amount": 0.1442644,
            "price": 118198.0,
            "trade_group_id": "BUY_20250718150005"
        },
        {
            "timestamp": "2025-07-21 00:28:27",
            "type": "sell",
            "amount": 1.45378686,
            "price": 116970.0,
            "trade_group_id": "SELL_20250721002827"
        }
    ],
    "pivot_protection": {
        "enabled": False,
        "tracker": {}
    }
}

print(f"\nWriting corrected resume file to: {resume_file}")
with open(resume_file, 'w') as f:
    json.dump(resume_data, f, indent=2)

print("\n✅ Resume file updated with correct SHORT position")
print(f"   Unrealized P&L: ${resume_data['unrealized_pnl']:.2f} (should be negative/loss)")