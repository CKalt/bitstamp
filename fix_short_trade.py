#!/usr/bin/env python3
"""
Fix the missing SHORT trade in trades.json
"""
import json
from datetime import datetime

# The SELL trade that needs to be added
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
        "type": "1",
        "amount": "1.45378686",
        "price": "116970"
    },
    "trade_group_id": "SELL_20250721002827",
    "total_usd": 170090.79,
    "fees": 170.09,  # Assuming 0.1% fee
    "net_proceeds": 169920.70
}

print(json.dumps(sell_trade, indent=2))
print("\nThis trade should be added to trades.json on the server")
print(f"\nSummary:")
print(f"  Action: SELL {sell_trade['amount']} BTC")
print(f"  Price: ${sell_trade['price']:,.2f}")
print(f"  Total USD: ${sell_trade['total_usd']:,.2f}")
print(f"  Fees: ${sell_trade['fees']:,.2f}")
print(f"  Net: ${sell_trade['net_proceeds']:,.2f}")