#!/usr/bin/env python3
import json
import sys
from datetime import datetime, timedelta

# Load trades
with open('/home/chris/projects/bitstamp/trades.json', 'r') as f:
    data = json.load(f)
    # Handle both formats - array or object with trades array
    if isinstance(data, list):
        trades = data
    else:
        trades = data['trades']

# Get trades from last 24-48 hours
recent_trades = []
for t in trades:
    if '2025-08-04' in t['timestamp'] or '2025-08-05' in t['timestamp']:
        recent_trades.append(t)

print(f"📊 RECENT TRADING ANALYSIS (Aug 4-5)")
print("=" * 50)
print(f"Total trades: {len(recent_trades)}")
print()

# Group by buy/sell
buys = [t for t in recent_trades if t['type'] == 'buy']
sells = [t for t in recent_trades if t['type'] == 'sell']

print(f"BUYs: {len(buys)}")
print(f"SELLs: {len(sells)}")
print()

# Calculate P&L
total_btc_bought = sum(t['amount'] for t in buys)
total_usd_spent = sum(t['amount'] * t['price'] for t in buys)
avg_buy_price = total_usd_spent / total_btc_bought if total_btc_bought > 0 else 0

total_btc_sold = sum(t['amount'] for t in sells)
total_usd_received = sum(t['amount'] * t['price'] for t in sells)
avg_sell_price = total_usd_received / total_btc_sold if total_btc_sold > 0 else 0

print(f"Total BTC bought: {total_btc_bought:.8f} @ avg ${avg_buy_price:,.2f}")
print(f"Total BTC sold: {total_btc_sold:.8f} @ avg ${avg_sell_price:,.2f}")
print()

# Show trades chronologically
print("Trade History:")
print("-" * 50)
for t in sorted(recent_trades, key=lambda x: x['timestamp']):
    print(f"{t['timestamp']}: {t['type'].upper():4} {t['amount']:.8f} BTC @ ${t['price']:,}")
    
# Calculate realized P&L if we've closed positions
if total_btc_sold > 0:
    realized_pnl = total_usd_received - (total_btc_sold / total_btc_bought * total_usd_spent)
    print()
    print(f"Realized P&L: ${realized_pnl:,.2f}")