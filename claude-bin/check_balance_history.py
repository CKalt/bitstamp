#!/usr/bin/env python3
import json
from datetime import datetime

print("CHECKING BITSTAMP BALANCE HISTORY")
print("=" * 50)

# Check account balance files
import os
import glob

balance_files = glob.glob('/home/chris/projects/bitstamp/account_balance_*.json')
balance_files.sort()

print(f"\nFound {len(balance_files)} balance snapshots")

# Show recent balances
recent = balance_files[-10:]
for f in recent:
    with open(f) as bf:
        data = json.load(bf)
        timestamp = f.split('_')[-1].replace('.json', '')
        usd = float(data.get('usd_balance', 0))
        btc = float(data.get('btc_balance', 0))
        btc_usd = float(data.get('btcusd_balance', 0))
        total_usd = usd + btc_usd
        
        # Parse timestamp
        ts = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}"
        print(f"{ts}: USD: ${total_usd:,.2f} (BTC: {btc:.8f})")

# Now check trades after the fix
print("\n" + "=" * 50)
print("TRADES AFTER HOURLY FIX (Aug 4 18:40 UTC):")
print("=" * 50)

with open('/home/chris/projects/bitstamp/trades.json') as f:
    trades = json.load(f)

fix_time = "2025-08-04 18:40"
after_fix = [t for t in trades if t['timestamp'] > fix_time]

print(f"\nTotal trades after fix: {len(after_fix)}")
print("-" * 50)

btc_balance = 0
usd_spent = 0
usd_received = 0

for t in sorted(after_fix, key=lambda x: x['timestamp']):
    print(f"{t['timestamp']}: {t['type'].upper():4} {t['amount']:.8f} BTC @ ${t['price']:,}")
    if t['type'] == 'buy':
        btc_balance += t['amount']
        usd_spent += t['amount'] * t['price']
    else:
        btc_balance -= t['amount']
        usd_received += t['amount'] * t['price']

print(f"\nSummary since fix:")
print(f"BTC bought: {usd_spent/114000:.8f} BTC for ${usd_spent:,.2f}")
print(f"BTC sold: {usd_received/114000:.8f} BTC for ${usd_received:,.2f}")
print(f"Net USD flow: ${usd_received - usd_spent:,.2f}")