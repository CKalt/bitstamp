#!/usr/bin/env python3
"""
Create resume-auto-trade.json file for SHORT position
"""
import json
from datetime import datetime

resume_data = {
    "position": "short",
    "amount": "170577usd",
    "entry_price": 117564,
    "btc_amount": 0.0,  # SHORT position has no BTC
    "usd_amount": 170577,
    "last_update": datetime.now().isoformat(),
    "resume_source": "manual_sell",
    "note": "Manual SELL executed, resuming as SHORT position"
}

filename = "resume-auto-trade.json"
with open(filename, 'w') as f:
    json.dump(resume_data, f, indent=2)

print(f"✅ Created {filename}")
print("\n📄 Contents:")
print(json.dumps(resume_data, indent=2))

print("\n📤 To copy to server:")
print(f"scp {filename} chriskoin:/home/chris/projects/bitstamp/")

print("\n⚠️  IMPORTANT: Copy this file to the server BEFORE starting tdr_server.py!")