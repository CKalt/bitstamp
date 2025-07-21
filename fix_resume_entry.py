#!/usr/bin/env python3
"""
Force correct entry price in resume file
"""
import json

resume_file = "resume-auto-trade.json"

# Read current file
with open(resume_file, 'r') as f:
    data = json.load(f)

# Force correct entry price
data['entry_price'] = 117545
data['current_price'] = 117545  # Update this too if needed
data['command'] = "resume_auto_trade 1.36btc long 117545"

# Write back
with open(resume_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✅ Updated {resume_file} with entry price $117,545")
print("Resume file contents:")
print(f"  Position: {data['position']}")
print(f"  Amount: {data['amount']} {data.get('unit', 'btc')}")
print(f"  Entry Price: ${data['entry_price']}")