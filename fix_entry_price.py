#!/usr/bin/env python3
"""
Fix entry price issue - clear old trade references
"""
import os
import json
import glob
from datetime import datetime

print("Fixing entry price issue...")

# 1. Update resume file to ensure no trade references
resume_file = "resume-auto-trade.json"
if os.path.exists(resume_file):
    with open(resume_file, 'r') as f:
        data = json.load(f)
    
    # Clear any trade references
    data['trade_references'] = []
    data['trades_executed'] = 0
    data['last_trade_time'] = None
    data['entry_price'] = 117545  # Force correct entry price
    
    with open(resume_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Cleared trade references in {resume_file}")

# 2. Remove any .trades files that might have old data
trades_files = glob.glob("*.trades")
for tf in trades_files:
    backup = f"{tf}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.rename(tf, backup)
    print(f"✅ Moved {tf} to {backup}")

# 3. Update best_strategy.json to remove Last_Trade_Price
best_strategy_file = "best_strategy.json"
if os.path.exists(best_strategy_file):
    with open(best_strategy_file, 'r') as f:
        strategy = json.load(f)
    
    # Remove Last_Trade_Price if it exists
    if 'Last_Trade_Price' in strategy:
        del strategy['Last_Trade_Price']
        with open(best_strategy_file, 'w') as f:
            json.dump(strategy, f, indent=2)
        print(f"✅ Removed Last_Trade_Price from {best_strategy_file}")

print("\n✅ Entry price issue fixed!")
print("Now restart the server with: python src/tdr.py --server")