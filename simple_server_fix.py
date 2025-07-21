#!/usr/bin/env python3
"""
Simple server fix - just the essentials
"""
import os
import json
import shutil
from datetime import datetime

# 1. Clean up old position data
resume_file = "resume-auto-trade.json"
if os.path.exists(resume_file):
    backup = f"{resume_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.move(resume_file, backup)
    print(f"✅ Moved old {resume_file} to {backup}")

# 2. Create correct LONG position
resume_data = {
    "timestamp": datetime.now().isoformat(),
    "position": "LONG",
    "amount": 1.36,
    "unit": "btc",
    "entry_price": 117545,
    "current_price": 117545,
    "unrealized_pnl": 0.0,
    "command": "resume_auto_trade 1.36btc long 117545",
    "strategy": {
        "type": "AdaptiveMultiStrategy",
        "short_window": 6,
        "long_window": 34,
        "current_regime": "unknown",
        "active_strategy": "trending"
    },
    "balances": {
        "btc": 1.36,
        "usd": 0.0
    },
    "trades_executed": 0,
    "last_trade_time": None,
    "trade_references": [],
    "pivot_protection": {
        "enabled": True,
        "tracker": {}
    }
}

with open(resume_file, "w") as f:
    json.dump(resume_data, f, indent=2)
print(f"✅ Created resume file: LONG 1.36 BTC @ $117,545")

print("\n✅ Ready to start server!")
print("\nRun: python src/tdr.py --server")