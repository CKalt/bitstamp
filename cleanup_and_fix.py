#!/usr/bin/env python3
"""
Clean up duplicate trades and fix resume file
"""
import json
import os

# 1. Fix trades.json - remove duplicate fake trades
trades_file = "trades.json"
if os.path.exists(trades_file):
    with open(trades_file, 'r') as f:
        trades_data = json.load(f)
    
    # Handle both formats
    if isinstance(trades_data, dict) and 'trades' in trades_data:
        trades = trades_data['trades']
    elif isinstance(trades_data, list):
        trades = trades_data
    else:
        trades = []
    
    # Remove all trades with is_manual_resume flag
    cleaned_trades = [t for t in trades if not t.get('is_manual_resume', False)]
    
    # Add ONE fake BUY trade at the end
    fake_trade = {
        "timestamp": "2025-07-21T15:30:00.000000",
        "type": "BUY",
        "amount": 1.36,
        "price": 117545.0,
        "cost": 159861.2,
        "balance_btc": 1.36,
        "balance_usd": 0.0,
        "signal": "MANUAL_RESUME",
        "strategy": "trending",
        "regime": "trending", 
        "confidence": 100.0,
        "is_manual_resume": True
    }
    cleaned_trades.append(fake_trade)
    
    # Save back
    if isinstance(trades_data, dict):
        trades_data['trades'] = cleaned_trades
        save_data = trades_data
    else:
        save_data = cleaned_trades
    
    with open(trades_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"✅ Cleaned trades.json - removed duplicates, added single fake BUY")

# 2. Fix resume file
resume_file = "resume-auto-trade.json"
if os.path.exists(resume_file):
    with open(resume_file, 'r') as f:
        resume_data = json.load(f)
    
    # Update entry price
    resume_data['entry_price'] = 117545
    resume_data['current_price'] = 117545
    resume_data['command'] = "resume_auto_trade 1.36btc long 117545"
    
    with open(resume_file, 'w') as f:
        json.dump(resume_data, f, indent=2)
    
    print(f"✅ Fixed resume-auto-trade.json with entry price $117,545")

print("\n✅ All fixed! Now run: python src/tdr.py --server")