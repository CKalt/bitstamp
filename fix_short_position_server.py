#!/usr/bin/env python3
"""
Comprehensive fix script to correct SHORT position tracking on server
Run this directly on the server to fix both trades.json and resume-auto-trade.json
"""
import json
import os
import sys
from datetime import datetime
import shutil

# Configuration
TRADES_FILE = "/home/chris/projects/bitstamp/trades.json"
RESUME_FILE = "/home/chris/projects/bitstamp/resume-auto-trade.json"
BACKUP_DIR = "/home/chris/projects/bitstamp/backups"

# SHORT position details
BTC_SOLD = 1.45378686
SELL_PRICE = 116970.0
USD_RECEIVED = BTC_SOLD * SELL_PRICE  # $170,090.79
SELL_TIMESTAMP = "2025-07-21 00:28:27"

def backup_file(filepath):
    """Create backup of file before modifying"""
    if not os.path.exists(filepath):
        print(f"⚠️  File not found: {filepath}")
        return False
    
    # Create backup directory if needed
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(filepath)
    backup_path = os.path.join(BACKUP_DIR, f"{filename}.{timestamp}.bak")
    
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backed up {filename} to {backup_path}")
    return True

def fix_trades_json():
    """Add missing SELL trade to trades.json"""
    print("\n📄 Fixing trades.json...")
    
    # Backup first
    if not backup_file(TRADES_FILE):
        return False
    
    # Load existing trades
    try:
        with open(TRADES_FILE, 'r') as f:
            trades = json.load(f)
        print(f"   Found {len(trades)} existing trades")
    except FileNotFoundError:
        print("   trades.json not found, creating new file")
        trades = []
    except json.JSONDecodeError:
        print("❌ Error: trades.json is corrupted")
        return False
    
    # The SELL trade that flipped us from LONG to SHORT
    sell_trade = {
        "type": "sell",
        "symbol": "btcusd",
        "amount": BTC_SOLD,
        "price": SELL_PRICE,
        "timestamp": SELL_TIMESTAMP,
        "signal_timestamp": "2025-07-21 00:28:00",
        "data_source": "live",
        "live_trading": True,
        "reason": "Manual sell to flip from LONG to SHORT - MA signal",
        "order_result": {
            "id": "1901456315228164",
            "market": "BTC/USD",
            "datetime": "2025-07-21 00:28:27.867000",
            "type": "1",  # 1 = sell order
            "amount": str(BTC_SOLD),
            "price": str(SELL_PRICE)
        },
        "trade_group_id": "SELL_20250721002827",
        "multi_part_sequence": 1,
        "multi_part_total": 1
    }
    
    # Check if this trade already exists
    trade_exists = False
    for trade in trades:
        if trade.get('order_result', {}).get('id') == sell_trade['order_result']['id']:
            trade_exists = True
            print("   ⚠️  SELL trade already exists in trades.json")
            break
    
    if not trade_exists:
        # Add the new trade
        trades.append(sell_trade)
        
        # Sort trades by timestamp
        trades.sort(key=lambda x: x.get('timestamp', ''))
        
        # Save back to file
        with open(TRADES_FILE, 'w') as f:
            json.dump(trades, f, indent=2)
        
        print(f"   ✅ Added SELL trade:")
        print(f"      Amount: {BTC_SOLD} BTC")
        print(f"      Price: ${SELL_PRICE:,.2f}")
        print(f"      Total: ${USD_RECEIVED:,.2f}")
        print(f"      Timestamp: {SELL_TIMESTAMP}")
    
    print(f"   Total trades now: {len(trades)}")
    return True

def fix_resume_json():
    """Fix resume-auto-trade.json with correct SHORT position"""
    print("\n📄 Fixing resume-auto-trade.json...")
    
    # Backup first
    if not backup_file(RESUME_FILE):
        return False
    
    # Load current resume data
    try:
        with open(RESUME_FILE, 'r') as f:
            current_data = json.load(f)
        print(f"   Current position: {current_data.get('position', 'UNKNOWN')}")
        print(f"   Current entry: ${current_data.get('entry_price', 0):,.2f}")
    except FileNotFoundError:
        print("   resume-auto-trade.json not found")
        current_data = {}
    except json.JSONDecodeError:
        print("❌ Error: resume-auto-trade.json is corrupted")
        return False
    
    # Create correct SHORT position data
    resume_data = {
        "timestamp": "2025-07-21T00:28:27.000000",
        "position": "SHORT",
        "amount": USD_RECEIVED,
        "unit": "usd",
        "entry_price": SELL_PRICE,
        "current_price": current_data.get('current_price', 117305.0),
        "unrealized_pnl": (SELL_PRICE - current_data.get('current_price', 117305.0)) * BTC_SOLD,
        "command": f"resume_auto_trade {USD_RECEIVED:.2f}usd short {SELL_PRICE:.0f}",
        "strategy": {
            "type": "AdaptiveMultiStrategy",
            "short_window": 10,
            "long_window": 46,
            "current_regime": "trending",
            "active_strategy": "trending"
        },
        "balances": {
            "btc": 0.0,
            "usd": USD_RECEIVED
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
                "timestamp": SELL_TIMESTAMP,
                "type": "sell",
                "amount": BTC_SOLD,
                "price": SELL_PRICE,
                "trade_group_id": "SELL_20250721002827"
            }
        ],
        "pivot_protection": {
            "enabled": False,
            "tracker": {}
        }
    }
    
    # Save updated resume file
    with open(RESUME_FILE, 'w') as f:
        json.dump(resume_data, f, indent=2)
    
    print(f"   ✅ Updated resume file:")
    print(f"      Position: SHORT")
    print(f"      Entry Price: ${SELL_PRICE:,.2f}")
    print(f"      USD Balance: ${USD_RECEIVED:,.2f}")
    print(f"      P&L: ${resume_data['unrealized_pnl']:,.2f}")
    
    return True

def verify_fix():
    """Verify the fixes were applied correctly"""
    print("\n🔍 Verifying fixes...")
    
    # Check trades.json
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'r') as f:
            trades = json.load(f)
        
        # Look for our SELL trade
        found_sell = False
        for trade in trades:
            if (trade.get('type') == 'sell' and 
                trade.get('timestamp') == SELL_TIMESTAMP and
                abs(trade.get('price', 0) - SELL_PRICE) < 1):
                found_sell = True
                break
        
        if found_sell:
            print("   ✅ SELL trade found in trades.json")
        else:
            print("   ❌ SELL trade NOT found in trades.json")
    
    # Check resume-auto-trade.json
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE, 'r') as f:
            resume_data = json.load(f)
        
        if (resume_data.get('position') == 'SHORT' and
            abs(resume_data.get('entry_price', 0) - SELL_PRICE) < 1):
            print("   ✅ Resume file has correct SHORT position")
            print(f"      Entry price: ${resume_data.get('entry_price', 0):,.2f}")
        else:
            print("   ❌ Resume file does NOT have correct SHORT position")

def main():
    """Main execution"""
    print("=" * 60)
    print("SHORT Position Fix Script")
    print("=" * 60)
    print(f"Server files to fix:")
    print(f"  - {TRADES_FILE}")
    print(f"  - {RESUME_FILE}")
    print(f"\nSHORT position details:")
    print(f"  - BTC Sold: {BTC_SOLD}")
    print(f"  - Sell Price: ${SELL_PRICE:,.2f}")
    print(f"  - USD Received: ${USD_RECEIVED:,.2f}")
    
    # Confirm before proceeding
    response = input("\nProceed with fix? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    # Fix trades.json
    if not fix_trades_json():
        print("\n❌ Failed to fix trades.json")
        return
    
    # Fix resume-auto-trade.json
    if not fix_resume_json():
        print("\n❌ Failed to fix resume-auto-trade.json")
        return
    
    # Verify fixes
    verify_fix()
    
    print("\n" + "=" * 60)
    print("✅ Fix complete! Next steps:")
    print("1. Stop auto-trader: python3 auto_trade.py stop")
    print("2. Resume with fixed position: python3 auto_trade.py resume")
    print("   (It will use the corrected resume-auto-trade.json)")
    print("=" * 60)

if __name__ == "__main__":
    main()