#!/usr/bin/env python3
"""
Fix position tracking to match actual trades
"""
import json
import os
import sys
from datetime import datetime

def load_json_file(filepath):
    """Load JSON file if it exists."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def save_json_file(filepath, data):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {filepath}")

def calculate_actual_position():
    """Calculate actual position from trades.json."""
    trades = load_json_file('trades.json')
    if not trades:
        print("❌ No trades.json found")
        return None
    
    # Calculate current position from trades
    btc_balance = 0.0
    usd_balance = 10000.0  # Starting balance
    
    for trade in trades:
        if trade['type'] == 'buy':
            btc_bought = trade['amount']
            usd_spent = btc_bought * trade['price']
            fees = trade.get('order_result', {}).get('fee', 0)
            if isinstance(fees, str):
                fees = float(fees)
            
            btc_balance += btc_bought
            usd_balance -= (usd_spent + fees)
            
        elif trade['type'] == 'sell':
            btc_sold = trade['amount']
            usd_received = btc_sold * trade['price']
            fees = trade.get('order_result', {}).get('fee', 0)
            if isinstance(fees, str):
                fees = float(fees)
            
            btc_balance -= btc_sold
            usd_balance += (usd_received - fees)
    
    # Determine position
    if btc_balance > 0.001:  # LONG position
        # Get the last complete buy group
        buy_trades = [t for t in trades if t['type'] == 'buy']
        last_group_id = buy_trades[-1].get('trade_group_id')
        group_trades = [t for t in buy_trades if t.get('trade_group_id') == last_group_id]
        
        total_btc = sum(t['amount'] for t in group_trades)
        total_usd = sum(t['amount'] * t['price'] for t in group_trades)
        avg_price = total_usd / total_btc if total_btc > 0 else 0
        
        return {
            'position': 'LONG',
            'btc_balance': btc_balance,
            'usd_balance': usd_balance,
            'entry_price': avg_price,
            'total_btc': total_btc
        }
    else:  # SHORT position
        sell_trades = [t for t in trades if t['type'] == 'sell']
        if sell_trades:
            last_sell = sell_trades[-1]
            entry_price = last_sell['price']
            
            return {
                'position': 'SHORT',
                'btc_balance': btc_balance,
                'usd_balance': usd_balance,
                'entry_price': entry_price,
                'total_usd': usd_balance
            }
    
    return None

def fix_resume_file(actual_position):
    """Fix resume-auto-trade.json to match actual position."""
    if actual_position['position'] == 'LONG':
        resume_data = {
            "position": "LONG",
            "amount": actual_position['btc_balance'],
            "unit": "btc",
            "entry_price": actual_position['entry_price'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "command": f"resume_auto_trade {actual_position['btc_balance']:.8f}btc long {int(actual_position['entry_price'])}",
            "strategy": "MA",
            "ma_short": 4,
            "ma_long": 20,
            "source": "fixed_from_trades"
        }
    else:  # SHORT
        resume_data = {
            "position": "SHORT",
            "amount": actual_position['usd_balance'],
            "unit": "usd",
            "entry_price": actual_position['entry_price'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "command": f"resume_auto_trade {actual_position['usd_balance']:.2f}usd short {int(actual_position['entry_price'])}",
            "strategy": "MA",
            "ma_short": 4,
            "ma_long": 20,
            "source": "fixed_from_trades"
        }
    
    save_json_file('resume-auto-trade.json', resume_data)
    return resume_data

def fix_best_strategy(actual_position):
    """Update best_strategy.json with correct entry price."""
    config = load_json_file('best_strategy.json') or {}
    
    config['Last_Trade_Price'] = actual_position['entry_price']
    config['Last_Trade_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config['auto_resume'] = True  # Enable auto-resume
    
    save_json_file('best_strategy.json', config)
    return config

def main():
    """Main fix function."""
    print("=" * 60)
    print("POSITION TRACKING FIX")
    print("=" * 60)
    
    # Change to server directory
    os.chdir('/home/chris/projects/bitstamp')
    
    # Calculate actual position
    print("\n📊 Calculating actual position from trades...")
    actual_position = calculate_actual_position()
    
    if not actual_position:
        print("❌ Could not calculate position from trades")
        return
    
    print(f"\n✅ ACTUAL POSITION FOUND:")
    print(f"   Position: {actual_position['position']}")
    print(f"   BTC Balance: {actual_position['btc_balance']:.8f}")
    print(f"   USD Balance: ${actual_position['usd_balance']:,.2f}")
    print(f"   Entry Price: ${actual_position['entry_price']:,.2f}")
    
    # Backup existing files
    print("\n📦 Creating backups...")
    for filename in ['resume-auto-trade.json', 'best_strategy.json']:
        if os.path.exists(filename):
            backup_name = f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(filename, backup_name)
            print(f"   Backed up {filename} → {backup_name}")
    
    # Fix resume file
    print("\n🔧 Fixing resume-auto-trade.json...")
    resume_data = fix_resume_file(actual_position)
    print(f"   Command will be: {resume_data['command']}")
    
    # Fix best_strategy.json
    print("\n🔧 Fixing best_strategy.json...")
    config = fix_best_strategy(actual_position)
    print(f"   Last_Trade_Price set to: ${config['Last_Trade_Price']:,.2f}")
    
    print("\n" + "=" * 60)
    print("✅ POSITION TRACKING FIXED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the server: python src/tdr.py --server")
    print("2. Wait for history to load (~3-5 minutes)")
    print("3. The server should auto-resume with the correct position")
    print("\nIf auto-resume doesn't work, manually run:")
    print(f"   {resume_data['command']}")

if __name__ == "__main__":
    main()