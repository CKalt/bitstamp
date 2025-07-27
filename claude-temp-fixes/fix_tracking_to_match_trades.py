#!/usr/bin/env python3
"""
Fix position tracking to match actual executed trades so the system can continue trading normally.
This doesn't choose a position - it just ensures tracking matches reality.
"""
import json
import os
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

def get_actual_state_from_trades():
    """Calculate the actual position state from executed trades."""
    trades = load_json_file('trades.json')
    if not trades:
        print("❌ No trades.json found")
        return None
    
    # Calculate balances from all trades
    btc_balance = 0.0
    usd_balance = 10000.0  # Initial USD
    
    print("\n📊 Processing trade history...")
    
    for trade in trades:
        if trade['type'] == 'buy':
            btc_bought = trade['amount']
            usd_spent = btc_bought * trade['price']
            fees = float(trade.get('order_result', {}).get('fee', 0) or 0)
            
            btc_balance += btc_bought
            usd_balance -= (usd_spent + fees)
            
        elif trade['type'] == 'sell':
            btc_sold = trade['amount']
            usd_received = btc_sold * trade['price']
            fees = float(trade.get('order_result', {}).get('fee', 0) or 0)
            
            btc_balance -= btc_sold
            usd_balance += (usd_received - fees)
    
    # Get last few trades for context
    print("\n🔍 Recent trades:")
    for trade in trades[-5:]:
        t_type = trade['type'].upper()
        amount = trade['amount']
        price = trade['price']
        time = trade['timestamp']
        print(f"  {time}: {t_type} {amount:.8f} BTC @ ${price:,.2f}")
    
    # Determine position and entry based on last trades
    if btc_balance > 0.001:  # LONG position
        # Find last buy group
        buy_trades = [t for t in trades if t['type'] == 'buy']
        last_group_id = buy_trades[-1].get('trade_group_id')
        group_trades = [t for t in buy_trades if t.get('trade_group_id') == last_group_id]
        
        total_btc = sum(t['amount'] for t in group_trades)
        total_usd = sum(t['amount'] * t['price'] for t in group_trades)
        avg_entry = total_usd / total_btc if total_btc > 0 else 0
        
        return {
            'position': 'LONG',
            'btc_balance': btc_balance,
            'usd_balance': usd_balance,
            'entry_price': avg_entry,
            'position_size': total_btc
        }
    else:  # SHORT position
        sell_trades = [t for t in trades if t['type'] == 'sell']
        if sell_trades:
            last_sell = sell_trades[-1]
            return {
                'position': 'SHORT',
                'btc_balance': btc_balance,
                'usd_balance': usd_balance,
                'entry_price': last_sell['price'],
                'position_size': usd_balance
            }
    
    return None

def main():
    """Fix tracking to match actual trades."""
    print("=" * 60)
    print("FIX POSITION TRACKING TO MATCH EXECUTED TRADES")
    print("=" * 60)
    
    # Change to server directory
    os.chdir('/home/chris/projects/bitstamp')
    
    # Get actual state
    actual_state = get_actual_state_from_trades()
    
    if not actual_state:
        print("❌ Could not determine position from trades")
        return
    
    print(f"\n✅ ACTUAL TRADING STATE:")
    print(f"   Position: {actual_state['position']}")
    print(f"   BTC Balance: {actual_state['btc_balance']:.8f}")
    print(f"   USD Balance: ${actual_state['usd_balance']:,.2f}")
    print(f"   Entry Price: ${actual_state['entry_price']:,.2f}")
    
    # Create backups
    print("\n📦 Creating backups...")
    backup_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for filename in ['resume-auto-trade.json', 'best_strategy.json']:
        if os.path.exists(filename):
            backup_name = f"{filename}.backup_{backup_time}"
            os.rename(filename, backup_name)
            print(f"   Backed up {filename}")
    
    # Create correct resume file
    print("\n🔧 Creating correct resume-auto-trade.json...")
    
    if actual_state['position'] == 'LONG':
        resume_data = {
            "position": "LONG",
            "amount": actual_state['position_size'],
            "unit": "btc",
            "entry_price": actual_state['entry_price'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "command": f"resume_auto_trade {actual_state['position_size']:.8f}btc long {int(actual_state['entry_price'])}",
            "strategy": "MA",
            "ma_short": 4,
            "ma_long": 20,
            "source": "fixed_from_actual_trades"
        }
    else:  # SHORT
        resume_data = {
            "position": "SHORT", 
            "amount": actual_state['position_size'],
            "unit": "usd",
            "entry_price": actual_state['entry_price'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "command": f"resume_auto_trade {actual_state['position_size']:.2f}usd short {int(actual_state['entry_price'])}",
            "strategy": "MA",
            "ma_short": 4,
            "ma_long": 20,
            "source": "fixed_from_actual_trades"
        }
    
    save_json_file('resume-auto-trade.json', resume_data)
    
    # Update best_strategy.json
    print("\n🔧 Updating best_strategy.json...")
    config = load_json_file('best_strategy.json') or {}
    config['Last_Trade_Price'] = actual_state['entry_price']
    config['auto_resume'] = True
    save_json_file('best_strategy.json', config)
    
    print("\n" + "=" * 60)
    print("✅ TRACKING FIXED TO MATCH ACTUAL TRADES")
    print("=" * 60)
    print(f"\nThe system will resume as: {resume_data['command']}")
    print("\nThis matches your actual executed trades. The system will continue")
    print("trading automatically based on MA crossover signals.")

if __name__ == "__main__":
    main()