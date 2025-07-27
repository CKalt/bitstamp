#!/usr/bin/env python3
"""
Analyze current position state from trades.json and resume files
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

def analyze_trades():
    """Analyze trades.json to determine actual position."""
    trades = load_json_file('trades.json')
    if not trades:
        print("❌ No trades.json found")
        return None
    
    print(f"\n📊 Found {len(trades)} trades in trades.json")
    
    # Show last 5 trades
    print("\n🔍 Last 5 trades:")
    for trade in trades[-5:]:
        trade_type = trade.get('type', 'unknown')
        amount = trade.get('amount', 0)
        price = trade.get('price', 0)
        timestamp = trade.get('timestamp', 'unknown')
        print(f"  {timestamp}: {trade_type.upper()} {amount:.8f} BTC @ ${price:,.2f}")
    
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
    
    # Determine position and entry price
    if btc_balance > 0.001:  # LONG position
        position = "LONG"
        # Calculate average entry price from recent buy trades
        buy_trades = [t for t in trades if t['type'] == 'buy']
        if buy_trades:
            # Group by trade_group_id to find complete buy transactions
            last_group_id = buy_trades[-1].get('trade_group_id')
            group_trades = [t for t in buy_trades if t.get('trade_group_id') == last_group_id]
            
            total_btc = sum(t['amount'] for t in group_trades)
            total_usd = sum(t['amount'] * t['price'] for t in group_trades)
            avg_price = total_usd / total_btc if total_btc > 0 else 0
            
            print(f"\n✅ ACTUAL POSITION: LONG")
            print(f"   BTC Balance: {btc_balance:.8f}")
            print(f"   USD Balance: ${usd_balance:,.2f}")
            print(f"   Entry Price: ${avg_price:,.2f}")
            return {
                'position': 'LONG',
                'btc_balance': btc_balance,
                'usd_balance': usd_balance,
                'entry_price': avg_price
            }
    else:  # SHORT position
        position = "SHORT"
        # Get entry price from last sell trade
        sell_trades = [t for t in trades if t['type'] == 'sell']
        if sell_trades:
            last_sell = sell_trades[-1]
            entry_price = last_sell['price']
            
            print(f"\n✅ ACTUAL POSITION: SHORT")
            print(f"   BTC Balance: {btc_balance:.8f}")
            print(f"   USD Balance: ${usd_balance:,.2f}")
            print(f"   Entry Price: ${entry_price:,.2f}")
            return {
                'position': 'SHORT',
                'btc_balance': btc_balance,
                'usd_balance': usd_balance,
                'entry_price': entry_price
            }
    
    return None

def analyze_resume_file():
    """Analyze resume-auto-trade.json."""
    resume_data = load_json_file('resume-auto-trade.json')
    if not resume_data:
        print("\n❌ No resume-auto-trade.json found")
        return None
    
    print("\n📄 Resume file shows:")
    print(f"   Position: {resume_data.get('position', 'unknown')}")
    print(f"   Amount: {resume_data.get('amount', 0):.8f} {resume_data.get('unit', '')}")
    print(f"   Entry Price: ${resume_data.get('entry_price', 0):,.2f}")
    print(f"   Command: {resume_data.get('command', 'none')}")
    
    return resume_data

def analyze_best_strategy():
    """Analyze best_strategy.json."""
    config = load_json_file('best_strategy.json')
    if not config:
        print("\n❌ No best_strategy.json found")
        return None
    
    print("\n⚙️ Config shows:")
    print(f"   Auto Resume: {config.get('auto_resume', False)}")
    print(f"   Last Trade Price: ${config.get('Last_Trade_Price', 0):,.2f}")
    
    return config

def main():
    """Main analysis function."""
    print("=" * 60)
    print("POSITION STATE ANALYSIS")
    print("=" * 60)
    
    # Change to server directory
    os.chdir('/home/chris/projects/bitstamp')
    
    # Analyze actual position from trades
    actual_position = analyze_trades()
    
    # Analyze resume file
    resume_data = analyze_resume_file()
    
    # Analyze config
    config = analyze_best_strategy()
    
    # Show discrepancies
    print("\n" + "=" * 60)
    print("DISCREPANCY ANALYSIS")
    print("=" * 60)
    
    if actual_position and resume_data:
        if actual_position['position'] != resume_data.get('position', '').upper():
            print(f"\n⚠️ POSITION MISMATCH!")
            print(f"   Trades show: {actual_position['position']}")
            print(f"   Resume shows: {resume_data.get('position', 'unknown')}")
        
        if abs(actual_position['entry_price'] - resume_data.get('entry_price', 0)) > 10:
            print(f"\n⚠️ ENTRY PRICE MISMATCH!")
            print(f"   Trades show: ${actual_position['entry_price']:,.2f}")
            print(f"   Resume shows: ${resume_data.get('entry_price', 0):,.2f}")
    
    return actual_position

if __name__ == "__main__":
    main()