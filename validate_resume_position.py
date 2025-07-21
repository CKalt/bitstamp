#!/usr/bin/env python3
"""
Validate resume position matches trades.json
Accounts for multiple BUY trades at 90% each due to Bitstamp constraints
"""
import json
import os
import sys

def validate_position():
    # Read resume file
    resume_file = "resume-auto-trade.json"
    if not os.path.exists(resume_file):
        print("❌ No resume-auto-trade.json found")
        return False
        
    with open(resume_file, 'r') as f:
        resume_data = json.load(f)
    
    resume_position = resume_data.get('position', '').upper()
    resume_amount = resume_data.get('amount', 0)
    resume_entry = resume_data.get('entry_price', 0)
    
    print(f"Resume position: {resume_position} {resume_amount} BTC @ ${resume_entry}")
    
    # Read trades file
    trades_file = "trades.json"
    if not os.path.exists(trades_file):
        print("❌ No trades.json found - cannot validate")
        return False
        
    with open(trades_file, 'r') as f:
        trades_data = json.load(f)
    
    trades = trades_data.get('trades', [])
    if not trades:
        print("❌ No trades in trades.json")
        return False
    
    # Find the most recent position-establishing trades
    # Look for either: last SELL trade OR last 1-3 consecutive BUY trades
    last_trade = trades[-1]
    
    if last_trade['type'] == 'SELL':
        # Validate against last SELL trade only
        position_type = 'SHORT'
        calculated_entry = last_trade['price']
        total_btc = last_trade['amount'] / last_trade['price']  # BTC sold
        print(f"\nValidating against last SELL trade: {last_trade['amount']} USD @ ${last_trade['price']}")
    
    elif last_trade['type'] == 'BUY':
        # Find last 1-3 consecutive BUY trades
        consecutive_buys = []
        for i in range(len(trades) - 1, -1, -1):
            if trades[i]['type'] == 'BUY':
                consecutive_buys.insert(0, trades[i])
                if len(consecutive_buys) >= 3:
                    break
            else:
                break
        
        # Sum up the consecutive BUYs
        total_btc = sum(t['amount'] for t in consecutive_buys)
        total_cost = sum(t['amount'] * t['price'] for t in consecutive_buys)
        position_type = 'LONG'
        calculated_entry = total_cost / total_btc if total_btc > 0 else 0
        
        print(f"\nValidating against last {len(consecutive_buys)} consecutive BUY trade(s):")
        for t in consecutive_buys:
            print(f"  {t['amount']} BTC @ ${t['price']}")
        print(f"  Total: {total_btc:.8f} BTC @ average ${calculated_entry:.2f}")
    
    else:
        print("❌ Unknown trade type")
        return False
    
    # Validate with tolerance
    tolerance = 0.01  # 1% tolerance for amounts
    price_tolerance = 100  # $100 tolerance for price
    
    position_match = resume_position == position_type
    amount_match = abs(abs(total_btc) - resume_amount) / resume_amount < tolerance if resume_amount > 0 else True
    price_match = abs(calculated_entry - resume_entry) < price_tolerance
    
    if position_match and amount_match and price_match:
        print("\n✅ Position validation PASSED")
        return True
    else:
        print("\n❌ Position validation FAILED")
        if not position_match:
            print(f"   Position type mismatch: {resume_position} vs {position_type}")
        if not amount_match:
            print(f"   Amount mismatch: {resume_amount} vs {abs(total_btc):.8f}")
        if not price_match:
            print(f"   Entry price mismatch: ${resume_entry} vs ${calculated_entry:.2f}")
        
        # Show recent trades for debugging
        print("\nRecent trades:")
        for trade in trades[-5:]:
            print(f"   {trade['timestamp']}: {trade['type']} {trade['amount']} @ ${trade['price']}")
        
        return False

if __name__ == "__main__":
    if not validate_position():
        print("\n⚠️  ABORTING: Resume position does not match trades.json")
        print("Please fix the discrepancy before starting the server")
        sys.exit(1)
    else:
        print("\nPosition validated - safe to start server")