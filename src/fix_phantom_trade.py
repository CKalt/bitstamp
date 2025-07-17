#!/usr/bin/env python3
"""
Emergency fix for phantom SELL trade that occurred during server restart.
This script safely removes the incorrect trade and prepares for position correction.
"""

import json
import shutil
from datetime import datetime
import os
import sys

def main():
    """Remove phantom SELL trade and fix position mismatch."""
    
    print("🚨 PHANTOM TRADE FIX UTILITY")
    print("=" * 50)
    
    # Determine path based on environment
    if os.path.exists("/home/chris/projects/bitstamp/trades.json"):
        trades_file = "/home/chris/projects/bitstamp/trades.json"
        print("✅ Running on server (chriskoin)")
    else:
        trades_file = "trades.json"
        print("⚠️  Running locally - for testing only")
    
    backup_file = trades_file.replace('.json', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    try:
        # Read current trades
        with open(trades_file, 'r') as f:
            trades = json.load(f)
        
        print(f"✅ Loaded {len(trades)} trades from {trades_file}")
        
        # Find the phantom trade
        phantom_indices = []
        
        for i, trade in enumerate(trades):
            # Look for the specific phantom trade
            if (trade.get('timestamp') == '2025-07-17 01:32:39' and 
                trade.get('type') == 'sell' and
                abs(trade.get('amount', 0) - 1.50271956) < 0.0001 and
                trade.get('reason', '').startswith('Pivot break: below support')):
                phantom_indices.append(i)
                print(f"\n🔍 Found phantom SELL trade at index {i}:")
                print(f"   Type: {trade['type']}")
                print(f"   Amount: {trade['amount']} BTC") 
                print(f"   Price: ${trade['price']}")
                print(f"   Timestamp: {trade['timestamp']}")
                print(f"   Reason: {trade['reason']}")
                print(f"   Order ID: {trade.get('order_result', {}).get('id', 'N/A')}")
        
        if not phantom_indices:
            print("\n❌ No phantom trade found! Checking for recent suspicious SELL trades...")
            
            # Look for any SELL trades in the last hour
            for i in range(max(0, len(trades) - 10), len(trades)):
                trade = trades[i]
                if trade.get('type') == 'sell':
                    print(f"\n📍 Recent SELL at index {i}:")
                    print(f"   Timestamp: {trade.get('timestamp')}")
                    print(f"   Amount: {trade.get('amount')} BTC")
                    print(f"   Price: ${trade.get('price')}")
                    print(f"   Reason: {trade.get('reason')}")
            
            return False
        
        # Create backup
        shutil.copy2(trades_file, backup_file)
        print(f"\n💾 Created backup: {backup_file}")
        
        # Remove phantom trades (in reverse order to maintain indices)
        for idx in sorted(phantom_indices, reverse=True):
            removed = trades.pop(idx)
            print(f"\n🗑️  Removed phantom trade from index {idx}")
        
        # Write fixed trades back
        with open(trades_file, 'w') as f:
            json.dump(trades, f, indent=2)
        
        print(f"\n✅ Wrote {len(trades)} trades back to {trades_file}")
        
        # Show the last real trades
        print("\n📊 Last 5 trades now in file:")
        for trade in trades[-5:]:
            trade_type = trade['type'].upper()
            symbol = "→" if trade_type == "BUY" else "←"
            print(f"   {trade['timestamp']} {symbol} {trade_type} {trade.get('amount', 'N/A')} BTC @ ${trade.get('price', 'N/A')}")
        
        # Find the last BUY to determine correct position
        last_buy_amount = 0
        last_buy_price = 0
        for trade in reversed(trades):
            if trade.get('type') == 'buy':
                # Sum up multi-part BUY trades
                if 'trade_group_id' in trade:
                    group_id = trade['trade_group_id']
                    group_total = 0
                    group_value = 0
                    for t in trades:
                        if t.get('trade_group_id') == group_id:
                            group_total += t.get('amount', 0)
                            group_value += t.get('amount', 0) * t.get('price', 0)
                    last_buy_amount = group_total
                    last_buy_price = group_value / group_total if group_total > 0 else 0
                else:
                    last_buy_amount = trade.get('amount', 0)
                    last_buy_price = trade.get('price', 0)
                break
        
        print("\n✅ PHANTOM TRADE REMOVED SUCCESSFULLY!")
        print("\n🎯 CRITICAL NEXT STEPS:")
        print("1. The server auto-trader is currently STOPPED")
        print("2. Restart the TDR server:")
        print("   pkill -f tdr_server.py")
        print("   cd /home/chris/projects/bitstamp")  
        print("   python src/tdr.py --server")
        print("\n3. Resume with the CORRECT position from Bitstamp:")
        print(f"   resume_auto_trade 1.50271956btc long 117266")
        print("\n⚠️  VERIFY with Bitstamp that you are LONG 1.50271956 BTC!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)