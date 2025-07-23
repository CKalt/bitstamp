#!/usr/bin/env python3
"""
Fix manual sell transaction in the trading system
Adds the manual sell to trades.json and creates proper resume-auto-trade.json
"""
import json
import os
from datetime import datetime
import sys

# Manual sell details
SELL_PRICE = 117564
USD_HOLDINGS = 170577
SELL_TIME = "2025-07-23T13:00:00Z"  # Approximate time of manual sell

def calculate_btc_amount(usd_amount, price, fee_rate=0.0012):
    """Calculate BTC amount from USD after fees"""
    # USD = BTC * Price * (1 - fee)
    # BTC = USD / (Price * (1 - fee))
    return usd_amount / (price * (1 - fee_rate))

def main():
    print("🔧 Fixing Manual Sell Transaction")
    print("=" * 60)
    
    # Calculate BTC amount that was sold
    btc_sold = calculate_btc_amount(USD_HOLDINGS, SELL_PRICE)
    
    print(f"Sell Price: ${SELL_PRICE:,.2f}")
    print(f"USD Received: ${USD_HOLDINGS:,.2f}")
    print(f"BTC Sold (calculated): {btc_sold:.8f}")
    
    # 1. Load existing trades.json
    trades_file = os.path.join(os.path.dirname(__file__), 'trades.json')
    if os.path.exists(trades_file):
        with open(trades_file, 'r') as f:
            trades = json.load(f)
        print(f"\nLoaded {len(trades)} existing trades")
    else:
        trades = []
        print("\nNo existing trades.json found, creating new one")
    
    # 2. Add the manual sell transaction
    manual_sell = {
        "timestamp": SELL_TIME,
        "type": "sell",
        "amount": btc_sold,
        "price": SELL_PRICE,
        "trade_id": "manual_sell_001",
        "order_id": "manual_sell",
        "fee": btc_sold * SELL_PRICE * 0.0012,  # Approximate fee
        "fee_currency": "USD",
        "source": "manual_fix",
        "note": "Manual sell executed outside of auto-trader"
    }
    
    trades.append(manual_sell)
    
    # 3. Save updated trades.json
    with open(trades_file, 'w') as f:
        json.dump(trades, f, indent=2)
    print(f"\n✅ Added manual sell to trades.json")
    
    # 4. Create proper resume-auto-trade.json
    resume_data = {
        "timestamp": datetime.now().isoformat(),
        "position": "SHORT",
        "amount": USD_HOLDINGS,
        "unit": "usd",
        "entry_price": SELL_PRICE,
        "current_price": SELL_PRICE,  # Will be updated when server starts
        "unrealized_pnl": 0.00,
        "command": f"resume_auto_trade {USD_HOLDINGS:.8f}usd short {SELL_PRICE}",
        "strategy": {
            "type": "MA",
            "short_window": 4,
            "long_window": 20,
            "current_regime": "unknown",
            "active_strategy": "MA"
        },
        "balances": {
            "btc": 0.0,
            "usd": float(USD_HOLDINGS)
        },
        "trades_executed": 1,
        "last_trade_time": SELL_TIME,
        "trade_references": [
            {
                "timestamp": SELL_TIME,
                "type": "sell",
                "amount": btc_sold,
                "price": SELL_PRICE
            }
        ],
        "pivot_protection": {
            "enabled": False,
            "tracker": {}
        }
    }
    
    resume_file = os.path.join(os.path.dirname(__file__), 'resume-auto-trade.json')
    with open(resume_file, 'w') as f:
        json.dump(resume_data, f, indent=2)
    print(f"✅ Created resume-auto-trade.json")
    
    # 5. Display summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Position: SHORT (holding USD)")
    print(f"Entry Price: ${SELL_PRICE:,.2f}")
    print(f"USD Amount: ${USD_HOLDINGS:,.2f}")
    print(f"BTC Sold: {btc_sold:.8f}")
    print("\nFiles updated:")
    print(f"  - trades.json (added manual sell)")
    print(f"  - resume-auto-trade.json (SHORT position)")
    print("\n✅ Ready to start server with correct position!")
    
    # 6. Instructions
    print("\n📝 Next Steps:")
    print("1. Start the server: python src/tdr.py --server")
    print("2. Wait for historical data to load")
    print("3. The system will resume with SHORT position")
    print("4. Will BUY when MA signal turns LONG (MA4 > MA20)")

if __name__ == "__main__":
    main()