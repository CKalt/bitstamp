#!/usr/bin/env python3
"""
Check system status using HTTP API - no manual commands needed
"""
import requests
import json
from datetime import datetime

def get_full_status():
    """Get comprehensive status via API"""
    server_url = "http://localhost:4000"
    
    print("📊 SYSTEM STATUS CHECK (via API)")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    try:
        # Get main status
        response = requests.get(f"{server_url}/api/status")
        if response.status_code != 200:
            print(f"❌ Failed to get status: {response.status_code}")
            return
        
        status = response.json()
        
        # Extract key information
        print("🔹 POSITION:")
        position_info = status.get('position', {})
        position_dir = "SHORT" if position_info.get('position') == -1 else "LONG" if position_info.get('position') == 1 else "NEUTRAL"
        print(f"  Direction: {position_dir}")
        print(f"  BTC Balance: {position_info.get('btc_balance', 0):.8f}")
        print(f"  USD Balance: ${position_info.get('usd_balance', 0):,.2f}")
        print(f"  Entry Price: ${position_info.get('entry_price', 0):,.2f}")
        
        # Calculate unrealized PnL
        current_price = status.get('last_price', 0)
        if position_info.get('position') == -1 and current_price > 0:
            # SHORT position
            position_size = abs(position_info.get('position_size', 0))
            entry = position_info.get('entry_price', 0)
            unrealized_pnl = position_size * (entry - current_price)
            print(f"  Current Price: ${current_price:,.2f}")
            print(f"  Unrealized PnL: ${unrealized_pnl:,.2f}")
        
        print(f"\n🔹 TRADING STATUS:")
        print(f"  Live Trading: {'ENABLED' if status.get('live_trading') else 'DISABLED'}")
        auto_trader = status.get('auto_trader', {})
        print(f"  Auto-Trader: {'Active' if auto_trader.get('active') else 'Inactive'}")
        print(f"  Strategy: {auto_trader.get('strategy', 'Unknown')}")
        print(f"  Trades Today: {auto_trader.get('trades_today', 0)}")
        
        # Get MA indicator data
        print(f"\n🔹 CHECKING MA INDICATORS...")
        
        # Force a signal evaluation to get latest data
        eval_response = requests.post(f"{server_url}/api/signal/evaluate")
        if eval_response.status_code == 200:
            eval_data = eval_response.json()
            
            print(f"\n🔹 SIGNAL EVALUATION:")
            print(f"  Action: {eval_data.get('action', 'NO_TRADE')}")
            print(f"  Current Signal: {'LONG' if eval_data.get('current_signal') == 1 else 'SHORT' if eval_data.get('current_signal') == -1 else 'NEUTRAL'}")
            print(f"  Current Position: {'LONG' if eval_data.get('current_position') == 1 else 'SHORT' if eval_data.get('current_position') == -1 else 'NEUTRAL'}")
            
            if eval_data.get('evaluation_details'):
                details = eval_data['evaluation_details']
                ma_short = details.get('ma_short', 0)
                ma_long = details.get('ma_long', 0)
                ma_diff = ma_short - ma_long
                
                print(f"\n🔹 MA ANALYSIS:")
                print(f"  MA4:  ${ma_short:,.2f}")
                print(f"  MA20: ${ma_long:,.2f}")
                print(f"  Difference: ${ma_diff:,.2f}")
                
                # Calculate proximity
                if ma_long > 0:
                    proximity = abs(ma_diff / ma_long) * 100
                    print(f"  Proximity: {proximity:.3f}%")
                    
                    # Get threshold from config
                    with open('best_strategy.json', 'r') as f:
                        config = json.load(f)
                    threshold = config.get('ma_separation_threshold', 0.3)
                    
                    print(f"  Threshold: {threshold}%")
                    print(f"  Distance to trigger: {proximity - threshold:.3f}%")
                    
                    if proximity <= threshold:
                        print("  🎯 IN TRIGGER ZONE - Trade should execute!")
                    elif proximity <= threshold + 0.05:
                        print("  ⚠️  VERY CLOSE to trigger!")
                    else:
                        print("  ⏳ Waiting for stronger signal")
            
            if eval_data.get('reason'):
                print(f"\n  Reason: {eval_data['reason']}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    get_full_status()