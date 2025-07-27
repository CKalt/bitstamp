#!/usr/bin/env python3
"""
Quick verification script to check if trading is properly enabled
"""
import json
import requests

# Check best_strategy.json
print("📋 Checking best_strategy.json configuration...")
with open('best_strategy.json', 'r') as f:
    config = json.load(f)

print(f"  Strategy: {config.get('Strategy')}")
print(f"  Strategy Type: {config.get('strategy_type')}")
print(f"  Enable Adaptive: {config.get('enable_adaptive_strategy')}")
print(f"  Live Trading: {config.get('do_live_trades')} {'✅ ENABLED' if config.get('do_live_trades') else '❌ DISABLED'}")
print(f"  MA Windows: {config.get('Short_Window')}/{config.get('Long_Window')}")

# Check server status
print("\n📡 Checking server status...")
try:
    response = requests.get("http://localhost:4000/api/status")
    if response.status_code == 200:
        status = response.json()
        auto_trading = status.get('auto_trading', {})
        
        print(f"  Server: ✅ Running")
        print(f"  Auto-trading: {'✅ Active' if auto_trading.get('active') else '❌ Inactive'}")
        
        if auto_trading.get('active'):
            print(f"  Position: {auto_trading.get('position')}")
            print(f"  Latest Signal: {auto_trading.get('latest_signal')}")
            print(f"  Live Trading Mode: {'✅ LIVE' if auto_trading.get('live_trading') else '❌ DRY-RUN'}")
            
            # Check if server config matches file
            if auto_trading.get('live_trading') != config.get('do_live_trades'):
                print(f"\n⚠️  WARNING: Server live_trading ({auto_trading.get('live_trading')}) doesn't match config file ({config.get('do_live_trades')})")
                print("  You need to restart the server to apply the configuration change!")
        
        # Force evaluation
        print("\n🔄 Forcing signal evaluation...")
        eval_response = requests.post("http://localhost:4000/api/signal/evaluate")
        if eval_response.status_code == 200:
            result = eval_response.json()
            print(f"  Result: {result.get('action', 'NO_TRADE')}")
            if result.get('reason'):
                print(f"  Reason: {result['reason']}")
    else:
        print(f"  Server: ❌ Not responding (status {response.status_code})")
        
except Exception as e:
    print(f"  Server: ❌ Error - {e}")
    print("\nMake sure the server is running with: python src/tdr.py --server")

print("\n💡 Next Steps:")
if not config.get('do_live_trades'):
    print("1. Live trading is DISABLED - trades won't execute")
    print("2. Edit best_strategy.json and set do_live_trades: true")
    print("3. Restart the server to apply changes")
else:
    print("1. Live trading is ENABLED in config ✅")
    print("2. Make sure to restart the server if you just changed this")
    print("3. Monitor logs with: tail -f logs/tdr_server.log")