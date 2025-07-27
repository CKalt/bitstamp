#!/usr/bin/env python3
"""
Simple script to check current signal evaluation and diagnose why trades aren't happening
"""
import requests
import json
import sys
from datetime import datetime

def check_signal_evaluation(server_url="http://localhost:4000"):
    """Check current signal evaluation status"""
    print("\n🔍 SIGNAL EVALUATION DIAGNOSTIC")
    print("="*60)
    
    try:
        # Get current status
        response = requests.get(f"{server_url}/api/status")
        if response.status_code != 200:
            print(f"❌ Failed to get status: {response.status_code}")
            return
        
        status = response.json()
        
        # Extract auto-trading info
        auto_trading = status.get('auto_trading', {})
        if not auto_trading.get('active'):
            print("❌ Auto-trading is not active!")
            print("Run: curl -X POST http://localhost:4000/api/autotrade/start")
            return
        
        print(f"✅ Auto-trading is active")
        print(f"\nCurrent Status:")
        print(f"  Position: {auto_trading.get('position', 'UNKNOWN')}")
        print(f"  Latest Signal: {auto_trading.get('latest_signal', 'UNKNOWN')}")
        print(f"  Amount: {auto_trading.get('amount', 0)} BTC")
        print(f"  Entry Price: ${auto_trading.get('entry_price', 0):,.2f}")
        
        # Get MA values
        indicators = status.get('indicators', {}).get('btcusd', {})
        ma_short = indicators.get(f"MA_{auto_trading.get('short_window', 4)}")
        ma_long = indicators.get(f"MA_{auto_trading.get('long_window', 20)}")
        
        if ma_short and ma_long:
            ma_diff = ma_short - ma_long
            print(f"\nMA Analysis:")
            print(f"  MA{auto_trading.get('short_window', 4)}: ${ma_short:,.2f}")
            print(f"  MA{auto_trading.get('long_window', 20)}: ${ma_long:,.2f}")
            print(f"  Difference: ${ma_diff:,.2f}")
            print(f"  Signal: {'LONG' if ma_diff > 0 else 'SHORT'}")
        
        # Check diagnostic data
        diagnostic = auto_trading.get('diagnostic_data', {})
        if diagnostic:
            print(f"\nDiagnostic Data:")
            print(f"  Last Evaluation: {diagnostic.get('timestamp', 'N/A')}")
            print(f"  Trade Status: {diagnostic.get('trade_status', 'N/A')}")
            if 'reason' in diagnostic:
                print(f"  Reason: {diagnostic['reason']}")
        
        # Force evaluation
        print("\n📡 Forcing signal evaluation...")
        eval_response = requests.post(f"{server_url}/api/signal/evaluate")
        if eval_response.status_code == 200:
            eval_result = eval_response.json()
            print(f"✅ Evaluation complete:")
            print(f"  Action: {eval_result.get('action', 'NO_TRADE')}")
            print(f"  Current Signal: {eval_result.get('current_signal')}")
            print(f"  Current Position: {eval_result.get('current_position')}")
            if eval_result.get('reason'):
                print(f"  Reason: {eval_result['reason']}")
            
            # Show evaluation details
            if 'evaluation_details' in eval_result:
                details = eval_result['evaluation_details']
                print(f"\nEvaluation Details:")
                print(f"  MA Short: ${details.get('ma_short', 0):,.2f}")
                print(f"  MA Long: ${details.get('ma_long', 0):,.2f}")
                print(f"  MA Diff: ${details.get('ma_diff', 0):,.2f}")
                print(f"  Position Match: {details.get('position_matches_signal')}")
                print(f"  Should Trade: {details.get('should_trade')}")
        else:
            print(f"❌ Evaluation failed: {eval_response.text}")
        
        # Check logs
        print("\n📋 Recent Log Entries:")
        log_response = requests.get(f"{server_url}/api/logs/read", params={"lines": 50})
        if log_response.status_code == 200:
            logs = log_response.json()
            signal_logs = [log for log in logs if 'SIGNAL_EVAL' in log or 'trade' in log.lower()]
            for log in signal_logs[-5:]:  # Last 5 relevant logs
                print(f"  {log}")
        
        # Recommendations
        print("\n💡 Troubleshooting Steps:")
        print("1. Check if MA values are being calculated correctly")
        print("2. Verify position tracking is accurate")
        print("3. Look for any error messages in logs")
        print("4. Ensure historical data is loaded properly")
        print("5. Check if trade execution is enabled (do_live_trades)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure the server is running at", server_url)


if __name__ == "__main__":
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:4000"
    check_signal_evaluation(server_url)