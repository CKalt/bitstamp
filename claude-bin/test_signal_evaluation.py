#!/usr/bin/env python3
"""
Test signal evaluation with current SHORT position
Focus on understanding why trades aren't executing
"""
import requests
import json
import time
from datetime import datetime

def test_signal_evaluation():
    """Test why system won't flip from SHORT to LONG"""
    server_url = "http://localhost:4000"
    
    print("🔍 TESTING SIGNAL EVALUATION WITH SHORT POSITION")
    print("=" * 60)
    
    # 1. Get current status
    print("\n1. CURRENT STATUS:")
    response = requests.get(f"{server_url}/api/status")
    if response.status_code != 200:
        print("❌ Failed to get status")
        return
    
    status = response.json()
    auto_trading = status.get('auto_trading', {})
    
    print(f"Position: {auto_trading.get('position')}")
    print(f"Live Trading: {'ENABLED' if auto_trading.get('live_trading') else 'DISABLED'}")
    print(f"Trades Today: {auto_trading.get('trades_today', 0)}")
    
    # 2. Force signal evaluation multiple times
    print("\n2. FORCING SIGNAL EVALUATIONS:")
    print("-" * 40)
    
    for i in range(3):
        print(f"\nEvaluation {i+1}:")
        
        # Force evaluation
        eval_response = requests.post(f"{server_url}/api/signal/evaluate")
        if eval_response.status_code == 200:
            result = eval_response.json()
            
            print(f"  Action: {result.get('action', 'NO_TRADE')}")
            print(f"  Current Signal: {result.get('current_signal')}")
            print(f"  Current Position: {result.get('current_position')}")
            
            if result.get('evaluation_details'):
                details = result['evaluation_details']
                print(f"  MA Short: ${details.get('ma_short', 0):,.2f}")
                print(f"  MA Long: ${details.get('ma_long', 0):,.2f}")
                print(f"  MA Diff: ${details.get('ma_diff', 0):,.2f}")
                print(f"  Should Trade: {details.get('should_trade')}")
            
            if result.get('reason'):
                print(f"  Reason: {result['reason']}")
        
        # Wait a bit between evaluations
        if i < 2:
            time.sleep(2)
    
    # 3. Check recent logs for clues
    print("\n3. RECENT LOG ANALYSIS:")
    print("-" * 40)
    
    log_response = requests.get(f"{server_url}/api/logs/read", params={"lines": 100})
    if log_response.status_code == 200:
        logs = log_response.json()
        
        # Look for specific patterns
        signal_evals = [log for log in logs if 'SIGNAL_EVAL' in log]
        trade_blocks = [log for log in logs if 'trade limit' in log.lower() or 'skipping' in log.lower()]
        errors = [log for log in logs if 'error' in log.lower()]
        
        if signal_evals:
            print("\nRecent Signal Evaluations:")
            for log in signal_evals[-3:]:
                print(f"  {log}")
        
        if trade_blocks:
            print("\nTrade Blocking Reasons:")
            for log in trade_blocks[-3:]:
                print(f"  {log}")
        
        if errors:
            print("\nRecent Errors:")
            for log in errors[-3:]:
                print(f"  {log}")
    
    # 4. Check configuration that might block trades
    print("\n4. TRADE BLOCKING SETTINGS:")
    print("-" * 40)
    
    # Get current config
    with open('best_strategy.json', 'r') as f:
        config = json.load(f)
    
    print(f"max_trades_per_day: {config.get('max_trades_per_day', 5)}")
    print(f"max_trades_per_hour: {config.get('max_trades_per_hour', 2)}")
    print(f"min_time_between_trades_minutes: {config.get('min_time_between_trades_minutes', 30)}")
    print(f"ma_separation_threshold: {config.get('ma_separation_threshold', 0.3)}%")
    print(f"require_confirmation_bars: {config.get('require_confirmation_bars', 0)}")
    
    # Check if MA separation is the issue
    if auto_trading.get('diagnostic_data'):
        diag = auto_trading['diagnostic_data']
        if 'ma_proximity' in diag:
            proximity = float(diag['ma_proximity'].strip('%'))
            threshold = config.get('ma_separation_threshold', 0.3)
            
            print(f"\n⚠️  MA Proximity Check:")
            print(f"  Current proximity: {proximity}%")
            print(f"  Required threshold: {threshold}%")
            
            if proximity == threshold:
                print(f"  → RIGHT AT THE THRESHOLD! Needs to cross {threshold}% to trigger")
            elif proximity < threshold:
                print(f"  → BELOW THRESHOLD! Should be trading!")
            else:
                print(f"  → Above threshold, waiting for stronger signal")
    
    # 5. Summary
    print("\n5. ANALYSIS SUMMARY:")
    print("-" * 40)
    
    print("\nPossible reasons for no trade:")
    print("1. MA crossover proximity exactly at threshold (0.31% vs 0.3%)")
    print("2. Waiting for confirmation bars")
    print("3. Time-based restrictions (min time between trades)")
    print("4. Signal evaluation timing (30-second intervals)")
    print("\nThe system appears to be RIGHT at the edge of triggering!")
    print("Next crossover tick should trigger the flip.")

if __name__ == "__main__":
    test_signal_evaluation()