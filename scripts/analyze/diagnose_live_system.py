#!/usr/bin/env python3
"""
Diagnostic script to analyze why the live system isn't trading
"""
import json
import os
import pandas as pd
from datetime import datetime
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("🔍 LIVE SYSTEM DIAGNOSTIC REPORT")
print("=" * 60)

# 1. Check configuration files
print("\n1. CONFIGURATION FILES:")
print("-" * 40)

# Check best_strategy.json
if os.path.exists('best_strategy.json'):
    with open('best_strategy.json', 'r') as f:
        config = json.load(f)
    
    print("📋 best_strategy.json:")
    print(f"  - do_live_trades: {config.get('do_live_trades')} {'✅' if config.get('do_live_trades') else '❌'}")
    print(f"  - auto_resume: {config.get('auto_resume')} {'✅' if config.get('auto_resume') else '❌ PROBLEM!'}")
    print(f"  - strategy_type: {config.get('strategy_type')}")
    print(f"  - enable_adaptive_strategy: {config.get('enable_adaptive_strategy')}")
    print(f"  - MA windows: {config.get('Short_Window')}/{config.get('Long_Window')}")
    print(f"  - Last signal: {config.get('Last_Signal_Action')} at {config.get('Last_Signal_Timestamp')}")
else:
    print("❌ best_strategy.json not found!")

# 2. Check resume file
print("\n2. POSITION TRACKING:")
print("-" * 40)

resume_file = 'resume-auto-trade.json'
if os.path.exists(resume_file):
    with open(resume_file, 'r') as f:
        resume_data = json.load(f)
    
    print(f"📋 resume-auto-trade.json:")
    print(f"  - Position: {resume_data.get('position')}")
    print(f"  - Amount: {resume_data.get('amount')} BTC")
    print(f"  - Entry Price: ${resume_data.get('entry_price'):,.2f}")
    print(f"  - Entry Time: {resume_data.get('entry_time')}")
    
    # Check if auto_resume is false
    if not config.get('auto_resume'):
        print("\n  ⚠️  WARNING: auto_resume is FALSE in config!")
        print("  This means the server won't load this position on startup!")
        print("  The system doesn't know it's SHORT and needs to flip to LONG!")
else:
    print("❌ resume-auto-trade.json not found!")
    print("  System has no position history to load")

# 3. Check recent trades
print("\n3. RECENT TRADES:")
print("-" * 40)

if os.path.exists('trades.json'):
    with open('trades.json', 'r') as f:
        trades_data = json.load(f)
    
    # Handle both formats
    if isinstance(trades_data, dict) and 'trades' in trades_data:
        trades = trades_data['trades']
    else:
        trades = trades_data
    
    if trades:
        last_trade = trades[-1]
        print(f"  Last trade: {last_trade['type']} {last_trade['amount']} BTC @ ${last_trade['price']:,.2f}")
        print(f"  Time: {last_trade['timestamp']}")
        
        # Count recent trades
        from datetime import datetime, timedelta
        now = datetime.now()
        today_trades = [t for t in trades if datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00')).date() == now.date()]
        print(f"  Trades today: {len(today_trades)}")
else:
    print("❌ trades.json not found!")

# 4. Calculate current MA values
print("\n4. CURRENT MA ANALYSIS:")
print("-" * 40)

try:
    from data.loader import parse_log_file
    from indicators.technical_indicators import add_moving_averages, generate_ma_signals
    
    # Load recent data
    df = parse_log_file('btcusd.log')
    if not df.empty:
        # Prepare data
        df.rename(columns={'price': 'close'}, inplace=True)
        
        # Calculate MAs with config windows
        short_window = config.get('Short_Window', 4)
        long_window = config.get('Long_Window', 20)
        
        df_ma = add_moving_averages(df.copy(), short_window, long_window, price_col='close')
        df_ma = generate_ma_signals(df_ma)
        
        # Get latest values
        if not df_ma.empty and len(df_ma) >= long_window:
            latest = df_ma.iloc[-1]
            current_price = latest['close']
            ma_short = latest['Short_MA']
            ma_long = latest['Long_MA']
            signal = latest['MA_Signal']
            
            print(f"  Current Price: ${current_price:,.2f}")
            print(f"  MA{short_window}: ${ma_short:,.2f}")
            print(f"  MA{long_window}: ${ma_long:,.2f}")
            print(f"  Difference: ${ma_short - ma_long:,.2f}")
            print(f"  Signal: {'LONG' if signal == 1 else 'SHORT' if signal == -1 else 'NEUTRAL'}")
            
            # Check if signal matches position
            if os.path.exists(resume_file) and resume_data.get('position'):
                current_pos = 1 if resume_data['position'] == 'LONG' else -1
                if current_pos != signal and signal != 0:
                    print(f"\n  🚨 SIGNAL MISMATCH DETECTED!")
                    print(f"  Current position: {resume_data['position']} ({current_pos})")
                    print(f"  MA Signal says: {'LONG' if signal == 1 else 'SHORT'} ({signal})")
                    print(f"  → System should flip position!")
                else:
                    print(f"\n  ✅ Position matches signal")
        else:
            print("  ❌ Not enough data for MA calculation")
    else:
        print("  ❌ No data in btcusd.log")
        
except Exception as e:
    print(f"  ❌ Error calculating MAs: {e}")

# 5. Diagnosis Summary
print("\n5. DIAGNOSIS SUMMARY:")
print("-" * 40)

problems = []

if not config.get('auto_resume'):
    problems.append("auto_resume is FALSE - system won't load position on startup")

if not os.path.exists(resume_file):
    problems.append("No resume-auto-trade.json - system has no position history")

if config.get('enable_adaptive_strategy'):
    problems.append("Adaptive strategy enabled when it should use pure MA")

if problems:
    print("🚨 PROBLEMS FOUND:")
    for i, problem in enumerate(problems, 1):
        print(f"  {i}. {problem}")
else:
    print("✅ No configuration problems found")

# 6. Recommended Actions
print("\n6. RECOMMENDED ACTIONS:")
print("-" * 40)

print("For testing with dry-run mode:")
print("1. Copy test_config_dry_run.json to best_strategy.json")
print("2. This will set:")
print("   - do_live_trades: false (dry-run mode)")
print("   - auto_resume: true (loads position correctly)")
print("   - log_signal_evaluation: true (detailed logging)")
print("   - verbose_logging: true (see what's happening)")
print("3. Restart server to test signal evaluation")
print("4. Monitor logs to see if trades would execute")
print("\nFor fixing live system:")
print("1. Set auto_resume: true in best_strategy.json")
print("2. Restart server to apply changes")
print("3. System will then know its position and can trade")

print("\n" + "=" * 60)