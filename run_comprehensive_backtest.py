#!/usr/bin/env python3
"""
Comprehensive backtest using proper parameters
Based on the working run.sh script
"""

import subprocess
import json
import os
from datetime import datetime

print("Running comprehensive backtest analysis...")
print("Using 120 days of data with proper frequency settings")
print("=" * 60)

# Test configurations - different MA combinations
test_configs = [
    # Current settings
    {"short": 10, "long": 46, "name": "current_10_46"},
    
    # Faster signals (might catch trends earlier)
    {"short": 5, "long": 15, "name": "very_fast_5_15"},
    {"short": 8, "long": 21, "name": "fibonacci_8_21"},
    {"short": 9, "long": 26, "name": "modified_macd_9_26"},
    {"short": 12, "long": 26, "name": "standard_macd_12_26"},
    
    # Medium speed
    {"short": 15, "long": 30, "name": "medium_15_30"},
    {"short": 20, "long": 50, "name": "classic_20_50"},
    {"short": 21, "long": 55, "name": "fibonacci_21_55"},
    
    # Slower but potentially more reliable
    {"short": 30, "long": 90, "name": "quarterly_30_90"},
    {"short": 50, "long": 100, "name": "slow_50_100"},
    {"short": 50, "long": 200, "name": "golden_cross_50_200"},
]

# Base command with your working parameters
base_cmd = [
    "python", "src/bktst.py",
    "--start-window-days-back", "120",
    "--end-window-days-back", "0", 
    "--high-frequency", "1H",
    "--low-frequency", "15T",
    "--initial", "10000"
]

results = []
best_return = -999
best_config = None

for config in test_configs:
    print(f"\nTesting {config['name']}: MA({config['short']}, {config['long']})")
    print("-" * 40)
    
    # Build full command
    cmd = base_cmd + [
        "--short", str(config['short']),
        "--long", str(config['long'])
    ]
    
    # Run backtest
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        
        # Parse output for key metrics
        total_return = 0
        final_balance = 10000
        num_trades = 0
        win_rate = 0
        sharpe_ratio = 0
        max_drawdown = 0
        
        for line in output.split('\n'):
            if "Total Return:" in line and "%" in line:
                try:
                    total_return = float(line.split(":")[1].split("%")[0].strip())
                except:
                    pass
            elif "Final Balance:" in line and "$" in line:
                try:
                    final_balance = float(line.split("$")[1].replace(",", "").strip())
                except:
                    pass
            elif "Total Trades:" in line:
                try:
                    num_trades = int(line.split(":")[1].strip())
                except:
                    pass
            elif "Win Rate:" in line and "%" in line:
                try:
                    win_rate = float(line.split(":")[1].split("%")[0].strip())
                except:
                    pass
            elif "Sharpe Ratio:" in line:
                try:
                    sharpe_ratio = float(line.split(":")[1].strip())
                except:
                    pass
            elif "Max Drawdown:" in line and "%" in line:
                try:
                    max_drawdown = float(line.split(":")[1].split("%")[0].strip())
                except:
                    pass
        
        config_result = {
            "config": config,
            "total_return": total_return,
            "final_balance": final_balance,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "profit": final_balance - 10000
        }
        
        results.append(config_result)
        
        print(f"  Return: {total_return:.2f}%")
        print(f"  Final Balance: ${final_balance:,.2f}")
        print(f"  Profit/Loss: ${final_balance - 10000:,.2f}")
        print(f"  Trades: {num_trades}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        
        if total_return > best_return:
            best_return = total_return
            best_config = config_result
            
    except subprocess.TimeoutExpired:
        print("  ERROR: Backtest timed out")
        continue
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        continue

# Sort results by return
results.sort(key=lambda x: x['total_return'], reverse=True)

# Save detailed results
output_file = "comprehensive_backtest_results.json"
with open(output_file, 'w') as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "test_period": "120 days",
        "initial_balance": 10000,
        "results": results,
        "best_config": best_config
    }, f, indent=2)

print("\n" + "=" * 60)
print("BACKTEST COMPLETE - TOP 3 STRATEGIES:")
print("=" * 60)

for i, result in enumerate(results[:3]):
    print(f"\n{i+1}. {result['config']['name']} - MA({result['config']['short']}, {result['config']['long']})")
    print(f"   Return: {result['total_return']:.2f}%")
    print(f"   Profit: ${result['profit']:,.2f}")
    print(f"   Trades: {result['num_trades']}")
    print(f"   Win Rate: {result['win_rate']:.1f}%")
    print(f"   Sharpe: {result['sharpe_ratio']:.2f}")

if best_config and best_config['total_return'] > 0:
    print("\n" + "=" * 60)
    print("RECOMMENDED ACTION:")
    print("=" * 60)
    print(f"Update best_strategy.json with:")
    print(f"  Short_Window: {best_config['config']['short']}")
    print(f"  Long_Window: {best_config['config']['long']}")
    print(f"\nThis could improve returns from 0% to {best_config['total_return']:.2f}%")
else:
    print("\n⚠️  WARNING: No profitable strategy found in backtest!")
    print("This might indicate:")
    print("1. Difficult market conditions in the test period")
    print("2. MA crossover strategies not suitable for current volatility")
    print("3. Need for different strategy types (momentum, mean reversion, etc.)")

print(f"\nDetailed results saved to: {output_file}")