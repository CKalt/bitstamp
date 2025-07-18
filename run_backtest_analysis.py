#!/usr/bin/env python3
"""
Backtest analysis to find profitable trading parameters
Saves results to JSON for analysis
"""

import json
import sys
import os
from datetime import datetime, timedelta

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Run the backtest
print("Running backtest analysis on recent data...")
print("This will test different MA window combinations to find profitable settings")
print("-" * 60)

# Test parameters
test_configs = [
    # Current settings
    {"short": 10, "long": 46, "name": "current"},
    
    # Faster signals
    {"short": 5, "long": 20, "name": "fast"},
    {"short": 8, "long": 21, "name": "8_21"},
    {"short": 12, "long": 26, "name": "macd_std"},
    
    # Medium speed
    {"short": 20, "long": 50, "name": "medium"},
    {"short": 15, "long": 30, "name": "15_30"},
    
    # Slower but potentially more reliable
    {"short": 50, "long": 200, "name": "slow"},
    {"short": 20, "long": 100, "name": "20_100"},
]

results = []
best_return = -999
best_config = None

for config in test_configs:
    print(f"\nTesting {config['name']}: MA({config['short']}, {config['long']})")
    
    # Build command
    cmd = f"python src/bktst.py --days 7 --short {config['short']} --long {config['long']} --initial 10000"
    
    # Run backtest
    import subprocess
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    
    # Parse output for key metrics
    output = result.stdout
    
    # Extract metrics
    total_return = 0
    num_trades = 0
    win_rate = 0
    
    for line in output.split('\n'):
        if "Total Return:" in line:
            try:
                total_return = float(line.split(":")[1].strip().replace('%', ''))
            except:
                pass
        elif "Number of Trades:" in line:
            try:
                num_trades = int(line.split(":")[1].strip())
            except:
                pass
        elif "Win Rate:" in line:
            try:
                win_rate = float(line.split(":")[1].strip().replace('%', ''))
            except:
                pass
    
    config_result = {
        "config": config,
        "total_return": total_return,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "output": output[-1000:]  # Last 1000 chars
    }
    
    results.append(config_result)
    
    print(f"  Return: {total_return:.2f}%")
    print(f"  Trades: {num_trades}")
    print(f"  Win Rate: {win_rate:.1f}%")
    
    if total_return > best_return:
        best_return = total_return
        best_config = config

# Save results
output_file = "backtest_results.json"
with open(output_file, 'w') as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "best_config": best_config,
        "best_return": best_return
    }, f, indent=2)

print("\n" + "=" * 60)
print(f"BEST CONFIGURATION: {best_config['name']}")
print(f"MA({best_config['short']}, {best_config['long']})")
print(f"Return: {best_return:.2f}%")
print("=" * 60)
print(f"\nDetailed results saved to: {output_file}")
print("\nTo apply the best settings:")
print(f"1. Edit best_strategy.json")
print(f"2. Set Short_Window: {best_config['short']}")
print(f"3. Set Long_Window: {best_config['long']}")
print(f"4. Restart auto trader")