#!/usr/bin/env python
"""Quick MA backtest for recent data"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tst', 'src'))

from datetime import datetime, timedelta
import json

# Test configurations
ma_configs = [
    {"short": 4, "long": 20, "name": "MA 4/20"},
    {"short": 6, "long": 34, "name": "MA 6/34"}, 
    {"short": 8, "long": 40, "name": "MA 8/40"},
    {"short": 10, "long": 46, "name": "MA 10/46"},
    {"short": 12, "long": 48, "name": "MA 12/48"},
    {"short": 5, "long": 25, "name": "MA 5/25"},
    {"short": 15, "long": 60, "name": "MA 15/60"},
]

print("Quick MA Backtest - Last 30 days")
print("=" * 50)

# Load the current best_strategy.json as template
with open("best_strategy.json", "r") as f:
    template = json.load(f)

best_return = -999
best_config = None

for config in ma_configs:
    print(f"\nTesting {config['name']}...")
    
    # Create test config
    test_config = template.copy()
    test_config["Short_Window"] = config["short"]
    test_config["Long_Window"] = config["long"]
    
    # Write temp config
    with open("temp_test_config.json", "w") as f:
        json.dump(test_config, f)
    
    # Run backtest (using the existing bktst.py)
    import subprocess
    result = subprocess.run([
        sys.executable, "src/bktst.py",
        "--data", "btcusd.log",
        "--config", "temp_test_config.json",
        "--start-date", (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        "--end-date", datetime.now().strftime("%Y-%m-%d")
    ], capture_output=True, text=True)
    
    # Parse results from output
    for line in result.stdout.split('\n'):
        if "Total Return:" in line:
            try:
                return_pct = float(line.split(':')[1].strip().replace('%', ''))
                print(f"  Return: {return_pct:.2f}%")
                
                if return_pct > best_return:
                    best_return = return_pct
                    best_config = config
            except:
                pass

print("\n" + "=" * 50)
print(f"BEST STRATEGY: {best_config['name']}")
print(f"Return: {best_return:.2f}%")
print("=" * 50)

# Save best config
if best_config:
    best_strategy = template.copy()
    best_strategy["Short_Window"] = best_config["short"]
    best_strategy["Long_Window"] = best_config["long"]
    best_strategy["_comment"] = f"Optimal from 30-day backtest: {best_config['name']} with {best_return:.2f}% return"
    
    with open("best_strategy_optimized.json", "w") as f:
        json.dump(best_strategy, f, indent=2)
    
    print(f"\nSaved to best_strategy_optimized.json")

# Cleanup
if os.path.exists("temp_test_config.json"):
    os.remove("temp_test_config.json")