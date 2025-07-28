#!/usr/bin/env python3
"""Quick backtest test with proper config"""
import subprocess
import json
import time
from datetime import datetime, timedelta

print("Running quick backtest test...")
print("=" * 60)

# Create a test config
config = {
    "Short_Window": 5,
    "Long_Window": 15,
    "initial_balance": 10000,
    "fee_rate": 0.0012,
    "slippage_rate": 0.0005,
    "enable_pivot_protection": False,
    "enable_adaptive_strategy": False,
    "strategy_type": "MA"
}

with open('test_config.json', 'w') as f:
    json.dump(config, f)

# Test with just 7 days
start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
end_date = datetime.now().strftime("%Y-%m-%d")

cmd = [
    "env/bin/python", "-u", "src/bktst.py",  # -u for unbuffered output
    "--data", "btcusd.log",
    "--config", "test_config.json",
    "--start-date", start_date,
    "--end-date", end_date,
    "--save-results", "test_result.json"
]

print(f"Command: {' '.join(cmd)}")
print(f"Period: {start_date} to {end_date} (7 days)")
print("-" * 60)

start_time = time.time()

# Run with real-time output
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                          text=True, bufsize=1, universal_newlines=True)

# Print output line by line
for line in iter(process.stdout.readline, ''):
    if line:
        print(f"[{time.time() - start_time:6.1f}s] {line.rstrip()}")

process.wait()

elapsed = time.time() - start_time
print(f"\nTotal time: {elapsed:.1f} seconds")

# Clean up
import os
if os.path.exists('test_config.json'):
    os.remove('test_config.json')
if os.path.exists('test_result.json'):
    with open('test_result.json', 'r') as f:
        results = json.load(f)
    print(f"\nResults: Return={results.get('total_return', 0)*100:.2f}%, Trades={results.get('num_trades', 0)}")
    os.remove('test_result.json')