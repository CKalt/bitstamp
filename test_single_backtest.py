#!/usr/bin/env python3
"""Quick test of a single backtest to see output"""
import subprocess
import time
from datetime import datetime, timedelta

print("Testing single backtest to diagnose output issue...")
print("=" * 60)

# Test with MA 5/15 (should be faster)
start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")  # Only 7 days
end_date = datetime.now().strftime("%Y-%m-%d")

cmd = [
    "env/bin/python", "src/bktst.py",
    "--data", "btcusd.log",
    "--start-date", start_date,
    "--end-date", end_date,
    "--ma-short", "5",
    "--ma-long", "15",
    "--initial", "10000"
]

print(f"Running: {' '.join(cmd)}")
print(f"Testing only 7 days of data for speed")
print("-" * 60)

start_time = time.time()

# Run without capturing output - show directly
subprocess.run(cmd)

elapsed = time.time() - start_time
print(f"\nCompleted in {elapsed:.1f} seconds")