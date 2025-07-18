#!/usr/bin/env python3
"""
Simple backtest runner to test with the current data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bktst import EnhancedBacktester, main
import json
import pandas as pd
from datetime import datetime, timedelta

# Load configuration
config_path = 'best_strategy.json'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    print(f"Error: {config_path} not found")
    sys.exit(1)

# Add defaults if missing
config.setdefault('initial_balance', 10000)
config.setdefault('fee_rate', 0.0012)
config.setdefault('slippage_rate', 0.0005)

print("Running backtest with last 30 days of data...")
print(f"Configuration: {config_path}")

# Use the main function with modified arguments
import sys
sys.argv = [
    'run_simple_backtest.py',
    '--data', 'btcusd.log',
    '--config', config_path,
    '--save-results', 'backtest_results_30days.json'
]

# Calculate date 30 days ago
end_date = datetime(2025, 6, 30)  # Based on the log showing last date
start_date = end_date - timedelta(days=30)

sys.argv.extend(['--start-date', start_date.strftime('%Y-%m-%d')])
sys.argv.extend(['--end-date', end_date.strftime('%Y-%m-%d')])

# Run the backtest
try:
    main()
except Exception as e:
    print(f"Error during backtest: {e}")
    import traceback
    traceback.print_exc()