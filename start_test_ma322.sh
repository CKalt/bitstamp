#!/bin/bash
# Start test server with MA 3/22 and enhanced logging

cd "$(dirname "$0")"
source source-venv.sh

echo "Starting test server with MA 3/22..."
echo "Configuration: Short=3, Long=22"
echo "Max position: 0.001 BTC"
echo "Comparison logging: ENABLED"

# Start server with enhanced MA strategy
python src/tdr.py --server --port 4002 --strategy enhanced_ma &

echo "Server starting on port 4002..."
echo "Logs: logs/backtest_comparison/"
