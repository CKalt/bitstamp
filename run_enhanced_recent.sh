#!/bin/bash
# Enhanced backtesting script for recent data only

echo "Running enhanced backtester on recent data (last 30 days)..."

# First use relaxed config
if [ -f "config_relaxed.json" ]; then
    echo "Using relaxed configuration..."
    cp config_relaxed.json config.json
fi

# Run enhanced backtester with just last 30 days
python src/bktst_enhanced.py \
  --start-window-days-back 30 \
  --end-window-days-back 0 \
  --high-frequency 1H \
  --low-frequency 15T

# Check if it succeeded
if [ $? -eq 0 ]; then
    echo "✅ Backtesting complete. Results in recommended_strategy.json"
    
    # Run diagnostics to understand results
    echo ""
    echo "Running diagnostics..."
    python diagnose_backtesting.py
    
    # Check if recommended_strategy.json exists
    if [ -f "recommended_strategy.json" ]; then
        # Show summary
        echo ""
        echo "Summary of results:"
        python -c "
import json
with open('recommended_strategy.json', 'r') as f:
    data = json.load(f)
    print(f\"Strategy: {data['optimal_parameters']['strategy']}\")
    print(f\"Return: {data['performance_metrics']['total_return_pct']:.2f}%\")
    print(f\"Trades: {data['performance_metrics']['total_trades']}\")
    print(f\"Ready: {data['validation_status']['ready_for_deployment']}\")
"
    fi
else
    echo "❌ Backtesting failed"
fi