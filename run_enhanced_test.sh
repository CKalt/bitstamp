#!/bin/bash
# Quick test run for enhanced backtesting with minimal parameter combinations

echo "🚀 Running enhanced backtesting in TEST MODE (2 parameter combinations only)..."
echo ""

# Run enhanced backtester with --test-run flag
python src/bktst_enhanced_shared.py \
  --start-window-days-back 120 \
  --end-window-days-back 0 \
  --high-frequency 1H \
  --low-frequency 15T \
  --test-run

# Check if it succeeded
if [ $? -eq 0 ]; then
    echo "✅ Test run complete. Results in recommended_strategy.json"
    
    # Check if recommended_strategy.json exists
    if [ -f "recommended_strategy.json" ]; then
        # Show summary
        echo ""
        echo "📊 Quick Summary:"
        python -c "
import json
with open('recommended_strategy.json', 'r') as f:
    data = json.load(f)
    print(f\"Strategy: {data['optimal_parameters']['strategy']}\")
    print(f\"Total Return: {data['performance_metrics']['total_return_pct']:.2f}%\")
    print(f\"Total Trades: {data['performance_metrics']['total_trades']}\")
    print(f\"Sharpe Ratio: {data['performance_metrics']['sharpe_ratio']:.2f}\")
"
        echo ""
        echo "For full optimization (2304 combinations), run:"
        echo "  ./run_enhanced.sh"
    else
        echo "❌ No recommended_strategy.json created. Check backtesting output."
    fi
else
    echo "❌ Test run failed"
fi