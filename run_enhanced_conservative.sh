#!/bin/bash
# Conservative optimization for enhanced backtesting (64 combinations, ~5-10 minutes)

echo "⚡ Running enhanced backtesting with CONSERVATIVE optimization..."
echo "This will test 64 parameter combinations (5-10 minutes)"
echo ""

# Run enhanced backtester with conservative preset
python src/bktst_enhanced_shared.py \
  --start-window-days-back 30 \
  --end-window-days-back 0 \
  --high-frequency 1H \
  --low-frequency 15T \
  --optimization-preset conservative

# Check if it succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Conservative optimization complete!"
    
    # Check if recommended_strategy.json exists
    if [ -f "recommended_strategy.json" ]; then
        # Show summary
        echo ""
        echo "📊 Results Summary:"
        python -c "
import json
with open('recommended_strategy.json', 'r') as f:
    data = json.load(f)
    print(f\"Strategy: {data['optimal_parameters']['strategy']}\")
    print(f\"Total Return: {data['performance_metrics']['total_return_pct']:.2f}%\")
    print(f\"Total Trades: {data['performance_metrics']['total_trades']}\")
    print(f\"Sharpe Ratio: {data['performance_metrics']['sharpe_ratio']:.2f}\")
    print(f\"Max Drawdown: {data['performance_metrics']['max_drawdown_pct']:.2f}%\")
    print(f\"Win Rate: {data['performance_metrics']['win_rate']:.2f}%\")
"
        echo ""
        echo "This was a quick optimization. For better results:"
        echo "  - Moderate (30-60 min): ./run_enhanced_moderate.sh"
        echo "  - Full (2-4 hours): ./run_enhanced.sh"
    else
        echo "❌ No recommended_strategy.json created. Check backtesting output."
    fi
else
    echo "❌ Optimization failed"
fi