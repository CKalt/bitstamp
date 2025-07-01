#!/bin/bash
# Interactive enhanced backtesting script

echo "🎯 Starting Interactive Enhanced Backtesting..."
echo ""

# Run enhanced backtester with interactive prompts
python src/bktst_enhanced_shared.py \
  --start-window-days-back 120 \
  --end-window-days-back 0 \
  --high-frequency 1H \
  --low-frequency 15T \
  --user-prompts

# Check if it succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Backtesting complete. Results in recommended_strategy.json"
    
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
        
        # Validate the results
        echo ""
        echo "📋 Validating strategy..."
        python src/validate_strategy.py
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Validation passed!"
            echo ""
            echo "Next steps:"
            echo "1. Review: recommended_strategy.json"
            echo "2. Compare: adaptive_strategy_optimization.csv"
            echo "3. Deploy: python src/strategy_migrator.py"
        else
            echo ""
            echo "⚠️  Validation warnings detected. Review before deploying."
        fi
    fi
else
    echo "❌ Backtesting failed"
fi