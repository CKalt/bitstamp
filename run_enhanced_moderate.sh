#!/bin/bash
# Moderate optimization for enhanced backtesting (216 combinations, ~30-60 minutes)

echo "🎯 Running enhanced backtesting with MODERATE optimization..."
echo "This will test 216 parameter combinations (30-60 minutes)"
echo ""

# Run enhanced backtester with moderate preset
python src/bktst_enhanced_shared.py \
  --start-window-days-back 60 \
  --end-window-days-back 0 \
  --high-frequency 1H \
  --low-frequency 15T \
  --optimization-preset moderate

# Check if it succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Moderate optimization complete!"
    
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
    
    # Show adaptive strategy details if available
    if data['optimal_parameters']['strategy'] == 'AdaptiveMulti':
        print(f\"\\nAdaptive Strategy Parameters:\")
        print(f\"  Short Window: {data['optimal_parameters'].get('short_window', 'N/A')}\")
        print(f\"  Long Window: {data['optimal_parameters'].get('long_window', 'N/A')}\")
        print(f\"  Regime Threshold: {data['optimal_parameters'].get('regime_switch_threshold', 'N/A')}\")
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
            echo "1. Review detailed results in adaptive_strategy_optimization.csv"
            echo "2. For more thorough analysis, run: ./run_enhanced.sh"
            echo "3. To deploy: python src/strategy_migrator.py"
        fi
    else
        echo "❌ No recommended_strategy.json created. Check backtesting output."
    fi
else
    echo "❌ Optimization failed"
fi