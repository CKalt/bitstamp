#!/bin/bash
# Enhanced backtesting script - Version 2

# Run enhanced backtester v2 (properly tests AdaptiveMultiStrategy)
python src/bktst_enhanced_v2.py \
  --start-window-days-back 120 \
  --end-window-days-back 0 \
  --high-frequency 1H \
  --low-frequency 15T

# Check if it succeeded
if [ $? -eq 0 ]; then
    echo "✅ Backtesting complete. Results in recommended_strategy.json"
    
    # Check if recommended_strategy.json exists
    if [ -f "recommended_strategy.json" ]; then
        # Validate the results
        python src/validate_strategy.py
        
        if [ $? -eq 0 ]; then
            echo "✅ Validation passed"
            
            # Show what will change
            echo "📋 Previewing changes..."
            python src/strategy_migrator.py --dry-run
            
            echo ""
            echo "To apply changes and create best_strategy.json:"
            echo "  python src/strategy_migrator.py"
            echo ""
            echo "Then run trading as usual:"
            echo "  python src/tdr.py"
        else
            echo "❌ Validation failed. Review issues before proceeding."
            echo ""
            echo "Check the following:"
            echo "  - Review recommended_strategy.json"
            echo "  - Check adaptive_strategy_optimization.csv for all tested parameters"
            echo "  - Consider relaxing constraints in config.json"
        fi
    else
        echo "❌ No recommended_strategy.json created. Check backtesting output."
    fi
else
    echo "❌ Backtesting failed"
fi