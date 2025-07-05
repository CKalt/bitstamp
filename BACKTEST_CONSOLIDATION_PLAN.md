# Backtest System Consolidation Plan

## Current State
We have 4 backtest files that evolved over time:
1. `bktst.py` - Original basic backtester
2. `bktst_enhanced.py` - Added AdaptiveMultiStrategy
3. `bktst_enhanced_v2.py` - Standalone simplified version  
4. `bktst_enhanced_shared.py` - Most complete, uses shared strategy core

## Recommendation

**Keep `bktst_enhanced_shared.py` as the primary backtester** because:
- It uses the SAME strategy code as live trading (AdaptiveStrategyCore)
- Most comprehensive and production-ready
- Best parameter handling and configuration
- Most accurate for testing what will actually run in production

## Simplified Usage

Instead of consolidating into one file, create a simple wrapper script:

```bash
#!/bin/bash
# backtest.sh - Simple backtesting wrapper

# Default to the production backtester
BACKTESTER="src/bktst_enhanced_shared.py"

# Pass all arguments through
python $BACKTESTER "$@"
```

## Safe Testing Workflow

1. **Always use custom output files**:
   ```bash
   python src/bktst_enhanced_shared.py --output-file test_strategy.json
   ```

2. **Never overwrite best_strategy.json during testing**:
   ```bash
   # Backup current production config
   cp best_strategy.json best_strategy.production.json
   
   # Test new parameters
   python src/bktst_enhanced_shared.py \
     --start-window-days-back 90 \
     --output-file test_90days.json
   ```

3. **Review results before deploying**:
   ```bash
   # Compare results
   diff best_strategy.json test_90days.json
   
   # Only update when ready
   cp test_90days.json best_strategy.json
   ```

## Quick Reference

### Test Different Time Periods
```bash
# Last 30 days
python src/bktst_enhanced_shared.py --start-window-days-back 30 --output-file test_30d.json

# Last 90 days  
python src/bktst_enhanced_shared.py --start-window-days-back 90 --output-file test_90d.json

# Specific date range (6 months to 1 month ago)
python src/bktst_enhanced_shared.py --start-window-days-back 180 --end-window-days-back 30 --output-file test_historical.json
```

### Test Different Strategies
```bash
# Skip adaptive strategy (test traditional only)
python src/bktst_enhanced_shared.py --skip-adaptive --output-file test_traditional.json

# Different timeframes
python src/bktst_enhanced_shared.py --high-frequency 4H --output-file test_4h.json
python src/bktst_enhanced_shared.py --high-frequency 30T --low-frequency 5T --output-file test_30m.json
```

### Archive Old Backtesters
```bash
# Create archive directory
mkdir -p src/archive/backtest_versions/

# Move old versions
mv src/bktst.py src/archive/backtest_versions/bktst_v1_basic.py
mv src/bktst_enhanced.py src/archive/backtest_versions/bktst_v2_enhanced.py
mv src/bktst_enhanced_v2.py src/archive/backtest_versions/bktst_v3_standalone.py

# Keep only the production version
# src/bktst_enhanced_shared.py -> src/backtest.py
```

## Key Safety Rules

1. **NEVER modify best_strategy.json while testing**
2. **ALWAYS use --output-file for test runs**
3. **BACKUP production config before updates**
4. **TEST changes locally before deploying**
5. **VERIFY server is using expected config**

## Future Enhancement

Consider adding a `--dry-run` flag that:
- Prevents writing to best_strategy.json
- Shows what would change
- Compares with current production config
- Requires explicit confirmation to apply