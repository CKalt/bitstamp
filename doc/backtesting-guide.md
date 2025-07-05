# TDR Backtesting Guide

## Overview

The TDR backtesting system allows you to test trading strategies against historical data before deploying them live. The backtester uses the **same strategy code** as the live trading system, ensuring accurate results.

## Quick Start

```bash
cd /Users/chris/projects/python/btc
source env/bin/activate

# Run default backtest (last 120 days)
python src/backtest.py

# Test last 90 days with custom output
python src/backtest.py --start-window-days-back 90 --output-file my_test.json
```

## Important Safety Rules

⚠️ **NEVER overwrite `best_strategy.json` during testing!** This file controls your live trading.

### Safe Testing Workflow

1. **Always use custom output files**:
```bash
python src/backtest.py --output-file test_strategy.json
```

2. **Use the safe wrapper script** (recommended):
```bash
./backtest --start-window-days-back 90
# Automatically creates timestamped output files
# Warns before overwriting production config
```

3. **Review before deploying**:
```bash
# Compare results
cat test_strategy.json | grep -E "Total_Return|Sharpe_Ratio"

# Backup current config
cp best_strategy.json best_strategy.backup.json

# Apply new config (only when ready!)
cp test_strategy.json best_strategy.json
```

## Command Line Options

### Date Range Options

```bash
# Test specific number of days back from today
--start-window-days-back 90    # Start 90 days ago (default: 120)
--end-window-days-back 0       # End today (default: 0)

# Or specify a trading window
--trading-window-days 60       # Test 60-day window from start date
```

### Strategy Options

```bash
# Skip adaptive strategy testing (faster)
--skip-adaptive

# Test specific timeframes
--high-frequency 1H    # Primary timeframe (default: 1H)
--low-frequency 15T    # Secondary timeframe (default: 15T)
# Options: 5T, 15T, 30T, 1H, 4H, 1D
```

### Output Options

```bash
# Specify output file (REQUIRED for safety!)
--output-file my_strategy.json

# Set optimization iterations
--max-iterations 100   # More iterations = better optimization (default: 50)
```

## Example Scenarios

### 1. Test Recent Market Conditions (30 days)
```bash
python src/backtest.py \
  --start-window-days-back 30 \
  --output-file recent_market_test.json
```

### 2. Test Longer Historical Period (6 months)
```bash
python src/backtest.py \
  --start-window-days-back 180 \
  --output-file historical_test.json
```

### 3. Test Specific Date Range (Skip recent volatile period)
```bash
# Test 90 days starting from 180 days ago, ending 30 days ago
python src/backtest.py \
  --start-window-days-back 180 \
  --end-window-days-back 30 \
  --output-file stable_period_test.json
```

### 4. Quick Test Without Adaptive Strategy
```bash
python src/backtest.py \
  --skip-adaptive \
  --max-iterations 20 \
  --output-file quick_test.json
```

### 5. Test Different Timeframes
```bash
# 4-hour candles
python src/backtest.py \
  --high-frequency 4H \
  --output-file conservative_4h.json

# 30-minute candles with 5-minute secondary
python src/backtest.py \
  --high-frequency 30T \
  --low-frequency 5T \
  --output-file aggressive_30m.json
```

## Understanding Results

### Output Files Created

1. **Main output file** (specified by --output-file):
   - Complete strategy configuration
   - Performance metrics
   - Ready to use as best_strategy.json

2. **Additional files** (automatically created):
   - `optimization_results.csv` - All parameter combinations tested
   - `strategy_comparison_enhanced.csv` - Comparison of strategies
   - `all_strategy_results_enhanced.csv` - Detailed results
   - `adaptive_strategy_detailed_results.json` - Regime performance breakdown

### Key Metrics to Evaluate

```json
{
  "Total_Return": 5.38,        // 538% gain (higher is better)
  "Sharpe_Ratio": 0.097,       // Risk-adjusted returns (> 0 is good, > 1 is excellent)
  "Total_Trades": 135,         // Number of trades
  "Average_Trades_Per_Day": 1.1, // Trading frequency
  "Profit_Factor": 1.019,      // Wins/Losses ratio (> 1 is profitable)
  "Max_Drawdown": -15.2        // Largest peak-to-trough loss %
}
```

### Adaptive Strategy Results

When testing adaptive strategies, review regime performance:

```json
"regime_performance": {
  "trending": {
    "trades": 89,
    "success_rate": 0.64,
    "avg_profit": 1.2
  },
  "ranging": {
    "trades": 32,
    "success_rate": 0.71,
    "avg_profit": 0.8
  },
  "volatile": {
    "trades": 14,
    "success_rate": 0.43,
    "avg_profit": -0.3
  }
}
```

## Comparing Multiple Tests

### Run Multiple Backtests
```bash
# Conservative test
python src/backtest.py \
  --start-window-days-back 180 \
  --output-file conservative.json

# Moderate test  
python src/backtest.py \
  --start-window-days-back 90 \
  --output-file moderate.json

# Aggressive test
python src/backtest.py \
  --start-window-days-back 30 \
  --high-frequency 30T \
  --output-file aggressive.json
```

### Compare Results
```bash
# Quick comparison
for f in conservative.json moderate.json aggressive.json; do
  echo "=== $f ==="
  cat $f | grep -E "Total_Return|Sharpe_Ratio|Total_Trades"
done

# Detailed comparison
diff conservative.json moderate.json
```

## Applying Backtest Results

### When to Update Production Config

Consider updating `best_strategy.json` when:
- ✅ Total Return > current strategy
- ✅ Sharpe Ratio > 0 (positive risk-adjusted returns)
- ✅ Reasonable trade frequency (0.5-5 trades/day)
- ✅ Tested on sufficient data (> 30 days)
- ✅ Lower max drawdown than current

### Deployment Process

1. **Stop trading** (on server):
```bash
tdr> stop_auto_trade
```

2. **Backup current config** (locally):
```bash
cp best_strategy.json best_strategy.$(date +%Y%m%d).json
```

3. **Apply new config**:
```bash
cp tested_strategy.json best_strategy.json
```

4. **Restart server** to load new config

5. **Resume trading**:
```bash
tdr> resume_auto_trade [amount] [position] [entry_price]
```

## Advanced Testing

### Custom Parameter Ranges

Edit the parameter ranges in `src/backtest.py`:

```python
# MA strategy parameters
short_range = range(5, 25, 5)    # Test 5, 10, 15, 20
long_range = range(20, 60, 10)   # Test 20, 30, 40, 50

# Adaptive parameters
regime_thresholds = [0.3, 0.4, 0.5, 0.6]
confirmation_bars = [1, 2, 3]
```

### Testing Specific Market Conditions

```bash
# Bull market test (if you know dates)
python src/backtest.py \
  --start-window-days-back 365 \
  --end-window-days-back 180 \
  --output-file bull_market.json

# Recent volatility
python src/backtest.py \
  --start-window-days-back 14 \
  --high-frequency 15T \
  --output-file volatile_period.json
```

## Troubleshooting

### Common Issues

1. **"No data found in the specified date range"**
   - Check that btcusd.log exists
   - Verify date range is within available data
   - Try `ls -la btcusd.log` to check file size

2. **"MemoryError" with large date ranges**
   - Reduce date range
   - Use higher frequency (4H instead of 1H)
   - Add `--skip-adaptive` for faster testing

3. **Results seem unrealistic**
   - Check for data quality issues
   - Verify fee and slippage settings
   - Review trade frequency constraints

### Getting Help

Check logs for detailed error messages:
```bash
# Recent errors
tail -50 logs/backtest.log

# Search for specific errors
grep ERROR logs/backtest.log
```

## Best Practices

1. **Test multiple time periods** to avoid overfitting
2. **Start conservative** - test longer periods first
3. **Document your tests** - keep notes on what you tested and why
4. **Never rush deployment** - a bad strategy can lose money quickly
5. **Monitor after deployment** - new strategies need extra attention

## Summary Checklist

- [ ] Always use `--output-file` to avoid overwriting production config
- [ ] Test sufficient historical data (minimum 30 days)
- [ ] Review all performance metrics, not just returns
- [ ] Backup current config before applying changes
- [ ] Stop trading before updating configuration
- [ ] Monitor closely after deploying new strategy

Remember: Backtesting shows historical performance. Future results may differ!