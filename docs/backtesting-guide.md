# TDR Backtesting Guide

## Overview

The TDR backtesting system tests trading strategies against historical Bitcoin price data. The backtester uses the **same strategy code** as the live trading system, ensuring accurate results.

**Current Implementation**: `src/backtest.py` (uses shared AdaptiveStrategyCore)

## Quick Start

```bash
cd /Users/chris/projects/python/btc
source env/bin/activate

# Run default backtest (last 120 days)
python src/backtest.py

# ALWAYS specify output file to avoid overwriting live config
python src/backtest.py --start-window-days-back 90 --output-file test_90d.json
```

## ⚠️ Critical Safety Rules

1. **NEVER overwrite `best_strategy.json` during testing** - This controls live trading!
2. **ALWAYS use `--output-file`** parameter
3. **BACKUP before applying changes** to production

### Safe Testing Workflow

```bash
# 1. Run backtest with custom output
python src/backtest.py --start-window-days-back 90 --output-file test_strategy.json

# 2. Use the safe wrapper (recommended)
./backtest --start-window-days-back 90
# Creates timestamped files automatically
# Warns before overwriting production config

# 3. Review results
cat test_strategy.json | grep -E "Total_Return|Sharpe_Ratio|Total_Trades"

# 4. Only when ready to deploy:
cp best_strategy.json best_strategy.backup.json
cp test_strategy.json best_strategy.json
```

## Command Line Options

### Date Range
```bash
# Days back from today
--start-window-days-back 90   # Start 90 days ago (default: 120)
--end-window-days-back 0      # End today (default: 0)

# Specific trading window
--trading-window-days 60      # Test 60-day window from start
```

### Strategy Selection
```bash
# Skip adaptive strategy (faster testing)
--skip-adaptive

# Timeframe selection
--high-frequency 1H    # Primary timeframe: 5T, 15T, 30T, 1H, 4H, 1D
--low-frequency 15T    # Secondary timeframe (for some strategies)
```

### Optimization
```bash
# Set optimization iterations
--max-iterations 100   # More = better optimization (default: 50)

# Output file (REQUIRED for safety)
--output-file results.json
```

## Testing Scenarios

### 1. Recent Market (30 days)
```bash
python src/backtest.py \
  --start-window-days-back 30 \
  --output-file recent_30d.json
```

### 2. Extended Period (180 days)
```bash
python src/backtest.py \
  --start-window-days-back 180 \
  --output-file extended_180d.json
```

### 3. Specific Date Range
```bash
# Test 90 days ending 30 days ago (skip recent volatility)
python src/backtest.py \
  --start-window-days-back 120 \
  --end-window-days-back 30 \
  --output-file stable_period.json
```

### 4. Different Timeframes
```bash
# Conservative 4-hour candles
python src/backtest.py \
  --high-frequency 4H \
  --output-file conservative_4h.json

# Aggressive 30-minute
python src/backtest.py \
  --high-frequency 30T \
  --low-frequency 5T \
  --output-file aggressive_30m.json
```

### 5. Quick Test (no adaptive)
```bash
python src/backtest.py \
  --skip-adaptive \
  --max-iterations 20 \
  --output-file quick_test.json
```

## Understanding Results

### Key Files Generated

1. **Main output** (your specified filename):
   - Complete strategy configuration
   - Ready to use as `best_strategy.json`

2. **Additional files**:
   - `optimization_results.csv` - All parameter combinations tested
   - `strategy_comparison_enhanced.csv` - Strategy comparison
   - `all_strategy_results_enhanced.csv` - Detailed results
   - `adaptive_strategy_detailed_results.json` - Regime performance

### Important Metrics

```json
{
  "Total_Return": 5.38,         // 538% gain (multiplier, not percentage)
  "Sharpe_Ratio": 0.097,        // Risk-adjusted returns (>0 good, >1 excellent)
  "Total_Trades": 135,          // Number of trades executed
  "Average_Trades_Per_Day": 1.1,// Trading frequency
  "Profit_Factor": 1.019,       // Win/loss ratio (>1 is profitable)
  "Max_Drawdown": -15.2,        // Largest loss from peak (%)
  "Win_Rate": 0.52              // Percentage of profitable trades
}
```

### Adaptive Strategy Results

```json
"regime_performance": {
  "TRENDING": {
    "trades": 89,
    "success_rate": 0.64,
    "total_return": 3.2
  },
  "RANGING": {
    "trades": 32,
    "success_rate": 0.71,
    "total_return": 1.5
  },
  "VOLATILE": {
    "trades": 14,
    "success_rate": 0.43,
    "total_return": 0.9
  }
}
```

## Deployment Process

### When to Update Production

Consider updating when ALL conditions are met:
- ✅ Total Return > current strategy
- ✅ Sharpe Ratio > 0 (positive risk-adjusted)
- ✅ Win Rate > 45%
- ✅ Max Drawdown < 25%
- ✅ Tested on > 30 days of data
- ✅ At least 0.5 trades per day average

### Safe Deployment Steps

```bash
# 1. Stop trading on server
tdr> stop_auto_trade

# 2. Backup current configuration
cp best_strategy.json backups/best_strategy.$(date +%Y%m%d_%H%M%S).json

# 3. Review changes
diff best_strategy.json test_strategy.json

# 4. Apply new configuration
cp test_strategy.json best_strategy.json

# 5. Restart server to load new config
# On remote: Restart the server process

# 6. Resume trading with position
tdr> resume_auto_trade 1.52275326btc long 108234
```

## Troubleshooting

### Common Issues

**"No data found"**
- Check `btcusd.log` exists
- Verify date range has data
- Try `ls -la btcusd.log`

**"MemoryError"**
- Reduce date range
- Use higher frequency (4H not 1H)
- Add `--skip-adaptive`

**Unrealistic results**
- Check for lookahead bias
- Verify fees/slippage included
- Review trade frequency

### Debugging

```bash
# Check available data range
head -1 btcusd.log  # First record
tail -1 btcusd.log  # Last record

# Test with minimal settings
python src/backtest.py \
  --start-window-days-back 7 \
  --skip-adaptive \
  --output-file debug_test.json
```

## Best Practices

1. **Test multiple periods** - Markets change, avoid overfitting
2. **Start conservative** - Test longer periods first  
3. **Document tests** - Keep notes on what/why you tested
4. **Monitor after deployment** - New strategies need attention
5. **Regular reoptimization** - Run monthly or after major market changes

## Parameter Guidelines

### MA Strategy Windows
- Short: 5-25 (typically 10-15)
- Long: 20-100 (typically 40-50)
- Constraint: Long > Short

### Adaptive Parameters
- `regime_switch_threshold`: 0.3-0.7 (default 0.4)
- `signal_confirmation_bars`: 1-3 (default 2)
- `min_trade_gap_minutes`: 15-60 (default 15)

### Position Sizing (Current: 100% only)
- System trades all-in/all-out
- No partial positions
- Every trade is a full reversal

## Summary Checklist

Before running backtest:
- [ ] Have backup of `best_strategy.json`
- [ ] Know your current live position
- [ ] Understand test parameters

During backtest:
- [ ] Use `--output-file` parameter
- [ ] Save test results
- [ ] Document test conditions

Before deployment:
- [ ] Stop live trading
- [ ] Review all metrics
- [ ] Backup current config
- [ ] Test on recent data
- [ ] Have rollback plan

Remember: **Backtesting shows historical performance. Future results will differ!**