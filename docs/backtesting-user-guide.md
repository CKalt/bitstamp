# Backtesting User Guide

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Understanding the Backtester](#understanding-the-backtester)
4. [Running Backtests](#running-backtests)
5. [Interpreting Results](#interpreting-results)
6. [Strategy Optimization](#strategy-optimization)
7. [Applying Results to Live Trading](#applying-results-to-live-trading)
8. [Backtesting vs Reality](#backtesting-vs-reality)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

## Overview

The enhanced backtesting system (`src/bktst.py`) allows you to test trading strategies using historical data before risking real money. It accurately simulates:

- **Pivot Protection**: The same sticky levels and profit-aware trailing used in live trading
- **100% Position Flips**: Always fully invested (either 100% BTC or 100% USD)
- **Adaptive Strategies**: Market regime detection (TRENDING, RANGING, VOLATILE)
- **Realistic Costs**: Actual Bitstamp fees (0.12%) and market slippage

## Quick Start

### 1. Basic Backtest

```bash
# Test with last 30 days using current settings
./run_backtest.sh --month

# Test with last 7 days (quick test)
./run_backtest.sh --quick

# Test specific date range
./run_backtest.sh --start 2024-01-01 --end 2024-12-31
```

### 2. Compare Multiple Strategies

```bash
# Run automated comparison of 10 different configurations
python compare_strategies.py
```

### 3. Custom Configuration Test

```bash
# Test with specific config file
python src/bktst.py --config my_test_config.json --data btcusd.log
```

## Understanding the Backtester

### Architecture

The backtester mirrors the live trading system architecture:

```
Historical Data (btcusd.log)
    ↓
EnhancedBacktester
    ├── Position Tracking (100% invested)
    ├── Pivot Protection System
    │   ├── Sticky Support/Resistance
    │   └── Profit-Aware Trailing
    ├── Regime Detection
    │   ├── TRENDING → MA Crossover
    │   ├── RANGING → RSI Mean Reversion
    │   └── VOLATILE → Pivot Protection Only
    └── Trade Execution
        ├── Fees (0.12%)
        └── Slippage (0.05%)
```

### Key Components

1. **Position Management**
   - Always maintains 100% position (LONG or SHORT)
   - Tracks entry price and cost basis accurately
   - Simulates the actual 3-part order execution for BUY orders

2. **Pivot Protection**
   - Support/Resistance levels lock when position entered
   - Levels can trail to protect profits (never adversely)
   - Immediate position flip on level break
   - Configurable buffer zones and profit tiers

3. **Market Regimes**
   - Detects market conditions every hour
   - Requires 80% confidence to switch regimes
   - Each regime uses different trading logic

## Running Backtests

### Using run_backtest.sh

The helper script provides convenient shortcuts:

```bash
# Quick test (7 days)
./run_backtest.sh --quick --save quick_results.json

# Monthly test with custom balance
./run_backtest.sh --month --balance 50000 --save monthly_results.json

# Full year with specific config
./run_backtest.sh --year --config alt_strategy.json --save yearly_results.json
```

### Direct Python Usage

For more control, use the Python script directly:

```bash
python src/bktst.py \
    --data btcusd.log \
    --config best_strategy.json \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --initial-balance 10000 \
    --save-results backtest_2024.json
```

### Configuration Files

The backtester uses JSON configuration files. Key parameters:

```json
{
    "initial_balance": 10000,
    "fee_rate": 0.0012,              // Bitstamp's 0.12% fee
    "slippage_rate": 0.0005,         // 0.05% slippage
    
    // Pivot Protection Settings
    "enable_pivot_protection": true,
    "pivot_buffer": 100,             // Buffer zone in dollars
    "pivot_lookback_hours": 2,       // Recent high/low window
    "enable_trailing_pivots": true,  // Profit protection
    "pivot_profit_tiers": [
        {"threshold": 0.05, "protection_ratio": 0.70},  // At 5% profit, protect 70%
        {"threshold": 0.10, "protection_ratio": 0.80},  // At 10% profit, protect 80%
        {"threshold": 0.15, "protection_ratio": 0.85},
        {"threshold": 0.20, "protection_ratio": 0.90}
    ],
    
    // Adaptive Strategy Settings
    "regime_lookback": 100,
    "regime_switch_threshold": 0.80,
    "min_trade_gap_minutes": 15,
    "signal_confirmation_bars": 2,
    
    // MA Strategy Parameters
    "short_window": 10,
    "long_window": 20
}
```

## Interpreting Results

### Console Output

After running a backtest, you'll see:

```
==============================================================
BACKTEST RESULTS SUMMARY
==============================================================

Performance Metrics:
  Initial Balance:     $10,000.00
  Final Equity:        $12,456.78
  Total Return:        24.57%
  Sharpe Ratio:        1.45
  Max Drawdown:        -8.23%

Trading Statistics:
  Total Trades:        45
  Win Rate:            42.2%
  Average Win:         $485.23
  Average Loss:        $-234.56
  Profit Factor:       2.07

Execution Costs:
  Total Fees:          $267.89
  Total Slippage:      $134.56
  Cost % of P&L:       16.4%

Pivot Protection:
  Pivot Trades:        12
  Pivot Win Rate:      66.7%

Regime Performance:

  TRENDING:
    Trades:          23
    Win Rate:        52.2%
    Total Profit:    $1,845.67
    Time in Regime:  145.3 hours

  RANGING:
    Trades:          18
    Win Rate:        27.8%
    Total Profit:    $-234.56
    Time in Regime:  98.7 hours
```

### JSON Output

When using `--save-results`, you get detailed data including:

- Every trade with entry/exit prices and reasons
- Complete equity curve
- Regime switch history
- Pivot level changes over time

### Key Metrics Explained

1. **Sharpe Ratio**: Risk-adjusted return (>1.0 is good, >2.0 is excellent)
2. **Max Drawdown**: Largest peak-to-trough loss
3. **Win Rate**: Percentage of profitable trades
4. **Profit Factor**: Ratio of gross profits to gross losses
5. **Pivot Win Rate**: Success rate of pivot-triggered trades

## Strategy Optimization

### Using compare_strategies.py

This script tests multiple configurations automatically:

```bash
python compare_strategies.py
```

It tests variations including:
- Different pivot buffer sizes ($50, $100, $200)
- Static vs trailing pivots
- Various MA periods (5/15, 10/20, 20/50)
- Different regime switching thresholds

Output includes:
- Comparison table sorted by Sharpe ratio
- Visual charts saved as `strategy_comparison.png`
- Detailed results in `strategy_comparison_detailed.json`

### Manual Optimization Process

1. **Baseline Test**: Run with current settings
2. **Identify Weaknesses**: Look for low win rate or high drawdown
3. **Test Variations**: Modify one parameter at a time
4. **Compare Results**: Use Sharpe ratio as primary metric
5. **Validate**: Test on different time periods

### Common Optimizations

**If Win Rate is Low (<40%)**:
- Increase `signal_confirmation_bars` (3 or 4)
- Widen `pivot_buffer` to reduce whipsaws
- Increase `min_trade_gap_minutes`

**If Drawdowns are Large**:
- Enable trailing pivots
- Lower profit protection thresholds
- Reduce position size (in live trading)

**If Missing Profitable Moves**:
- Decrease `regime_switch_threshold`
- Use faster MA periods
- Reduce `signal_confirmation_bars`

## Applying Results to Live Trading

### 1. Update Configuration

After finding optimal parameters, update your live trading config:

```bash
# Copy the best performing configuration
cp backtest_best_config.json best_strategy.json

# Or manually edit specific parameters
vim best_strategy.json
```

### 2. Restart Trading System

On the server (chriskoin):

```bash
# Pull latest changes
git pull origin stable-added-adaptive-trad-n-chart-more

# Restart with new configuration
bin/server_control.sh restart
```

### 3. Monitor Performance

After applying changes:

```bash
# Check that new parameters are active
tdr> status long

# Monitor for a few days
tdr> trades
tdr> performance_summary
```

### Important Integration Points

The backtester shares code with the live system:

1. **Pivot Logic**: Both use the same support/resistance calculation
2. **Regime Detection**: Identical market classification algorithms
3. **Signal Generation**: Same MA crossover and RSI logic
4. **Position Management**: 100% position flips in both

This ensures backtesting results are representative of live performance.

## Backtesting vs Reality

### What Backtesting Captures Well

✅ **Strategy Logic**: Exact same trading rules and signals
✅ **Pivot Protection**: Identical support/resistance calculations
✅ **Regime Detection**: Same market classification
✅ **Position Management**: 100% invested constraint
✅ **Trading Fees**: Accurate Bitstamp fee structure
✅ **Basic Slippage**: Conservative estimates

### Limitations and Differences

❌ **Market Impact**: Large orders may move the market in reality
❌ **Execution Delays**: API latency and processing time
❌ **Partial Fills**: Bitstamp may partially fill orders
❌ **System Downtime**: Server crashes or network issues
❌ **Black Swan Events**: Extreme volatility or exchange outages
❌ **Psychological Factors**: Emotion-driven manual interventions

### Reality Adjustments

**Slippage Reality**:
- Backtest assumes 0.05% slippage
- Reality can be 0.1-0.5% during volatile periods
- Larger positions face more slippage

**Execution Timing**:
- Backtest executes instantly at signal
- Reality has 5-30 second delays for:
  - Signal confirmation
  - API communication
  - Order processing

**Market Conditions**:
- Backtest uses historical data
- Live markets may behave differently
- Regime detection may lag in real-time

### Best Practices

1. **Conservative Estimates**: If backtest shows 20% return, expect 15% live
2. **Out-of-Sample Testing**: Always test on data not used for optimization
3. **Paper Trading**: Run parallel paper account before committing capital
4. **Gradual Implementation**: Start with small position sizes
5. **Continuous Monitoring**: Compare live results to backtest expectations

## Advanced Features

### Custom Strategy Development

The backtester can be extended with new strategies:

```python
# In bktst.py, add new signal generation method
def generate_custom_signal(self, df: pd.DataFrame) -> Tuple[int, str]:
    # Your custom logic here
    if condition_met:
        return 1, "Custom signal: BUY condition"
    return 0, ""
```

### Multi-Timeframe Analysis

Test strategies on different bar sizes:

```python
# Resample 1-minute data to different timeframes
df_5min = df.resample('5T').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})
```

### Monte Carlo Simulations

Run multiple simulations with randomized parameters:

```python
# Vary parameters randomly within ranges
for i in range(100):
    config = base_config.copy()
    config['pivot_buffer'] = random.randint(50, 200)
    config['short_window'] = random.randint(5, 15)
    results = run_backtest(config)
```

## Troubleshooting

### Common Issues

**"Insufficient data for backtesting"**
- Need at least `max(long_window, regime_lookback) + 10` data points
- Check your date range isn't too restrictive

**"No trades executed"**
- Signals may not be confirming (check `signal_confirmation_bars`)
- Trade gap restriction may be too high
- Regime switching threshold may be preventing strategy changes

**Performance Doesn't Match Expectations**
- Verify configuration matches live system
- Check for look-ahead bias in custom modifications
- Ensure data quality (no gaps or corrupted entries)

**Memory Issues with Large Datasets**
- Use date ranges to limit data
- Increase system RAM or use a more powerful machine
- Consider sampling data for initial tests

### Debug Mode

Enable detailed logging:

```python
# In bktst.py constructor
self.logger.setLevel(logging.DEBUG)
```

This shows:
- Every signal generated
- Regime detection details
- Pivot level updates
- Trade execution logic

### Data Validation

Check your historical data:

```bash
# Verify data file integrity
head -n 100 btcusd.log
tail -n 100 btcusd.log

# Check for gaps
python -c "
import pandas as pd
from data.loader import parse_log_file
df = parse_log_file('btcusd.log')
gaps = df.index.to_series().diff()
print(f'Max gap: {gaps.max()}')
print(f'Gaps > 1 hour: {len(gaps[gaps > pd.Timedelta(hours=1)])}')
"
```

## Summary

The backtesting system provides a realistic simulation of the live trading system, allowing you to:

1. Test strategy changes without risking capital
2. Optimize parameters based on historical performance
3. Understand how pivot protection affects results
4. Compare different market regime strategies
5. Estimate realistic returns and risks

Remember that past performance doesn't guarantee future results, but proper backtesting significantly improves your odds of success. Always validate results across different time periods and market conditions before applying changes to live trading.

For questions or issues, check the codebase documentation or review the implementation in `src/bktst.py`.