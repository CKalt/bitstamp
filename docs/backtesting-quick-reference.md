# Backtesting Quick Reference

## Essential Commands

### Quick Tests
```bash
# Last 7 days (quick check)
./run_backtest.sh --quick

# Last 30 days (standard test)
./run_backtest.sh --month

# Last year (comprehensive)
./run_backtest.sh --year

# Custom date range
./run_backtest.sh --start 2024-01-01 --end 2024-12-31
```

### Strategy Comparison
```bash
# Compare 10 different configurations automatically
python compare_strategies.py

# Results saved to:
# - strategy_comparison.png (visual charts)
# - strategy_comparison_detailed.json (raw data)
```

### Custom Configuration Test
```bash
# Test specific config
python src/bktst.py --config my_config.json --save-results results.json

# With custom balance
python src/bktst.py --initial-balance 50000 --config best_strategy.json
```

## Key Configuration Parameters

### Pivot Protection
```json
{
    "pivot_buffer": 100,              // $50, $100, or $200
    "enable_trailing_pivots": true,   // Dynamic profit protection
    "pivot_profit_tiers": [           // When to lock profits
        {"threshold": 0.05, "protection_ratio": 0.70}
    ]
}
```

### Strategy Tuning
```json
{
    "short_window": 10,               // Fast MA period
    "long_window": 20,                // Slow MA period
    "signal_confirmation_bars": 2,    // Signals needed
    "min_trade_gap_minutes": 15      // Minimum between trades
}
```

## Performance Metrics Guide

| Metric | Good | Excellent | Concern |
|--------|------|-----------|---------|
| Sharpe Ratio | >1.0 | >2.0 | <0.5 |
| Win Rate | >45% | >55% | <40% |
| Max Drawdown | <15% | <10% | >20% |
| Profit Factor | >1.5 | >2.0 | <1.2 |

## Common Optimization Scenarios

### Low Win Rate (<40%)
```json
{
    "signal_confirmation_bars": 3,    // Increase from 2
    "pivot_buffer": 150,              // Increase from 100
    "min_trade_gap_minutes": 30       // Increase from 15
}
```

### High Drawdowns (>20%)
```json
{
    "enable_trailing_pivots": true,
    "pivot_profit_tiers": [
        {"threshold": 0.03, "protection_ratio": 0.50},  // Protect earlier
        {"threshold": 0.05, "protection_ratio": 0.70}
    ]
}
```

### Too Few Trades
```json
{
    "regime_switch_threshold": 0.60,  // Lower from 0.80
    "signal_confirmation_bars": 1,    // Lower from 2
    "min_trade_gap_minutes": 10       // Lower from 15
}
```

## Applying Results to Live Trading

1. **Find Best Config**
   ```bash
   # From compare_strategies.py output
   cat strategy_comparison_detailed.json | jq '.[0]' > best_config.json
   ```

2. **Update Live System**
   ```bash
   # Copy to live config
   cp best_config.json best_strategy.json
   git add best_strategy.json
   git commit -m "Update strategy parameters from backtesting"
   git push
   ```

3. **Restart Server** (on chriskoin)
   ```bash
   git pull
   bin/server_control.sh restart
   ```

## Warning Signs in Results

🚨 **Red Flags**:
- Win rate below 35%
- Sharpe ratio below 0.5
- More than 50% of profits from one lucky trade
- Drawdown exceeds 25%
- High sensitivity to small parameter changes

✅ **Good Signs**:
- Consistent profits across different regimes
- Pivot trades outperform regular trades
- Low correlation with market direction
- Robust to parameter variations
- Profitable in both trending and ranging markets

## Data Requirements

- **Minimum**: 30 days for basic testing
- **Recommended**: 6 months for reliable results  
- **Optimal**: 1+ years covering different market conditions

## File Locations

- **Historical Data**: `btcusd.log`
- **Current Config**: `best_strategy.json`
- **Test Configs**: `configs/` directory
- **Results**: `backtest_results/` directory
- **Charts**: `strategy_comparison.png`