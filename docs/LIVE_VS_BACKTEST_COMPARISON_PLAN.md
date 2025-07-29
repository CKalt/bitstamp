# Live Trading vs Backtest Comparison System

## Overview

This document outlines the system for comparing live trading results on the test server with backtesting results to verify that our backtesting engine accurately reflects real trading behavior.

## Current Infrastructure

### 1. Enhanced Logging System (✅ Implemented)

We've created `BacktestComparisonLogger` that captures:

#### Hourly Bar Data
- Timestamp of each hourly bar
- OHLCV data (Open, High, Low, Close, Volume)
- MA values calculated at that hour
- Bar hash for verification

#### Signal Evaluations
- Every time the system checks for a trade signal
- Current price and MA values
- Previous vs current signal
- Decision made (trade/no trade) and reason

#### Trade Executions
- Exact timestamp when trade occurs
- Trade type (BUY/SELL)
- Price and amount
- Position before and after
- MA values at time of trade
- Verification hash

### 2. Comparison Tool (✅ Implemented)

`compare_live_vs_backtest.py` that:
- Loads live trading logs from test server
- Runs equivalent backtest for same time period
- Compares trades to find:
  - Matched trades (same time, same direction)
  - Live-only trades (trades that backtest missed)
  - Backtest-only trades (trades live system missed)
- Calculates accuracy percentage

## Planned Workflow

### Day 1-3: Data Collection Phase
1. **Test server runs continuously** with MA 3/22
   - Logs every hourly bar
   - Logs every signal evaluation
   - Logs any trades executed

2. **Files created daily**:
   ```
   logs/backtest_comparison/
   ├── live_trading_2025-07-28.jsonl    # Main trading log
   ├── signals_2025-07-28.jsonl         # Signal evaluations
   └── hourly_bars_2025-07-28.jsonl     # Hourly bar data
   ```

### Day 4+: Comparison Phase

1. **Run daily comparison**:
   ```bash
   python claude-bin/compare_live_vs_backtest.py 2025-07-28
   ```

2. **Expected outputs**:
   - Comparison report showing:
     - Number of trades matched
     - Any discrepancies
     - Accuracy percentage
   - Saved report: `logs/backtest_comparison/comparison_2025-07-28.json`

## What We're Verifying

### 1. Trade Timing Accuracy
- Do trades happen at the exact same hourly bars?
- Are MA crossovers detected at the same moments?

### 2. Price Accuracy
- Are the hourly close prices identical?
- Do MA calculations match exactly?

### 3. Signal Processing
- Does the live system evaluate signals every hour?
- Are trading rules applied consistently?

### 4. Edge Cases
- First hour of the day
- Weekend gaps
- Low volume periods
- Rapid price movements

## Success Criteria

1. **100% Trade Matching**: Every trade in live should match backtest timing
2. **MA Value Accuracy**: MA calculations should match within 0.01%
3. **No Ghost Trades**: No trades should appear in only one system

## Troubleshooting Guide

### If trades don't match:

1. **Check hourly bar alignment**
   - Verify both systems use same hourly boundaries
   - Check timezone handling

2. **Verify MA calculations**
   - Ensure same number of bars used
   - Check for data gaps

3. **Review signal evaluation logs**
   - Find where decisions diverged
   - Check for race conditions

## Next Steps

### Immediate (Today)
- ✅ Test server running with MA 3/22
- ✅ Comparison logging enabled
- ⏳ Collecting data...

### Tomorrow
- First comparison run after 24 hours of data
- Review any discrepancies
- Adjust logging if needed

### This Week
- Daily comparison runs
- Build confidence in backtest accuracy
- Document any findings

### Future Enhancements
1. **Real-time comparison dashboard**
2. **Automated alerts for discrepancies**
3. **Statistical analysis of differences**
4. **Integration with main trading system**

## Command Reference

### Check if logging is working:
```bash
# On server
ssh ck "source ~/ggmap && gg tst && ls -la logs/backtest_comparison/"
```

### View recent signals:
```bash
# On server
ssh ck "source ~/ggmap && gg tst && tail -f logs/backtest_comparison/signals_$(date +%Y-%m-%d).jsonl"
```

### Run comparison:
```bash
# On Mac
gg tst
python claude-bin/compare_live_vs_backtest.py
```

### View comparison report:
```bash
# On Mac
gg tst
cat logs/backtest_comparison/comparison_$(date +%Y-%m-%d).json | python -m json.tool
```

## Important Notes

1. **MA 3/22 is very aggressive** - Expect 1-2 trades per day based on backtests
2. **Position limited to 0.001 BTC** - Safe for testing
3. **Logs are created daily** - New files each day at midnight
4. **Comparison requires historical data** - btcusd.log must be accessible

## Monitoring

To ensure the system is working:

1. **Check for hourly bars**:
   - Should see 24 entries per day
   - Each hour should have MA values

2. **Check for signal evaluations**:
   - Should see evaluations throughout the day
   - Most will be "No signal change"

3. **Check for trades** (when they occur):
   - Full details logged
   - Verification hash for matching

This system will provide definitive proof that our backtesting accurately reflects live trading behavior, giving us confidence to optimize strategies based on backtest results.