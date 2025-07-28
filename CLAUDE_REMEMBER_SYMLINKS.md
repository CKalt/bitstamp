# CRITICAL: btcusd.log SYMLINK STRUCTURE

## 🔗 SYMLINK ARCHITECTURE (NEVER FORGET!)

### On Mac (local):
- `/Users/chris/projects/python/btc/btcusd.log` - ACTUAL FILE (4.4GB)
- `/Users/chris/projects/python/btc-testing/btcusd.log` → SYMLINK to ../btc/btcusd.log

### On Server (ck):
- `/home/chris/projects/bitstamp/btcusd.log` - ACTUAL FILE  
- `/home/chris/projects/bitstamp-testing/btcusd.log` → SYMLINK to ../bitstamp/btcusd.log

## What This Means:
1. **Both directories can access btcusd.log** - no need to copy
2. **Backtesting works in BOTH directories** - they see the same data
3. **Development directory (gg tst) is FULLY FUNCTIONAL** for backtesting

## Correct Workflow:
```bash
# Work in development directory
gg tst  # or cd to btc-testing

# btcusd.log is available here via symlink
ls -la btcusd.log  # Shows -> ../btc/btcusd.log

# Run backtests directly in development
source source-venv.sh
python run_backtest_with_progress.py
```

## THIS CHANGES EVERYTHING:
- Run new features in development directory
- Test there first
- No need to work in live directory for backtesting
- Both directories share the same price data