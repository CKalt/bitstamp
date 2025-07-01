# 🚀 Quick Deployment Guide - Enhanced Backtesting

## Pre-Flight Checklist (5 minutes)
```bash
# 1. Create backup
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
cp best_strategy.json backups/$(date +%Y%m%d_%H%M%S)/

# 2. Test compatibility
python src/test_compatibility.py

# 3. Quick backtest (last 30 days)
python src/bktst_enhanced.py --start-window-days-back 30
```

## Deploy Sequence (10 minutes)
```bash
# 1. Validate new strategy
python src/validate_strategy.py

# 2. Preview changes
python src/strategy_migrator.py --dry-run

# 3. Stop trading (if running)
# In TDR: save_resume_state then exit

# 4. Apply new config
python src/strategy_migrator.py

# 5. Start paper trading
# Edit best_strategy.json: "do_live_trades": false
python src/tdr.py
```

## In TDR Shell - Testing (15 minutes)
```bash
# Enable monitoring
enable_commands

# Check status
status
strategy_diagnostics

# Test new commands
backtest_current_params 1
signal_analysis
risk_metrics

# Monitor for 15 minutes
# Look for proper signals and no errors
```

## Go Live
```bash
# Only after successful paper trading
# Edit best_strategy.json: "do_live_trades": true
# Restart TDR
```

## Emergency Rollback
```bash
# If anything goes wrong:
cp backups/[latest]/best_strategy.json .
# Restart TDR
```

## Key Commands Reference
- `backtest_current_params [days]` - Test current params
- `compare_strategies` - Compare with recommended
- `regime_history [hours]` - Show regime changes
- `signal_analysis [verbose]` - Why trading/not trading
- `risk_metrics` - Current exposure
- `validate_performance [hours]` - Check vs backtest

**Remember: Safety first! Test thoroughly in paper mode.**