# Claude Session Context - Complete Reference

## 🚨 CRITICAL ARCHITECTURE - ALWAYS REMEMBER 🚨

### System Architecture Overview

| Component | Location | Access | Purpose |
|-----------|----------|--------|---------|
| **Backtesting** | Mac (local) | Direct file access | Strategy testing & analysis |
| **Live Trading** | EC2 (remote) | `ssh ck` | Production trading |
| **Test Trading** | EC2 (remote) | `ssh ck` | Development testing |
| **Claude (You)** | Mac (local) | Working dir: `/Users/chris/projects/python/btc` | Development & analysis |

### Directory Navigation

```bash
# Mac (local) - where Claude runs
gg btc  → /Users/chris/projects/python/btc         (live code)
gg tst  → /Users/chris/projects/python/btc-testing (test code)

# Server (EC2) - remote trading systems
ssh ck
gg btc  → /home/chris/projects/bitstamp            (live trading)
gg tst  → /home/chris/projects/bitstamp-testing    (test trading)
```

### Deployment Workflow

**ALWAYS: Mac → GitHub → Server**

```bash
# 1. Make changes on Mac
gg btc
# edit files...

# 2. Commit and push
git add -A
git commit -m "Update description"
git push

# 3. Deploy to server
ssh ck
gg btc
git pull
# restart services as needed
```

## Current System Status (Live)

- **Position**: LONG 1.437 BTC @ $118,166 entry
- **Strategy**: MA 6/34 (corrected from 12/36)
- **Server**: EC2 t3.large (8GB RAM)
- **Branch**: stable-added-adaptive-trad-n-chart-more

## Key Configuration Files

### best_strategy.json (Complete Format)
```json
{
  "Frequency": "1H",
  "Strategy": "MA", 
  "Short_Window": 6,        // Fast MA period
  "Long_Window": 34,        // Slow MA period
  "do_live_trades": true,   // MUST be true for trading
  "strategy_type": "MA",
  "enable_adaptive_strategy": false,
  "auto_resume": false,     // Server forces true (bug)
  "ma_separation_threshold": 0.3,
  "max_trades_per_day": 5,
  "max_trades_per_hour": 2,
  "min_time_between_trades_minutes": 30,
  "enable_pivot_protection": false,
  "enable_regime_detection": false,
  "log_signal_evaluation": true,
  "verbose_logging": true
}
```

## Screen Sessions & Ports

### Mac (Local)
- `screen -S claude-tdr` - Claude Code session
- `screen -S client-tdr` - Live TDR client
- `screen -S client-tst` - Test TDR client

### Server (Remote via ssh ck)
- `screen -S btc` - Price feed (shared)
- `screen -S server` - Live trading (port 4000)
- `screen -S server-tst` - Test trading (port 4002)

### Port Configuration
- 4000: Live server
- 8050: Live charts
- 4002: Test server  
- 8051: Test charts

## Common Operations

### Check Trading Status
```bash
# Local check
curl http://localhost:4000/api/status

# Remote check
ssh ck "cd /home/chris/projects/bitstamp && tail -20 logs/tdr_server.log | grep -E '🎯|🚨|SIGNAL_EVAL'"
```

### Restart Trading Server
```bash
ssh ck
gg btc
screen -S server -X quit
screen -dmS server bash -c 'source source-venv.sh && python src/tdr.py --server'
# Wait 3-5 minutes for data load
# Then restart auto_trade with position info
```

### Run Backtests
```bash
# In development directory with progress tracking
gg tst && source source-venv.sh && python run_backtest_with_progress.py

# Quick 30-day test
python src/bktst.py --start-window-days-back 30 --ma-short 6 --ma-long 34
```

## Known Issues & Fixes

### Fixed This Session
1. MA configuration (was 12/36, fixed to 6/34)
2. EC2 memory upgrade (t3.small → t3.large)
3. Position tracking mismatch
4. Virtual environment setup in development

### Still Present
1. Auto-resume forces True regardless of config
2. Entry price calculated in 5+ places
3. No single source of truth for MA values
4. Resume file can override configuration

## Emergency Procedures

```bash
# Stop live trading
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "stop_auto_trade"}'

# Kill screen session
screen -S [name] -X quit
```

## Session Resume Instructions

### Quick Resume
```bash
claude --resume
```

### After Long Break
```bash
claude "Please read docs/CLAUDE_SESSION_CONTEXT.md and continue where we left off"
```

## Important Reminders

1. **NEVER** suggest direct file copy to server - use git
2. **NEVER** run trading commands locally - they run on EC2
3. **ALWAYS** check architecture when suggesting deployments
4. **btcusd.log** (4.4GB) is on Mac for backtesting
5. **Trading servers** run on EC2 via ssh ck
6. **Development** happens on Mac, deployment via git

## Current TODO List Priority

1. Run comprehensive backtest in development branch ⏳
2. Test optimized strategy on test server
3. Deploy system verifier to production
4. Fix auto-resume forcing True
5. Consolidate position tracking