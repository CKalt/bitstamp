# CLAUDE-RESUME.md

## Current State (2025-08-11)

### Production System Status (gg btc - port 4000)
- **Status**: NOT currently running (no process found)
- **Last Known Position**: LONG ~1.2 BTC
- **Branch**: `stable-added-adaptive-trad-n-chart-more`
- **Configuration**: 
  - Hourly candles (1h)
  - LIVE trading
  - MA crossover strategy
  - Proximity threshold: 0.3%

### Test System Status (gg tst - port 4001)
- **Status**: RUNNING on development branch
- **Position**: SHORT 0.0395 BTC @ $118,525 (paper trading)
- **Current Price**: ~$118,649
- **Configuration**:
  - 1-minute candles for rapid testing (60x faster)
  - PAPER trading only (no real money)
  - Adaptive strategy (trending/ranging/volatile modes)
  - Auto-resume: enabled
  - Proximity threshold: 0.3%
- **Performance**: 
  - 4 trades today
  - Lost 28.77% over 7 days in backtesting
  - Win rate: ~15% (needs improvement)

## Major Work Completed (2025-08-11)

### 1. Indexed Backtesting System ✅
**Problem**: Reading 16.7M lines (4.5GB) took 5-10 minutes for 1 day of data.

**Solution**: Created daily index system:
- `btcusd.log.idx` - JSON index with daily summaries
- `/src/backtesting/build_index.py` - Index builder
- `/src/backtesting/backtest_adaptive_indexed.py` - Fast backtest using index

**Results**: 
- First run: 90 seconds to build index
- Subsequent runs: 2 seconds for 1 day of data
- 300x speed improvement!

### 2. Test Server Fixes on Development Branch ✅
**Deployed to ck test server**:
- Fixed auto-resume bug (now respects `auto_resume` flag)
- Fixed negative entry price bug (line 1132 in strategies.py)
- Switched from MA to adaptive strategy
- Committed changes: `9130e66`

### 3. Backtesting Infrastructure ✅
**Created complete backtesting system**:
- Matches paper trading configuration exactly
- 1-minute bars for rapid testing
- Adaptive strategy with three market modes
- Easy-to-use scripts in `claude-bin/`

## Key Findings

### Market Analysis
- **Current Market**: Extremely choppy/sideways (0.1-0.5% daily moves)
- **MA Strategies**: ALL losing money in this market
- **Adaptive Strategy**: Also struggling (16% win rate)
- **Root Cause**: No strategy works well in directionless markets

### Performance Metrics
| System | Strategy | Timeframe | Performance |
|--------|----------|-----------|-------------|
| Production | MA 6/34 | Hourly | Unknown (not running) |
| Test (paper) | Adaptive | 1-min | -6.92% (1 day) |
| Test (backtest) | Adaptive | 1-min | -28.77% (7 days) |

## File Locations

### Local (Mac)
- **Main**: `/Users/chris/projects/python/btc` (gg btc)
- **Test**: `/Users/chris/projects/python/btc-testing` (gg tst)

### Remote (ck server)
- **Production**: `/home/chris/projects/bitstamp` (gg btc after `source ~/ggmap`)
- **Test**: `/home/chris/projects/bitstamp-testing` (gg tst after `source ~/ggmap`)

## Important Scripts

### Backtesting
```bash
# Run indexed backtest (fast!)
gg btc
./claude-bin/run_adaptive_backtest.sh --days 7

# Build/rebuild index
python3 src/backtesting/build_index.py --rebuild
```

### Monitoring
```bash
# Check test server status
ssh ck 'curl -s http://localhost:4001/api/status | python3 -m json.tool'

# Monitor paper trading
ssh ck 'tail -f /home/chris/projects/bitstamp-testing/logs/tdr_server.log'
```

## Navigation Conventions
- **Always use `gg` system**: `gg btc` or `gg tst`, never `cd`
- **On server**: First run `source ~/ggmap` then use `gg` shortcuts
- **Claude scripts**: Always in `claude-bin/` directory
- **General scripts**: In `bin/` directory

## Critical Reminders
1. **Test server** (gg tst) is on **development branch**
2. **Production** (gg btc) is on **stable-added-adaptive-trad-n-chart-more branch**
3. **Never copy files directly** to server - use git deployment
4. **Paper trading only** on test server (no real money)
5. **Commits**: Made to development branch on test server

## Next Session Priorities
1. **Tune adaptive strategy parameters** - Current 16% win rate is too low
2. **Consider mean reversion strategy** for sideways markets
3. **Add stop-loss/take-profit** mechanisms
4. **Monitor paper trading** for improvement patterns
5. **Restart production server** if needed

## Recent Commits
- `9130e66` - Fix critical bugs and switch to adaptive strategy (development branch)

## Summary
Successfully created indexed backtesting system (300x faster), deployed adaptive strategy to test server, and identified that all momentum strategies are losing money in current sideways market. Infrastructure is working perfectly - strategy needs tuning for market conditions.