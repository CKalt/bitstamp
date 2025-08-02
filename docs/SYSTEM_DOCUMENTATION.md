# TDR Trading System Documentation

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Current Trading Status](#current-trading-status)
3. [Development Environment](#development-environment)
4. [Critical Operations](#critical-operations)
5. [Development Process](#development-process)
6. [Recent Work & Plans](#recent-work--plans)

---

## System Architecture

### Overview
The Bitcoin trading system consists of backtesting components running locally on Mac and live trading servers running on EC2. All code deployment flows through GitHub.

### Component Locations

#### Local Mac Environment
- **Purpose**: Development, backtesting, and strategy optimization
- **Live Directory**: `/Users/chris/projects/python/btc`
- **Test Directory**: `/Users/chris/projects/python/btc-testing` (development branch)
- **Data**: btcusd.log (4.4GB of historical tick data)

#### Remote EC2 Environment
- **Purpose**: Live and test trading servers
- **Access**: `ssh ck`
- **Live Directory**: `/home/chris/projects/bitstamp`
- **Test Directory**: `/home/chris/projects/bitstamp-testing`
- **Instance Type**: t3.large (8GB RAM)

### Directory Navigation (ggmap shortcuts)
```bash
# Local Mac
gg btc  → /Users/chris/projects/python/btc         (live code)
gg tst  → /Users/chris/projects/python/btc-testing (test code)

# Server (ssh ck)
source ~/ggmap  # Required first
gg btc  → /home/chris/projects/bitstamp            (live code)
gg tst  → /home/chris/projects/bitstamp-testing    (test code)
```

### Network Architecture

#### Ports
- **4000**: Live TDR server API
- **4002**: Test TDR server API
- **8050**: Live Dash charts
- **8051**: Test Dash charts

#### SSH Tunnels (autossh)
Managed via `~/projects/tunnels/tunnels.json` and `fix-tunnels` command.

---

## Current Trading Status

### Live Position (as of 2025-07-28)
- **Status**: SHORT position
- **Amount**: ~1.444 BTC equivalent in USD
- **Entry**: ~$118,000
- **Strategy**: MA 6/34 with adaptive features
- **Server**: Running on EC2 port 4000

### Test Position
- **Status**: LONG position (testing MA 3/22)
- **Amount**: 0.001 BTC (limited to ~$100)
- **Purpose**: Verify backtest accuracy
- **Server**: Running on EC2 port 4002

---

## Development Environment

### Screen Sessions

#### On Mac
- `screen -S claude-tdr` - Claude Code session
- `screen -S client-tdr` - Live trading client
- `screen -S client-tst` - Test trading client (when needed)

#### On Server (ssh ck)
- `screen -S btc` - WebSocket price feed (shared by all)
- `screen -S server` - Live trading server
- `screen -S server-tst` - Test trading server

### Git Workflow
1. **Branches**:
   - `stable-added-adaptive-trad-n-chart-more`: Production
   - `development`: Testing (local to Mac)

2. **Deployment Process**:
   - Develop and test on Mac
   - Commit and push to GitHub
   - SSH to server and pull changes
   - Restart affected services

3. **NEVER**:
   - Copy files directly to server
   - Run trading commands locally
   - Modify production without testing

---

## Critical Operations

### Starting Servers

#### Live Server (EC2)
```bash
ssh ck
source ~/ggmap && gg btc
screen -dmS server bash -c 'source source-venv.sh && python src/tdr.py --server'
```

#### Test Server (EC2)
```bash
ssh ck
source ~/ggmap && gg tst
screen -dmS server-tst bash -c 'source source-venv.sh && python src/tdr.py --server --port 4002'
```

### Connecting Clients

#### Live Client (Mac)
```bash
gg btc
source source-venv.sh
python src/tdr.py  # Connects to localhost:4000 via tunnel
```

#### Test Client (Mac)
```bash
gg tst
source source-venv.sh
python src/tdr.py --server-url http://localhost:4002
```

### Emergency Operations

#### Stop All Trading
```bash
# On server
ssh ck
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "stop_auto_trade"}'
```

#### Check Status
```bash
curl -s http://localhost:4000/api/status | python3 -m json.tool
```

---

## Development Process

### Preventing Regressions

1. **Feature Flags**: Every change behind a flag that defaults to false
2. **Parallel Implementation**: Never modify existing code directly
3. **Test Environment First**: All changes tested on port 4002
4. **Position Verification**: Always verify position tracking before/after changes

### Known Complexity Points

1. **Position Tracking**:
   - Dual system: `position` (1/-1) and `position_size`/`position_cost_basis`
   - Must keep synchronized across all components

2. **Entry Price Calculation**:
   - Calculated in multiple places
   - Must handle both real and theoretical trades

3. **Strategy Interfaces**:
   - MACrossoverStrategy vs AdaptiveMultiStrategy
   - Different attributes and behaviors

### Testing Checklist

Before deploying any change:
- [ ] Test on development branch
- [ ] Verify position tracking unchanged
- [ ] Check entry price calculations
- [ ] Confirm auto-resume works
- [ ] Test with both strategies

---

## Recent Work & Plans

### Completed (July 2025)
1. ✅ Fixed critical backtest bug (was using tick data, not hourly)
2. ✅ Found MA 3/22 as top performer (8.64% in 30 days)
3. ✅ Created comparison logging system
4. ✅ Deployed test environment with MA 3/22
5. ✅ Fixed position display bug ("NEUTRAL error")

### In Progress
- 🔄 Collecting live trading data for backtest comparison
- 🔄 Monitoring MA 3/22 performance on test server

### Planned
1. **Backtest Verification** (This Week)
   - Run daily comparisons of live vs backtest
   - Verify trade timing matches exactly
   - Document any discrepancies

2. **System Improvements** (Next Week)
   - Deploy system verifier for regression detection
   - Create feature flags system
   - Consolidate position tracking

3. **Strategy Optimization** (Once Verified)
   - Test top strategies from backtest
   - Implement safeguards for aggressive strategies
   - Consider ensemble approaches

### Configuration Files

#### best_strategy.json (Live)
```json
{
  "Short_Window": 6,
  "Long_Window": 34,
  "do_live_trades": true,
  "strategy_type": "MA",
  "enable_adaptive_strategy": false
}
```

#### best_strategy.json (Test - MA 3/22)
```json
{
  "Short_Window": 3,
  "Long_Window": 22,
  "do_live_trades": true,
  "max_position_btc": 0.001,
  "max_position_usd": 100
}
```

---

## Important Reminders

1. **Always specify host**: When giving commands, specify "ON MAC" or "ON SERVER"
2. **Use gg shortcuts**: `gg tst` instead of `cd $(ggdir tst)`
3. **Source ggmap on server**: Always run `source ~/ggmap` first on EC2
4. **Check tunnels**: Run `fix-tunnels` if connection issues
5. **Git only deployment**: Never copy files directly to server
6. **Position limits**: Test server limited to 0.001 BTC for safety

---

## Support Information

- **Claude Code Help**: `/help`
- **Report Issues**: https://github.com/anthropics/claude-code/issues
- **System Logs**: 
  - Live: `ssh ck "tail -f ~/projects/bitstamp/logs/tdr_server.log"`
  - Test: `ssh ck "tail -f ~/projects/bitstamp-testing/logs/tdr_server.log"`

---

*Last Updated: 2025-07-29*

---

## USK (Update Session Knowledge)

**This document serves as the primary USK reference for the TDR Trading System.**

When starting a new Claude session, reference this document to understand:
- Current system state and trading positions
- Architecture and directory structure  
- Active development work and plans
- Critical operations and procedures

### Directory Structure Convention
- **`src/*`** - Core Python application code and general-use Python scripts
- **`bin/*`** - General-use shell scripts for operations
- **`claude-bin/*`** - Claude Code-specific scripts (both .py and .sh) for analysis, monitoring, and verification
- **`docs/*`** - All documentation (see Documentation Layout below)

**Important**: All scripts should be run from the project root directory:
- `python3 src/tdr.py`
- `bash bin/quick_restart.sh`
- `python3 claude-bin/check_change_status.py`

Key points for new session:
1. Live system is SHORT ~1.41 BTC @ ~$119k (MA 6/34)
2. Test system is SHORT position @ ~$118,200 (MA 3/22) with JSONL logging
3. Dual environment: btc (live) and tst (test) directories
4. Never copy files to server - use git deployment only
5. Always specify "ON MAC" or "ON SERVER" for commands
6. Current focus: Verifying JSONL trade logging and backtest accuracy

### Documentation Layout

#### Core Documentation
- **SYSTEM_DOCUMENTATION.md** - Primary USK and system overview (this file)
- **README.md** - Project overview and quick start
- **SECURITY_NOTICE.md** - Security considerations and API key handling

#### Development & Deployment
- **DEPLOYMENT_CHECKLIST.md** - Step-by-step deployment procedures
- **QUICK_DEPLOY_GUIDE.md** - Rapid deployment reference
- **development-process-summary.md** - Development workflow and best practices
- **development-process-and-complexity-management.md** - Managing system complexity

#### Change Tracking
- **CODE_CHANGE_LOG.md** - Git commit tracking and verification status
- **CHANGE_TRACKING.md** - Detailed change verification procedures
- **daily_tracking_template.md** - Template for daily progress tracking
- **session_updates/** - Session-specific change logs
- **weekly_reviews/** - Weekly progress summaries

#### System Behavior & Features
- **SYSTEM_KEY_BEHAVIORS.md** - Critical system behaviors to preserve
- **auto-resume-flow.md** - Auto-resume feature documentation
- **signal-timing-improvement-plan.md** - Signal evaluation improvements
- **whipsaw_features.md** - Whipsaw protection features

#### Operations & Monitoring
- **server_management.md** - Server operations guide
- **server-user-guide.md** - Using the TDR server
- **tdr-user-guide.md** - TDR client usage
- **monitoring-guide.md** - System monitoring procedures
- **monitoring-setup-howto.md** - Setting up monitoring
- **screen-architecture.md** - Screen session management
- **safe-restart-procedure.md** - Safe system restart procedures

#### Trading & Strategy
- **backtesting-user-guide.md** - Comprehensive backtesting guide
- **backtesting-quick-reference.md** - Quick backtesting commands
- **ENHANCED_BACKTESTING_SUMMARY.md** - Advanced backtesting features
- **LIVE_VS_BACKTEST_COMPARISON_PLAN.md** - Verifying backtest accuracy
- **trading_diagnostics_guide.md** - Debugging trading issues
- **strategy-8-hour.md** - 8-hour strategy documentation

#### API & Technical
- **bitstamp-api-doc.yaml** - Bitstamp API reference
- **dual-version-setup.md** - Running live and test systems
- **deployment-early-warning-and-fixes.md** - Early warning system

### Recent Session Work (2025-07-29)
1. **Fixed critical bug**: Position was being updated BEFORE trade execution
   - Moved position updates to AFTER execute_trade/buy_in_three_parts
   - Fixed in both MACrossoverStrategy and AdaptiveMultiStrategy
   - Deployed to test server

2. **Enabled comparison logging**:
   - Integrated BacktestComparisonLogger into strategies.py
   - Added enable_comparison_logging parameter to shell.py
   - Fixed method signatures (log_signal_evaluation, log_trade_decision)
   - Deployed to test server - logs are being created
   - Still debugging JSON serialization issues with numpy int64 types

3. **Test server status**:
   - Running with MA 3/22 strategy
   - Successfully executing trades (1-2 trades today)
   - Position tracking bug is fixed
   - Comparison logging partially working (session logs created)
   - Signal logs not yet populating due to serialization errors

4. **Outstanding issues**:
   - JSON serialization errors with numpy int64 types in comparison logging
   - API sometimes shows stale position data from data_manager
   - Test server using different git repo than Mac development

5. **Implemented JSONL trade logging**:
   - Changed trades.json from JSON array to JSONL format
   - PRE_TRADE entries logged before order placement
   - POST_TRADE entries logged with Bitstamp response
   - Deployed to test server, awaiting trade to verify
   - Added claude-bin/check_change_status.py for verification

6. **API Monitoring Tools** (Added 2025-07-29):
   - Created claude-bin/api_monitor.py for direct API access
   - Alternative to file-based command system for Claude
   - Provides real-time dashboard and signal monitoring
   - Does NOT change human user workflow (still use src/tdr.py)
   - Commands:
     ```bash
     # Dashboard mode
     python3 claude-bin/api_monitor.py --dashboard
     
     # Single command
     python3 claude-bin/api_monitor.py --command "status"
     
     # Show curl equivalent
     python3 claude-bin/api_monitor.py --command "status" --curl
     ```

7. **Documentation Consolidation** (Added 2025-07-29):
   - Created `/Users/chris/projects/python/btc-testing/prompts/CLAUDE-START-HERE.md`
   - Central navigation guide for all TDR documentation
   - Organizes docs by category and use case
   - Provides quick reference for common tasks
   - Essential starting point for new Claude sessions

8. **Next steps**:
   - Verify JSONL logging when first trade occurs
   - Add MA values to server API for better signal tracking
   - Fix JSON serialization by converting all numpy types
   - Run daily backtest comparisons once logging fully works
   - Monitor for MA crossover signals