# In Case I Crash - Critical System State Details

## CURRENT SITUATION (as of 2025-07-27)
Dual-version development environment created to prevent regression bugs. Live system running SHORT position @ $118,011 entry (flipped from LONG @ $116,526 with $2,142.90 profit).

## DUAL-VERSION SETUP

### Overview
We run two parallel versions for safe development:
1. **Live**: Full position trading (~1.444 BTC)
2. **Test**: Limited position (0.001 BTC / ~$100)

### Directory Navigation (ggmap)
```bash
# Local Mac
`gg btc`  → /Users/chris/projects/python/btc         (live)
`gg tst`  → /Users/chris/projects/python/btc-testing (test)

# Server (ck)
`gg btc`  → /home/chris/projects/bitstamp            (live)
`gg tst`  → /home/chris/projects/bitstamp-testing    (test)
```

### Screen Sessions

#### Local Mac
- `screen -S claude-tdr` - Claude Code session (this session)
- `screen -S client-tdr` - Live TDR client
- `screen -S client-tst` - Test TDR client (to be created)

#### Server (ck)
- `screen -S btc` - Price feed (websock-ticker2.py) - SHARED BY BOTH
- `screen -S server` - Live trading server (port 4000)
- `screen -S server-tst` - Test trading server (port 4002)

### Port Configuration
- **4000**: Live TDR server
- **8050**: Live TDR charts
- **4002**: Test TDR server (changed from 4001 to avoid EDS conflict)
- **8051**: Test TDR charts

### Autossh Tunnels
Updated `~/projects/tunnels/tunnels.json` to include test ports:
```json
{
  "local": 4002,
  "remote": 4002,
  "description": "TDR TEST SERVER"
},
{
  "local": 8051,
  "remote": 8051,
  "description": "TDR TEST CHARTS"
}
```

Restart tunnels: `fix-tunnels` (alias for `cd ~/projects/tunnels && bin/ssh-add-two`)

### Price Feed Sharing
Both versions share the same price feed via symlink:
```
/home/chris/projects/bitstamp/btcusd.log         (actual file)
/home/chris/projects/bitstamp-testing/btcusd.log → ../bitstamp/btcusd.log
```

### Test Server Configuration
`best_strategy.json` in test directories:
```json
{
  "trading_mode": "development",
  "max_position_btc": 0.001,
  "max_position_usd": 100,
  "server_port": 4002,
  "log_prefix": "DEV"
}
```

### Branch Structure
- Live runs on: `stable-added-adaptive-trad-n-chart-more`
- Test runs on: `development` (branched from stable)

### Git Repository Setup

#### Local Mac (where Claude runs)
```
/Users/chris/projects/python/btc/              (original, `gg btc`)
├── .git/                                      
├── Remote: github-ckalt.com:CKalt/bitstamp    
└── Branch: stable-added-adaptive-trad-n-chart-more

/Users/chris/projects/python/btc-testing/      (cloned today, `gg tst`)
├── .git/                                      
├── Remote: github-ckalt.com:CKalt/bitstamp    
└── Branch: development
```

#### Server (ck)
```
/home/chris/projects/bitstamp/                 (original, `gg btc`)
├── .git/
├── Remote: github-ckalt.com:CKalt/bitstamp
└── Branch: stable-added-adaptive-trad-n-chart-more

/home/chris/projects/bitstamp-testing/         (existed since Jul 1, `gg tst`)
├── .git/
├── Remote: github-ckalt.com:CKalt/bitstamp
└── Branch: development (reset today from stable)
```

### Where Things Run

#### Claude Code (YOU!)
- **Runs on**: Your Mac in `screen -S claude-tdr`
- **Working directory**: `/Users/chris/projects/python/btc`
- **Can access**: Both btc and btc-testing via scripts
- **Cannot cd to**: Parent directories (security restriction)

#### Trading Servers
- **Live server**: Runs on `ck` in `screen -S server`
- **Test server**: Will run on `ck` in `screen -S server-tst`
- **Price feed**: Runs on `ck` in `screen -S btc`

#### Trading Clients
- **Live client**: Runs on Mac in `screen -S client-tdr`
- **Test client**: Will run on Mac in `screen -S client-tst`

### Git Workflow
1. **Development**: Make changes in test directories (`gg tst`)
2. **Test locally**: Run client against test server
3. **Commit**: Push to `development` branch
4. **Deploy to server**: `git pull` on server test directory
5. **Verify**: Test with 0.001 BTC positions
6. **Promote**: Cherry-pick or merge to stable branch
7. **Deploy to live**: Pull stable branch in live directories

### Architecture Overview

```
┌─────────────── YOUR MAC ─────────────────┐     ┌─────────────── SERVER (ck) ─────────────────┐
│                                          │     │                                             │
│  Claude Code (YOU are here!)             │     │  Price Feed (shared)                        │
│  ┌────────────────────────────┐         │     │  ┌────────────────────────────┐            │
│  │ screen -S claude-tdr       │         │     │  │ screen -S btc             │            │
│  │ cd /Users/.../python/btc   │         │     │  │ cd /home/.../bitstamp     │            │
│  │ (can run scripts & tools)  │         │     │  │ python src/websock-ticker2│            │
│  └────────────────────────────┘         │     │  │ writes → btcusd.log       │            │
│                                          │     │  └────────────────────────────┘            │
│  Live Client                             │     │                                             │
│  ┌────────────────────────────┐         │     │  Live Server                                │
│  │ screen -S client-tdr       │ SSH     │     │  ┌────────────────────────────┐            │
│  │ `gg btc`                  │─────────┼─────┼─▶│ screen -S server          │            │
│  │ python src/tdr.py          │ :4000   │     │  │ `gg btc`                  │            │
│  │ (connects to server:4000)  │         │     │  │ python src/tdr.py --server│            │
│  └────────────────────────────┘         │     │  │ Port: 4000                │            │
│                                          │     │  └────────────────────────────┘            │
│  Test Client (to create)                 │     │                                             │
│  ┌────────────────────────────┐         │     │  Test Server (to create)                    │
│  │ screen -S client-tst       │ SSH     │     │  ┌────────────────────────────┐            │
│  │ `gg tst`                  │─────────┼─────┼─▶│ screen -S server-tst      │            │
│  │ python src/tdr.py          │ :4002   │     │  │ `gg tst`                  │            │
│  │ (connects to server:4002)  │         │     │  │ ./start-dev.sh            │            │
│  └────────────────────────────┘         │     │  │ Port: 4002                │            │
│                                          │     │  └────────────────────────────┘            │
│                                          │     │         ▲                                   │
│  Git Repos:                              │     │         │ symlink                          │
│  btc/        → stable branch             │     │         └── btcusd.log                     │
│  btc-testing/ → development branch       │     │                                             │
└──────────────────────────────────────────┘     └─────────────────────────────────────────────┘

Tunnels (autossh): 4000, 8050, 4002, 8051
```

### Starting Test Server
```bash
ssh ck
`gg tst`
./start-dev.sh
```

Then start trading:
```bash
curl -X POST http://localhost:4002/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "auto_trade 0.001btc MA short=6 long=34 do_live_trades=True hist_position=long"}'
```

### Key Scripts Created
- `/home/chris/projects/bitstamp-testing/start-dev.sh` - Start test server
- `/home/chris/projects/bitstamp-testing/check-status.sh` - Check both systems
- `/home/chris/projects/bitstamp-testing/monitor-all.sh` - Monitor everything

### Important Files with Credentials
- `.bitstamp` files exist in both live and test directories
- NEVER read or expose these files - they contain API credentials

## ✅ LIVE SYSTEM STATUS (Updated 2025-07-27 12:35 UTC)
- Position: SHORT (0 BTC / $170,366 USD)
- Entry: $118,011 (flipped at 09:00 UTC)
- Previous: LONG @ $116,526 → Profit: $2,142.90
- Server running cleanly with auto_trade active
- Auto-resume fixed to respect config setting
- Trade count today: 1/5

## 🧪 TEST SYSTEM STATUS
- Server STOPPED to conserve memory (only 3.7GB total)
- Was causing memory pressure alongside live server
- Can be restarted with limited history if needed

## CRITICAL CONTEXT
1. **System is NOT broken** - It's working correctly
2. **User runs server via screen, NOT systemctl**
3. **Development process** - Parallel implementation to prevent regressions

## KEY FILES & RECENT CHANGES

### 1. Enhanced Logging Added
**File**: `/Users/chris/projects/python/btc/src/tdr_core/strategies.py`
**Changes**: Added critical trade decision logging
```python
# CRITICAL TRADE DECISION LOG (line ~847)
self.logger.warning(f"🎯 TRADE DECISION: Signal={latest_signal} vs Position={self.position} | "
                  f"Will trade? {(latest_signal == 1 and self.position <= 0) or (latest_signal == -1 and self.position >= 0)} | "
                  f"Live={self.live_trading} | Today's trades={self.trade_count_today}/{self.max_trades_per_day}")
```

### 2. Development Process Documentation
**File**: `/Users/chris/projects/python/btc/prompts/dev-plan.md`
- Comprehensive development process to prevent regressions
- Feature flags, shadow mode, progressive rollout
- System verifier for continuous validation

### 3. Key Configuration
**File**: `best_strategy.json`
- `do_live_trades: true` (MUST be true for trades to execute)
- `auto_resume: true` (maintains position across restarts)
- `ma_separation_threshold: 0.3` (trigger threshold)

## HOW TO RESUME

### 1. Check Current System State
```bash
ssh ck
# Live system
`gg btc`
screen -ls
tail -50 logs/tdr_server.log | grep -E "🎯|🚨|SIGNAL_EVAL"

# Test system
`gg tst`
./check-status.sh
```

### 2. If Server Crashed - Manual Restart Procedure
```bash
# For live
`gg btc`
python src/tdr.py --server
# Wait for "Historical data loaded successfully" (3-5 minutes)
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "auto_trade 1.444btc MA short=6 long=34 do_live_trades=True hist_position=long"}'

# For test
`gg tst`
./start-dev.sh
curl -X POST http://localhost:4002/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "auto_trade 0.001btc MA short=6 long=34 do_live_trades=True hist_position=long"}'
```

## DEVELOPMENT WORKFLOW
1. Make changes in test directory (`gg tst`)
2. Test with 0.001 BTC position
3. Monitor for errors and correct behavior
4. Once verified, cherry-pick or merge to live
5. Deploy to live with confidence

## RECENT ISSUES FIXED
1. **Entry price calculation bugs** - Calculated in 5+ places
2. **Position tracking complexity** - Dual tracking system
3. **Strategy interface mismatches** - Adaptive vs Pure MA
4. **Auto-resume forcing True** - Ignores config setting
5. **Tunnel configuration** - Fixed by running setup-tunnels.sh

## AUTO-RESUME SYSTEM (FIXED!)

### How Auto-Resume Works
1. **Saves position** to `resume-auto-trade.json` after each trade
2. **Loads position** on server restart if `auto_resume: true` in config
3. **Key data saved**: position type, size, entry price, balances, trade refs

### When Auto-Resume Triggers
- **Saves**: After trades, when starting/stopping auto_trade, periodically
- **Loads**: After historical data loads on server start (if enabled)
- **Command**: `resume_auto_trade 1.444btc long 116526` (example)

### Auto-Resume Fix Applied
- Fixed forcing of `auto_resume = True` in two files:
  - `tdr_server.py` line 219: Check config before auto-resuming
  - `tdr_server_standalone.py` line 132: Respect config setting
- Now properly respects `auto_resume: false` in best_strategy.json

## CURRENT BUGS TO FIX (NOT URGENT)
1. **Resume file validation** - Accepts USD amounts for LONG positions
2. **Strategy type mismatch** - Saves wrong strategy type in resume
3. **Position tracking** - Need single source of truth

## DEVELOPMENT GOALS
- Parallel implementation (never modify working code directly)
- Feature flags for everything
- Shadow mode testing before production
- Progressive rollout (10% → 25% → 50% → 100%)
- System verification to catch regressions

## EMERGENCY PROCEDURES
```bash
# Stop live trading
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "stop_auto_trade"}'

# Stop test trading
curl -X POST http://localhost:4002/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "stop_auto_trade"}'

# Kill screens
screen -S [name] -X quit
```

## REMOTE MONITORING COMMAND
```bash
# Live system
ssh ck "cd /home/chris/projects/bitstamp && tail -20 logs/tdr_server.log | grep -E '🎯|🚨|Executing trade'"

# Test system
ssh ck "cd /home/chris/projects/bitstamp-testing && tail -20 logs/tdr_server_dev.log | grep -E '🎯|🚨|Executing trade|DEV'"
```

## KEY UNDERSTANDING
- System is 100% flip strategy (always LONG or SHORT, never neutral)
- Test environment trades with real money but limited to 0.001 BTC
- Both use same price feed for accurate testing
- Development branch allows safe experimentation

## NON-OBVIOUS CONTEXT THAT MATTERS

### Server Access & Inspection
1. **Servers run on remote host `ck`** - NOT locally
   - Access via: `ssh ck`
   - Live server runs in: `screen -S server`
   - Price feed runs in: `screen -S btc`
   
2. **Tunnels make remote ports appear local**
   - `localhost:4000` → `ck:4000` (live server)
   - `localhost:4002` → `ck:4002` (test server)
   - Managed by autossh, restart with: `fix-tunnels`

3. **Screen sessions on ck, NOT local**
   - Check screens: `ssh ck "screen -ls"`
   - View live logs: `ssh ck "tail -f /home/chris/projects/bitstamp/logs/tdr_server.log"`

### Memory Constraints
- **Server has only 3.7GB RAM total**
- Live server uses ~1.5GB after loading 4.4GB historical data
- **OOM killer likely cause of crashes**
- Test server must be stopped when memory tight

### Manual Trading Operations
- **NO systemctl** - User runs everything via screen
- Server restart is MANUAL process:
  1. Start server: `python src/tdr.py --server`
  2. Wait 3-5 min for data load
  3. Manually start auto_trade with position info
- **auto_resume was set to false** - required manual intervention

### Critical File Locations
- **On Mac (Claude)**: `/Users/chris/projects/python/btc`
- **On Server (ck)**: `/home/chris/projects/bitstamp`
- **Price feed**: `/home/chris/projects/bitstamp/btcusd.log` (4.4GB)
- **Both share same feed** via symlink in test directory

## SESSION SUMMARY (2025-07-27)
- Fixed auto-resume forcing True bug in both server files
- Created comprehensive System Verifier for regression detection
- Integrated System Verifier into both strategy classes
- Fixed tunnel configuration by running setup-tunnels.sh
- Test server running successfully on port 4002
- System Verifier will check every 30 seconds for:
  - Position consistency
  - Entry price sanity
  - Balance integrity
  - Cost basis accuracy
  - Trade execution issues
  - Data synchronization

## FINAL NOTE
This dual-version setup prevents the regression bugs that have plagued the system. Test with tiny amounts, verify behavior, then deploy to live with confidence.