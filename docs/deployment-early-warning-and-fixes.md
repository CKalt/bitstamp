# Complete Deployment Guide: Critical Fixes + Early Warning System

## CRITICAL: Resume File Bug Fix (Deploy First!)

The resume file hasn't been updating due to missing `last_trade_time` initialization.
This MUST be fixed first or system will resume with wrong position after restart.

### Step 1: Deploy Resume File Fix

```bash
# From your local machine
ssh ck
cd /home/chris/projects/bitstamp

# Pull the critical fix
git pull

# Find your server screen
screen -ls
screen -r [server-screen-name]

# Stop server (Ctrl+C)
# Restart server
python src/run_server.py

# Detach (Ctrl+A, D)
```

### Step 2: Verify Resume File Updates

```bash
# Check that resume file now updates
tail -f logs/tdr_server.log | grep -E "save resume|Failed to save"

# After a minute, check the file was updated
ls -la resume-auto-trade.json
cat resume-auto-trade.json | grep -E "position|timestamp" | head -3

# Should show:
# - timestamp: Recent (within last minute)  
# - position: "LONG" (current position)
```

### Step 3: Force Resume File Update

```bash
# Connect to server API to force save
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "save_resume_state"}'

# Verify it saved correctly
cat resume-auto-trade.json | python3 -m json.tool | head -20
```

## Early Warning System Deployment

### Step 4: Test Early Warning Locally First

```bash
# On your local machine
cd /Users/chris/projects/python/btc

# Run tests
python tests/test_early_warning.py

# Should see: "✅ ALL TESTS PASSED"
```

### Step 5: Integrate Early Warning (Safe)

```bash
# Still on local machine
python integrate_early_warning.py

# This will:
# 1. Create backup of strategies.py
# 2. Add early warning imports
# 3. Add monitor initialization  
# 4. Add monitor calls
# 5. Create toggle script

# Verify integration
grep -n "early_warning" src/tdr_core/strategies.py
```

### Step 6: Deploy to Server

```bash
# Commit the early warning integration
git add -A
git commit -m "Add early warning monitor (disabled by default)

- Monitor checks 5-min MAs for advance warning
- Completely separate from trading logic
- Disabled by default via config flag
- Can be toggled without restart"
git push

# On server
ssh ck
cd /home/chris/projects/bitstamp
git pull
```

### Step 7: Enable Early Warning (Optional)

```bash
# First run WITHOUT early warning for 24h to ensure stability

# When ready to test early warning:
python toggle_early_warning.py true

# Restart server in screen
screen -r [server-screen]
# Ctrl+C then restart
python src/run_server.py
# Ctrl+A, D

# Monitor early warning logs
tail -f logs/tdr_server.log | grep -E "EARLY WARNING|early_warning"
```

## Verification Checklist

### Critical Checks:
- [ ] Resume file has current timestamp (not July 23)
- [ ] Resume file shows position: "LONG" 
- [ ] No "Failed to save resume state" errors in logs
- [ ] Server running without errors

### Early Warning Checks (if enabled):
- [ ] See "Early Warning Monitor enabled" in logs
- [ ] No early warning errors
- [ ] Warning messages appear when MAs converge

## Monitoring Commands

```bash
# Check system status
curl -s http://localhost:4000/api/status | python3 -m json.tool

# Watch for resume saves
tail -f logs/tdr_server.log | grep -E "save resume|position_tracking"

# Monitor early warnings (if enabled)
tail -f logs/early_warning_analysis.log

# Check all critical events
tail -f logs/tdr_server.log | grep -E "🎯|🚨|EARLY WARNING|ERROR|save resume"
```

## Rollback Procedures

### If Resume File Issues:
```bash
# The fix is minimal, but if needed:
git checkout HEAD~1 src/tdr_core/strategies.py
git add src/tdr_core/strategies.py
git commit -m "Revert resume fix"
git push
```

### If Early Warning Issues:
```bash
# Method 1: Disable via config
python toggle_early_warning.py false
# Restart server

# Method 2: Full revert
cp src/tdr_core/strategies.py.backup_[timestamp] src/tdr_core/strategies.py
git add src/tdr_core/strategies.py
git commit -m "Revert early warning integration"
git push
```

## Current System State (as of deployment)

- Position: LONG (since 19:00 UTC / 15:00 EDT)
- Entry: $116,526
- Current: ~$117,300  
- PnL: +$1,118
- MA Proximity: ~0.90%
- Resume file: NEEDS UPDATE (showing old SHORT position)

## Priority Order

1. **URGENT**: Fix resume file bug - Without this, restart = wrong position
2. **Test**: Run system for a few hours to verify resume updates
3. **Optional**: Deploy early warning after confirming stability
4. **Monitor**: Use screen sessions to track both systems

The resume file bug is CRITICAL - it means the system wouldn't resume correctly after a restart!