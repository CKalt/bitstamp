# Server Restart Steps After Bug Fix

## Current Status
- Bug fixed: validate_position_tracking error
- Position is correctly loaded as SHORT @ $117,564
- MA Proximity: 0.75% (away from 0.3% trigger)
- Profit: +$3,519.95

## Steps to Apply Fix

### 1. On the Server - Pull the Fix
```bash
ssh ck
cd /home/chris/projects/bitstamp
git pull
```

### 2. Verify Current Position Before Restart
```bash
# Check current status
curl -s http://localhost:4000/api/status | jq '.position'

# Save output - should show:
# position: -1 (SHORT)
# entry_price: 117564.0
```

### 3. Restart the Server
```bash
# If using systemd:
sudo systemctl restart tdr-server

# If using screen:
screen -r tdr
# Ctrl+C to stop
# Wait for "Shutdown complete"
# Restart: python3 src/tdr.py --server
```

### 4. Wait and Verify (CRITICAL)
```bash
# Wait 30 seconds for full initialization
sleep 30

# Check position loaded correctly
curl -s http://localhost:4000/api/status | jq '.position'

# Verify via command API
curl -s -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "status"}' | grep -A10 "Position Details"
```

Must show:
- Direction: Short
- Entry Price: $117,564
- Position: -1

### 5. Verify Enhanced Logging Works
```bash
# Check logs for SIGNAL_EVAL entries
tail -f logs/tdr_server.log | grep -E "SIGNAL_EVAL|CHECK_FOR_SIGNALS"
```

Should see entries like:
```
📊 SIGNAL_EVAL: MA4=X MA20=Y Diff=Z Prox=0.75% Sig=-1 Pos=-1 Action=NO_TRADE
```

### 6. Start Monitoring
Once verified, proceed with monitoring setup from docs/monitoring-setup-howto.md

## If Issues Occur

1. **If position shows as NEUTRAL after restart**:
   - STOP immediately
   - Check resume-auto-trade.json still exists
   - Verify auto_resume is still true in best_strategy.json

2. **If new errors appear**:
   - Check git pull succeeded
   - Verify the fix is in the code: `grep -A2 validate_position_tracking src/tdr_core/strategies.py`

3. **If logging still doesn't work**:
   - Verify verbose_logging and log_signal_evaluation are true
   - Check logs directory permissions

## Success Checklist
- [ ] Git pull successful
- [ ] Server restarted cleanly
- [ ] Position still SHORT @ $117,564
- [ ] No validate_position_tracking errors
- [ ] SIGNAL_EVAL logs appearing every 30 seconds
- [ ] MA proximity showing ~0.75%