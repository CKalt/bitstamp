# Manual Resume Process for Live Server

This document describes the manual process for resuming trading on the live server, which is necessary due to auto-resume reliability issues.

## Why Manual Resume?

Auto-resume has proven unreliable due to:
- Incorrect entry price calculations for multi-part trades
- Position/unit mismatches (e.g., USD amount for LONG positions)
- Stale or corrupted resume files
- Silent failures during resume

## Pre-Resume Checklist

1. **Verify Configuration**
   ```bash
   ssh ck
   gg btc
   cat best_strategy.json
   ```
   - Confirm MA parameters match your intended strategy
   - Check `do_live_trades: true`
   - Verify `max_trades_per_day` setting

2. **Check Current Market Position**
   ```bash
   # On your local machine (uses SSH tunnel)
   curl -s http://localhost:4000/api/status | jq '.last_price'
   ```

3. **Review Last Position**
   ```bash
   # On server
   tail -20 logs/tdr_server.log | grep -E "Executing trade|Position:|Entry"
   
   # Check trades.json for actual trades
   tail trades.json | jq '.'
   ```

## Manual Resume Steps

### 1. Connect to Server
```bash
ssh ck
gg btc
screen -r server
```

### 2. Stop Current Server (if running)
- Press `Ctrl-C` to stop the server
- Wait for clean shutdown message

### 3. Pull Latest Code
```bash
git pull
```

### 4. Start Server
```bash
python src/tdr.py --server
```
- Wait for "Historical data loaded successfully" message (3-5 minutes)
- Look for any error messages during startup

### 5. Manually Resume Trading

#### For LONG Position
```bash
# From another terminal or after detaching from screen (Ctrl-A D)
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{
    "command": "auto_trade 1.25btc long"
  }'
```

#### For SHORT Position
```bash
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{
    "command": "auto_trade 161000usd short"
  }'
```

### 6. Verify Resume Success
```bash
# Check status
curl -s http://localhost:4000/api/status | jq '{
  position: .position.position,
  size: .position.position_size,
  entry: .position.entry_price,
  strategy: .strategy
}'

# Monitor heartbeats (new window)
ssh ck "tail -f ~/projects/bitstamp/logs/tdr_server.log | grep -E 'HEARTBEAT|SIGNAL_EVAL|ERROR'"
```

## Important Position Values

When resuming, you need:
1. **Position Direction**: `long` or `short`
2. **Amount**: 
   - For LONG: BTC amount (e.g., `1.25btc`)
   - For SHORT: USD amount (e.g., `161000usd`)
3. **Entry Price** (optional): Can be calculated from trades.json

## Common Issues and Solutions

### Issue: "Auto-trading is already running"
**Solution**: The server thinks trading is active. Stop it first:
```bash
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "stop_auto_trade"}'
```

### Issue: Wrong MA Configuration Loaded
**Solution**: 
1. Stop server
2. Update `best_strategy.json` on server
3. Restart and resume

### Issue: Position Mismatch Error
**Solution**: Check actual trades in `trades.json` and use correct values

### Issue: Server Crashes After Resume
**Solution**: 
1. Check logs for specific error
2. Verify valid position data
3. Consider removing `resume-auto-trade.json` if corrupted

## Emergency Stop

To immediately stop all trading:
```bash
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "stop_auto_trade"}'
```

## Best Practices

1. **Always verify configuration** before resuming
2. **Check recent trades** to confirm position details
3. **Monitor heartbeats** after resume to ensure strategy loop is running
4. **Keep notes** of your actual position for manual resume
5. **Create backups** of resume files before major changes

## Auto-Resume Validation (After Fixes)

With the recent fixes, auto-resume now includes validation:
- Checks position/unit consistency (LONG must use BTC, SHORT must use USD)
- Validates amounts are positive and reasonable
- Creates backups of resume files
- Logs validation errors clearly

If auto-resume fails validation, you'll see:
```
Resume data validation failed: [specific errors]
Skipping auto-resume due to invalid data
Please manually resume with: auto_trade <amount><btc|usd> <long|short>
```

## Quick Reference Card

```bash
# SSH to server
ssh ck && gg btc

# Check status
curl -s http://localhost:4000/api/status | jq '.'

# Resume LONG
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "auto_trade 1.25btc long"}'

# Resume SHORT  
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "auto_trade 161000usd short"}'

# Monitor
ssh ck "tail -f ~/projects/bitstamp/logs/tdr_server.log | grep HEARTBEAT"

# Emergency stop
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "stop_auto_trade"}'
```