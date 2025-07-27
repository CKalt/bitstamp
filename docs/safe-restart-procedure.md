# Safe Server Restart Procedure

## CRITICAL ISSUES FOUND:

1. **`auto_resume: false`** in best_strategy.json - Server won't load position!
2. **Strategy mismatch** - Resume shows "AdaptiveMultiStrategy" but config shows "MA"
3. **Must fix BEFORE restart** or position will be lost

## Step-by-Step Safe Restart Procedure

### Step 1: Backup Current State

```bash
ssh ck
cd /home/chris/projects/bitstamp

# Backup all critical files
cp best_strategy.json best_strategy.json.backup.$(date +%Y%m%d_%H%M%S)
cp resume-auto-trade.json resume-auto-trade.json.backup.$(date +%Y%m%d_%H%M%S)
cp trades.json trades.json.backup.$(date +%Y%m%d_%H%M%S)
```

### Step 2: Fix Configuration Issues

```bash
# 1. Enable auto_resume in best_strategy.json
sed -i 's/"auto_resume": false/"auto_resume": true/' best_strategy.json

# 2. Enable logging settings
sed -i 's/"log_signal_evaluation": false/"log_signal_evaluation": true/' best_strategy.json
sed -i 's/"verbose_logging": false/"verbose_logging": true/' best_strategy.json

# 3. Verify changes
grep -E "auto_resume|log_signal|verbose" best_strategy.json
```

Expected output:
```
"auto_resume": true,
"log_signal_evaluation": true,
"verbose_logging": true,
```

### Step 3: Fix Resume File Strategy Type

The resume file shows "AdaptiveMultiStrategy" but config uses "MA". Update it:

```bash
# Edit resume file to match current strategy
sed -i 's/"type": "AdaptiveMultiStrategy"/"type": "MACrossoverStrategy"/' resume-auto-trade.json

# Verify
grep -A5 "strategy" resume-auto-trade.json
```

### Step 4: Verify Position Before Restart

```bash
# Check current server status
curl -s http://localhost:4000/api/status | jq '.position'
```

Should show:
```json
{
  "btc_balance": 0.0,
  "entry_price": 117564.0,
  "position": -1,
  "position_size": -1.4509288557721751,
  "usd_balance": 10000.0
}
```

### Step 5: Commit Code Changes

```bash
# Add the enhanced logging changes
git add src/tdr_core/strategies.py
git commit -m "Add enhanced signal evaluation logging for monitoring"
git push
```

### Step 6: Stop Server Gracefully

```bash
# Check if using systemd
sudo systemctl status tdr-server

# If systemd:
sudo systemctl stop tdr-server

# If running in screen:
screen -r tdr
# Press Ctrl+C to stop gracefully
# Wait for "Shutdown complete" message
```

### Step 7: Verify Files One More Time

```bash
# Final check before restart
echo "=== CRITICAL SETTINGS CHECK ==="
echo "auto_resume: $(grep auto_resume best_strategy.json | grep -o 'true\|false')"
echo "do_live_trades: $(grep do_live_trades best_strategy.json | grep -o 'true\|false')"
echo "Position in resume: $(grep position resume-auto-trade.json | head -1)"
echo "Strategy type: $(grep strategy_type best_strategy.json)"
```

Must show:
- auto_resume: true
- do_live_trades: true  
- Position: "SHORT"
- strategy_type: "MA"

### Step 8: Start Server

```bash
# If using systemd:
sudo systemctl start tdr-server
sleep 5
sudo systemctl status tdr-server

# If using screen:
screen -S tdr
cd /home/chris/projects/bitstamp
python3 src/tdr.py --server
# Wait for "AUTO-RESUME: Loaded position from resume-auto-trade.json"
# Detach: Ctrl+A, D
```

### Step 9: Verify Resume Worked

**CRITICAL**: Check position was loaded correctly:

```bash
# Wait 30 seconds for server to fully initialize
sleep 30

# Check status
curl -s http://localhost:4000/api/status | jq '.position'

# Should still show position = -1 (SHORT)

# Check via command API
curl -s -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "status"}' | grep -A10 "Position Details"
```

Must show:
- Direction: Short
- Entry Price: $117,564
- BTC: 0.0
- USD: $10,000 (reserve) + $170,577 (position)

### Step 10: Verify Trading Active

```bash
# Check auto-trader is active
curl -s http://localhost:4000/api/status | jq '.auto_trader'
```

Should show:
```json
{
  "active": true,
  "strategy": "MACrossoverStrategy",
  "trades_today": 0
}
```

### Step 11: Check Enhanced Logging Works

```bash
# Should see SIGNAL_EVAL entries every 30 seconds
tail -f logs/tdr_server.log | grep SIGNAL_EVAL
```

Wait 30-60 seconds, should see:
```
📊 SIGNAL_EVAL: MA4=X MA20=Y Diff=Z Prox=W% Sig=-1 Pos=-1 Action=NO_TRADE
```

## If Resume Fails

If position shows as NEUTRAL (0) instead of SHORT (-1):

```bash
# DO NOT PANIC - We can fix it

# Stop the server immediately
sudo systemctl stop tdr-server

# Check the backup files
ls -la *.backup.*

# Restore and try again with correct settings
```

## Checklist Before Restart

- [ ] Backed up all critical files
- [ ] Set auto_resume: true
- [ ] Enabled enhanced logging
- [ ] Fixed strategy type in resume file
- [ ] Committed code changes
- [ ] Verified current position

## Checklist After Restart

- [ ] Position shows as SHORT (-1)
- [ ] Entry price is $117,564
- [ ] Auto-trader is active
- [ ] SIGNAL_EVAL logs appearing
- [ ] MA proximity showing ~0.5%

Only proceed with monitoring setup after ALL checks pass!