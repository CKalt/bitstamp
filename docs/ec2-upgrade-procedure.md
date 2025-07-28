# EC2 Instance Upgrade Procedure (t3.large)

## Current Position to Resume
- **Position**: LONG 1.437 BTC (per Bitstamp account) ✅ VERIFIED
- **Entry Price**: $118,175 (from trades.json: $118,166)
- **MA Strategy**: 6/34 ✅ NOW RUNNING CORRECTLY
- **MA Status**: MA6=$118,470 MA34=$118,136 (0.28% proximity, DIVERGING)
- **Trade Risk**: VERY LOW - MAs moving apart, strengthening LONG signal
- **Resume Command**: Server has auto_resume=true (forced), will auto-resume

## Complete Shutdown & Upgrade Steps

### 1. Pre-Upgrade: Graceful Shutdown (on server)

```bash
# From Mac client-tdr screen:
stop_auto_trade
status

# Then SSH to server:
ssh ck
gg btc

# Verify position saved
cat resume-auto-trade.json
cp resume-auto-trade.json resume-auto-trade.json.before_upgrade

# Stop the trading server
screen -S server -X quit

# Stop the price feed
screen -S btc -X quit

# Verify screens are stopped
screen -ls  # Should show no screens
```

### 2. AWS Console: Upgrade Instance

1. **Login to AWS Console** → EC2 → Instances
2. **Stop Instance**: Select → Actions → Instance State → Stop (wait for "Stopped")
3. **Change Type**: Actions → Instance Settings → Change Instance Type → Select "t3.large" → Apply
4. **Start Instance**: Actions → Instance State → Start (wait for "Running")

### 3. Post-Upgrade: Restart Services (on server)

```bash
ssh ck
gg btc

# Verify memory upgrade
free -h  # Should show ~8GB total

# Start price feed first
screen -dmS btc bash -c 'source source-venv.sh && python src/websock-ticker2.py'

# Start server
screen -dmS server bash -c 'source source-venv.sh && python src/tdr.py --server'

# Monitor startup (wait for history to load)
tail -f logs/tdr_server.log
# Wait for: "Historical data loaded successfully"
# Then: "Auto-trading resumed successfully"
# Press Ctrl+C to exit tail

# Check MA values are correct (should show MA6 and MA34, not MA12/MA36)
tail -20 logs/tdr_server.log | grep 'SIGNAL_EVAL'
# Should see: MA6=... MA34=... (NOT MA12=... MA36=...)

# Exit server SSH
exit

# From Mac client-tdr screen:
status
# Should show: auto_trader active, position LONG 1.437 BTC
```

### 4. Verify from Client Side

**From Mac client-tdr screen:**
```bash
# Check system status
status

# View recent trades and signals
view_recent_trades

# If auto-resume failed (unlikely), manually resume:
resume_auto_trade 1.437btc long 118175
```

### 5. Verify Correct MA Configuration

**CRITICAL**: Ensure the system is using MA 6/34, not MA 12/36:
```bash
# SSH to server to check logs
ssh ck "cd /home/chris/projects/bitstamp && tail -50 logs/tdr_server.log | grep 'SIGNAL_EVAL' | tail -5"
# Should show: MA6=... MA34=... (NOT MA12=... MA36=...)
```

## Total Expected Downtime: 10-15 minutes
- EC2 stop/start: 3-5 minutes
- History loading: 3-5 minutes
- Verification: 2-5 minutes

## Important Notes
- The system is currently LONG 1.437 BTC @ $118,175 ✅ VERIFIED
- MA configuration FIXED and RUNNING: MA 6/34 ✅ CONFIRMED IN LOGS
- Both best_strategy.json and resume-auto-trade.json corrected
- Server forces auto_resume=true regardless of config
- **Trade Risk Assessment**: MAs diverging (0.28%), very low crossover risk
- If MAs have crossed during downtime, system will trade after resuming