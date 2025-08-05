# Safe Testing Guide - NO RISK Approach

## Why This Testing Plan is Safe

1. **Paper Trading Only** - No real money at risk
2. **5-Minute Bars First** - See bugs quickly (288 signals/day vs 24)
3. **Isolated Testing** - Can run separately from main system
4. **Progressive Stages** - Start small, validate, then scale up

## Testing Stages

### Stage 1: Local 5-Minute Simulation (Day 1)
Run locally first to validate logic:

```bash
cd /Users/chris/projects/python/btc
python3 claude-bin/paper_test_5min.py
```

This will:
- Simulate 24 hours of 5-minute bars
- Test proximity threshold logic
- Show all "would have" trades
- Count proximity blocks
- Calculate theoretical savings

**What to Look For:**
- ✅ Proximity blocks working (should see "NO_TRADE_PROXIMITY")
- ✅ Reasonable block rate (30-50% in choppy market)
- ✅ No errors or crashes
- ✅ Trades only when proximity > 0.5%

### Stage 2: Server 5-Minute Paper Trading (Day 2-3)
Deploy to server with real data but paper trading:

```bash
# Deploy 5-minute test config
scp /tmp/stage1_5min_test.json ck:/home/chris/projects/bitstamp/best_strategy.json

# Start server (paper mode)
ssh ck
cd /home/chris/projects/bitstamp
screen -S paper-test python src/tdr_server.py

# Resume with SHORT position (your last position)
python src/tdr.py
TDR> resume_auto_trade 0btc short 113793
```

Monitor with:
```bash
./claude-bin/monitor_5min_test.sh
```

**What to Look For:**
- ✅ ~12 evaluations per hour (every 5 min)
- ✅ Proximity blocks preventing bad trades
- ✅ Clear logging of would-have trades
- ✅ No system errors

### Stage 3: Hourly Paper Trading (Day 4-5)
Switch to production-like hourly bars:

```bash
# Deploy hourly test config
scp /tmp/stage2_hourly_test.json ck:/home/chris/projects/bitstamp/best_strategy.json

# Restart server
# Monitor for 48 hours
```

**What to Look For:**
- ✅ Only 1 evaluation per hour
- ✅ Proximity threshold still working
- ✅ Daily trade limits respected
- ✅ Overnight behavior correct

## Safety Checklist

Before going live, confirm:

- [ ] Ran 5-minute tests for 24+ hours
- [ ] Ran hourly tests for 48+ hours  
- [ ] Proximity threshold blocked bad trades
- [ ] No system errors or crashes
- [ ] Theoretical P&L looks reasonable
- [ ] Ready to risk real money again

## Monitoring Commands

```bash
# Check current status
./claude-bin/check_ma_status.sh

# Watch paper trades in real-time
./claude-bin/monitor_5min_test.sh

# Analyze results
ssh ck 'python3 /home/chris/projects/bitstamp/claude-bin/monitor_paper_trades.py'

# Check for errors
ssh ck 'grep -i error /home/chris/projects/bitstamp/logs/tdr_server.log | tail -20'
```

## Emergency Stop

If anything goes wrong:
```bash
# Stop the server
ssh ck 'pkill -f python.*tdr_server'

# Verify it's stopped
ssh ck 'ps aux | grep tdr_server'
```

## When to Go Live

Only go live when ALL of these are true:
1. ✅ Both test stages completed successfully
2. ✅ Proximity threshold prevented excessive trading
3. ✅ No bugs or errors found
4. ✅ Market conditions have improved (less chop)
5. ✅ You're confident in the system

Remember: **There's no rush**. Better to test thoroughly than lose money to bugs.