# How to Set Up System Monitoring and Verification

This guide walks you through setting up comprehensive monitoring to verify the TDR system is working correctly, especially as it approaches trade trigger points.

## Prerequisites
- TDR server is running on remote host
- You have SSH access to the server
- The monitoring scripts are in `claude-bin/screen-monitoring/`

## Easy Setup Using Screen Scripts

### Step 1: Push Monitoring Scripts to Server

From your local machine:
```bash
cd ~/projects/python/btc
git add claude-bin/screen-monitoring/
git commit -m "Add screen monitoring scripts"
git push
```

On the server:
```bash
ssh ck
cd /home/chris/projects/bitstamp
git pull
chmod +x claude-bin/screen-monitoring/*.sh
```

### Step 2: Set Up Monitoring Screens

#### Screen 1 - Signal Evaluation Logs
```bash
ssh ck
cd /home/chris/projects/bitstamp
screen -S tdr-logs
./claude-bin/screen-monitoring/screen-1.sh
# Detach: Ctrl+A, D
```

#### Screen 2 - MA Proximity Monitor
```bash
ssh ck
cd /home/chris/projects/bitstamp
screen -S tdr-proximity
./claude-bin/screen-monitoring/screen-2.sh
# Detach: Ctrl+A, D
```

#### Screen 3 - Trade Activity Monitor
```bash
ssh ck
cd /home/chris/projects/bitstamp
screen -S tdr-trades
./claude-bin/screen-monitoring/screen-3.sh
# Detach: Ctrl+A, D
```

#### Screen 4 - Error Monitor (Optional)
```bash
ssh ck
cd /home/chris/projects/bitstamp
screen -S tdr-errors
./claude-bin/screen-monitoring/screen-4.sh
# Detach: Ctrl+A, D
```

## Step 3: Managing Your Screens

### List All Running Screens
```bash
screen -ls
```

You should see:
```
There are screens on:
    12345.tdr-logs      (Detached)
    12346.tdr-proximity (Detached)
    12347.tdr-trades    (Detached)
    12348.tdr-errors    (Detached)
```

### Reattach to Any Screen
```bash
screen -r tdr-logs      # View signal evaluations
screen -r tdr-proximity # View MA proximity updates
screen -r tdr-trades    # View trade alerts
screen -r tdr-errors    # View errors/warnings
```

### Kill a Screen When Done
```bash
screen -X -S tdr-logs quit
```

## Step 4: What Each Screen Shows

### Screen 1 (tdr-logs) - Signal Evaluations
Shows every 30 seconds:
```
[11:30:00] SIGNAL_EVAL: MA4=115658 MA20=116498 Diff=-840 Prox=0.72% Sig=-1 Pos=-1 Action=NO_TRADE
[11:30:00] CHECK_FOR_SIGNALS: signal=-1, price=$115297, position=-1, time=2025-07-25 15:00:00, live=True
```

### Screen 2 (tdr-proximity) - MA Proximity
Shows every 30 seconds:
```
[11:30:00] Price: $115,297 | Proximity: 0.72% | PnL: +$3,289
[11:30:30] Price: $115,305 | Proximity: 0.71% | PnL: +$3,277
⚠️  APPROACHING TRIGGER ZONE!  (when < 0.4%)
🎯 IN TRIGGER ZONE! TRADE SHOULD EXECUTE!  (when ≤ 0.3%)
```

### Screen 3 (tdr-trades) - Trade Alerts
Quiet until a trade happens, then:
```
🚨🚨🚨 NEW TRADE DETECTED! 🚨🚨🚨
Time: Fri Jul 25 12:00:00 EDT 2025
Trade details: 
  "type": "BUY",
  "amount": 1.45092886,
  "price": 115300.00
```

### Screen 4 (tdr-errors) - Errors/Warnings
Should be quiet if everything is working. Shows:
```
❌ [11:30:00] ERROR - Connection lost to exchange
⚠️  [11:30:05] WARNING - Retrying connection
```

## Quick Reference - Most Important Screens

If you only want minimal monitoring, use these two:
1. **tdr-proximity** - Know when approaching trigger
2. **tdr-trades** - Know when trades execute

## Local Client Monitoring (Optional)

On your local machine, you can also run:
```bash
cd ~/projects/python/btc
./claude-bin/monitor_trigger_zone.sh
```

## Step 5: Verify Trade Execution

When you see trade alerts in the screens:

1. Check position changed:
   ```bash
   # From local machine
   ./claude-bin/check_ma_status.sh
   ```

2. Verify trades.json was updated:
   ```bash
   ssh ck "cd /home/chris/projects/bitstamp && tail -10 trades.json"
   ```

## Troubleshooting

### If No Signal Evaluations Appear

1. Check if auto-trader is active:
   ```bash
   curl -s http://localhost:4000/api/status | grep -A3 "auto_trader"
   ```

2. Restart monitoring on server:
   ```bash
   ssh ck
   screen -r tdr-logs
   # Ctrl+C to stop, then restart tail command
   ```

### If Trades Don't Execute at Trigger

Check for blockers:
```bash
# Recent errors
tail -20 logs/tdr_server.log | grep -i error

# Trade limits
grep -E "limit|grace period" logs/tdr_server.log | tail -10
```

## Summary Checklist

- [ ] Terminal 1: Server logs showing SIGNAL_EVAL every 30 seconds
- [ ] Terminal 2: Verification script confirming correct behavior  
- [ ] Terminal 3: Status monitor showing real-time updates
- [ ] Terminal 4: Ready for manual checks
- [ ] All monitors show consistent proximity values
- [ ] You understand what correct output looks like
- [ ] You know what alerts indicate trade execution

## When to Be Most Alert

- Proximity < 0.4% - Entering warning zone
- Proximity < 0.35% - Very close to trigger
- Proximity ≤ 0.3% - Should see trade execute
- Any time proximity is decreasing rapidly

Remember: The system evaluates every 30 seconds, so trades can happen at :00 or :30 past any minute, not just at hourly bar closes.