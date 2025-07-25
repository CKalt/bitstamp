# Easy Screen Monitoring Setup

## Quick Setup (4 Screens)

### Screen 1: Signal Evaluation Logs
```bash
ssh ck
cd /home/chris/projects/bitstamp
screen -S tdr-logs
./claude-bin/screen-monitoring/screen-1.sh
# Detach: Ctrl+A, D
```

### Screen 2: MA Proximity Monitor
```bash
ssh ck
cd /home/chris/projects/bitstamp
screen -S tdr-proximity
./claude-bin/screen-monitoring/screen-2.sh
# Detach: Ctrl+A, D
```

### Screen 3: Trade Activity Monitor
```bash
ssh ck
cd /home/chris/projects/bitstamp
screen -S tdr-trades
./claude-bin/screen-monitoring/screen-3.sh
# Detach: Ctrl+A, D
```

### Screen 4: Error Monitor
```bash
ssh ck
cd /home/chi/projects/bitstamp
screen -S tdr-errors
./claude-bin/screen-monitoring/screen-4.sh
# Detach: Ctrl+A, D
```

## What Each Screen Shows

- **Screen 1**: SIGNAL_EVAL entries every 30 seconds (verify system is evaluating)
- **Screen 2**: Current proximity % and alerts when approaching trigger
- **Screen 3**: Alerts when trades execute with details
- **Screen 4**: Any errors or warnings (should be quiet if all is well)

## Managing Screens

List all screens:
```bash
screen -ls
```

Reattach to a screen:
```bash
screen -r tdr-logs     # or tdr-proximity, tdr-trades, tdr-errors
```

Kill a screen (when done):
```bash
screen -X -S tdr-logs quit
```

## Most Important Screens

If you only want 2 screens:
1. **tdr-proximity** (Screen 2) - Shows when approaching trigger
2. **tdr-trades** (Screen 3) - Alerts when trades happen

## Expected Output

Screen 1 (every 30 seconds):
```
[11:30:00] SIGNAL_EVAL: MA4=115658 MA20=116498 Diff=-840 Prox=0.72% Sig=-1 Pos=-1 Action=NO_TRADE
```

Screen 2 (every 30 seconds):
```
[11:30:00] Price: $115,297 | Proximity: 0.72% | PnL: +$3,289
```

Screen 3 (when trade executes):
```
🚨🚨🚨 NEW TRADE DETECTED! 🚨🚨🚨
Time: Fri Jul 25 12:00:00 EDT 2025
Trade details: ...
```