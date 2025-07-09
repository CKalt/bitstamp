# Entry Price Bug Fix Rollout Guide

## Overview
This fix corrects the entry price tracking for SHORT positions in the TDR trading system.

## Current Status
- **Position**: SHORT (holding $166,415.76 USD)
- **Incorrect Entry**: $108,996
- **Auto-trader**: Running with AdaptiveMultiStrategy

## Deployment Steps

### 1. Prepare the Fix
First, commit the changes locally:
```bash
cd /Users/chris/projects/python/btc
git add src/tdr_core/strategies.py
git commit -m "Fix entry price tracking for SHORT positions

- Prevent position tracking reset when position_size near zero
- Properly restore SHORT position tracking using last_trade_price
- Improve LONG->SHORT transition logic with clear logging
- Ensure entry prices are preserved across position reversals"
```

### 2. Deploy to Server
Push to the remote and deploy on chriskoin:
```bash
# Push to remote repository
git push origin stable-added-adaptive-trad-n-chart-more

# SSH to the server
ssh chriskoin

# Navigate to the project
cd /home/chris/projects/bitstamp/
source env/bin/activate

# Pull the latest changes
git pull origin stable-added-adaptive-trad-n-chart-more
```

### 3. Stop Auto-Trader FIRST
**CRITICAL**: Stop the auto-trader before restarting the server:

```bash
# On the server, connect to the running instance
python src/tdr.py

# In the TDR shell
tdr> stop_auto_trade
tdr> exit
```

### 4. Restart the Server
```bash
# Still on chriskoin
# The server should be running in a screen/tmux session
# Find and restart it

# If using screen:
screen -r tdr
# Ctrl+C to stop the server
python src/tdr.py --server
# Ctrl+A, D to detach

# If using tmux:
tmux attach -t tdr
# Ctrl+C to stop the server
python src/tdr.py --server
# Ctrl+B, D to detach
```

### 5. Verify and Resume Auto-Trading
From your local machine with SSH tunnel active:

```bash
cd /Users/chris/projects/python/btc
source env/bin/activate
python src/tdr.py

tdr> enable_commands
tdr> status long
```

Check that:
1. Server is connected
2. Position shows as SHORT
3. Entry price should now show correctly (likely ~$108,572 based on your last sell)

### 6. Resume Auto-Trading
Since the position is SHORT with specific USD amount:

```bash
tdr> resume_auto_trade 166415.76usd short 108572
```

Or if the entry price now shows correctly in status:
```bash
tdr> resume_auto_trade
```

### 7. Monitor the Fix
After resuming:
```bash
tdr> status long
tdr> strategy_diagnostics
```

Verify:
- Entry price shows correctly (~$108,572)
- Position tracking is accurate
- Auto-trader is running
- Unrealized P&L calculations are correct

## Important Notes

1. **DO NOT** restart the server without first stopping auto-trading
2. The fix will restore position tracking using the last_trade_price when position_size is near zero
3. Future LONG->SHORT transitions will maintain proper entry price tracking
4. Check logs for "SHORT position tracking restored" message if position was recovered

## Rollback Plan
If issues occur:
```bash
# On server
git checkout HEAD~1 src/tdr_core/strategies.py
# Restart server and resume with previous logic
```

## Validation
The fix ensures:
- Entry prices persist across position reversals
- SHORT positions correctly track average entry price
- Position cost basis is maintained for accurate P&L calculations
EOF < /dev/null