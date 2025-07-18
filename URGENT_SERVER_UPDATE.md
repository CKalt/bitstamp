# URGENT: Disable Pivot Protection on Server

## Action Required

The pivot protection system is causing excessive losses. Please update the server immediately:

### On the server (chriskoin):

1. Edit the configuration:
```bash
cd /home/chris/projects/bitstamp/
vim best_strategy.json
```

2. Change this line:
```json
"enable_pivot_protection": true,
```

To:
```json
"enable_pivot_protection": false,
```

3. Restart the server:
```bash
bin/server_control.sh restart
```

## Why This is Urgent

The backtest revealed:
- **44 trades in 24 hours** (excessive churning)
- **2.3% win rate** (losing 97.7% of trades!)
- **-6.87% loss in one day**
- Pivot buffer of $100 is too tight, causing whipsaws

## What This Does

- Disables pivot-triggered position flips
- Returns to pure MA crossover and adaptive strategy signals
- Stops the bleeding while we optimize parameters

## Next Steps

We'll use backtesting to tune pivot parameters:
- Test wider buffers ($200-500)
- Adjust profit tiers
- Find optimal settings before re-enabling

**This change is critical to stop losses immediately!**