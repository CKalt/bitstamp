# Dual Version Setup: Live + Development (0.001 BTC)

## Overview
Run two versions of the trading system simultaneously:
- **LIVE**: Full position trading (current system)
- **DEV**: Small position (0.001 BTC) for testing changes

Both versions share the same price feed via symlinked `btcusd.log` from the live version.

## Directory Structure

```
/home/chris/projects/
├── bitstamp/                    # LIVE version (gg btc)
│   ├── src/
│   ├── btcusd.log              # Real-time price feed (from websock-ticker2.py)
│   ├── logs/
│   │   └── tdr_server.log      # Live trading logs
│   ├── trades.json             # Live trades
│   ├── resume-auto-trade.json  # Live position
│   └── best_strategy.json      # Live config
│
└── bitstamp-testing/           # DEV version (gg tst)
    ├── src/                    # Same code, different branch
    ├── btcusd.log -> ../bitstamp/btcusd.log  # Symlink!
    ├── logs/
    │   └── tdr_server_dev.log  # Dev trading logs
    ├── trades_dev.json         # Dev trades
    ├── resume-auto-trade.json  # Dev position (0.001 BTC)
    └── best_strategy.json      # Dev config (small position)
```

## Initial Setup

### 1. Clone Development Version
```bash
cd /home/chris/projects
git clone /home/chris/projects/bitstamp bitstamp-dev
cd bitstamp-dev
git checkout -b development
```

### 2. Create Symlink for Price Feed
```bash
cd /home/chris/projects/bitstamp-dev
ln -s ../bitstamp/btcusd.log btcusd.log
# Verify it works
tail -5 btcusd.log
```

### 3. Configure Development Version
Edit `bitstamp-dev/best_strategy.json`:
```json
{
  ...existing config...,
  "trading_mode": "development",
  "max_position_btc": 0.001,
  "do_live_trades": true,
  "log_prefix": "DEV"
}
```

### 4. Separate Log Files
Modify `bitstamp-dev/src/tdr_server.py` to use different log:
```python
# In setup_logging()
log_file = 'logs/tdr_server_dev.log' if 'dev' in os.getcwd() else 'logs/tdr_server.log'
```

### 5. Separate Trade Files
Modify `bitstamp-dev/src/tdr_core/strategies.py`:
```python
# In __init__
if 'dev' in os.getcwd() or self.config.get('trading_mode') == 'development':
    self.trade_log_file = 'trades_dev.json'
else:
    self.trade_log_file = 'trades.json'
```

## Running Both Versions

### Terminal 1: Price Feed (Live Directory)
```bash
cd /home/chris/projects/bitstamp
screen -S ticker
python src/websock-ticker2.py
# Ctrl+A, D to detach
```

### Terminal 2: Live Trading
```bash
cd /home/chris/projects/bitstamp
screen -S live-trading
python src/tdr.py --server
# Wait for loading...
# Start trading with full position
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "auto_trade 1.444btc MA short=6 long=34 do_live_trades=True hist_position=long"}'
# Ctrl+A, D to detach
```

### Terminal 3: Dev Trading
```bash
cd /home/chris/projects/bitstamp-dev
screen -S dev-trading
# Run on different port
python src/tdr.py --server --port 4001
# Wait for loading...
# Start trading with tiny position
curl -X POST http://localhost:4001/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "auto_trade 0.001btc MA short=6 long=34 do_live_trades=True hist_position=long"}'
# Ctrl+A, D to detach
```

## Monitoring Both Versions

### Combined Status Script
Create `check-both-versions.sh`:
```bash
#!/bin/bash
echo "=== LIVE TRADING STATUS ==="
curl -s http://localhost:4000/api/command -H 'Content-Type: application/json' \
  -d '{"command": "status"}' | grep -E "Position:|Entry Price:|PnL:"

echo -e "\n=== DEV TRADING STATUS ==="
curl -s http://localhost:4001/api/command -H 'Content-Type: application/json' \
  -d '{"command": "status"}' | grep -E "Position:|Entry Price:|PnL:"

echo -e "\n=== RECENT SIGNALS ==="
echo "LIVE:" $(tail -1 /home/chris/projects/bitstamp/logs/tdr_server.log | grep SIGNAL_EVAL)
echo "DEV:"  $(tail -1 /home/chris/projects/bitstamp-dev/logs/tdr_server_dev.log | grep SIGNAL_EVAL)
```

## Development Workflow

### 1. Test Changes in Dev First
```bash
cd /home/chris/projects/bitstamp-dev
# Make code changes
git add -A
git commit -m "Test: new feature X"

# Restart dev server
screen -r dev-trading
# Ctrl+C, then restart
python src/tdr.py --server --port 4001
```

### 2. Monitor Dev Performance
```bash
# Watch dev trades
tail -f logs/tdr_server_dev.log | grep -E "Executing trade|ERROR"

# Compare entry prices
grep "Entry price" logs/tdr_server_dev.log | tail -5
```

### 3. Promote to Live After Verification
```bash
cd /home/chris/projects/bitstamp
git cherry-pick [commit-hash-from-dev]
# Or merge dev branch
git merge development
```

## Position Size Limits

Add safety check in `strategies.py`:
```python
def validate_trade_size(self, amount, trade_type):
    """Ensure dev version only trades small amounts."""
    if self.config.get('trading_mode') == 'development':
        max_btc = self.config.get('max_position_btc', 0.001)
        if trade_type == 'buy' and amount > max_btc:
            self.logger.warning(f"DEV: Limiting buy to {max_btc} BTC")
            return max_btc
    return amount
```

## Advantages of This Setup

1. **Real money testing** - 0.001 BTC trades test actual execution
2. **Same market data** - Both use identical price feed
3. **Safe experimentation** - Losses limited to ~$100
4. **Easy comparison** - Can diff logs between versions
5. **Quick rollback** - Just switch which version is "live"

## Emergency Procedures

### Stop Dev Trading
```bash
curl -X POST http://localhost:4001/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "stop_auto_trade"}'
```

### Kill Dev Server
```bash
screen -r dev-trading
# Ctrl+C
```

### Swap Dev to Live (if dev proves better)
```bash
cd /home/chris/projects
mv bitstamp bitstamp-old
mv bitstamp-dev bitstamp
# Update ports and restart
```

## Next Steps

1. Set up the development clone
2. Configure position size limits
3. Create monitoring scripts
4. Start dev version with 0.001 BTC
5. Test your first enhancement in dev

This gives you a real production testing environment with minimal risk!