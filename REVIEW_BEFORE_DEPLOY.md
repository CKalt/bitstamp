# Review Before Deployment

## Changes Made

### 1. **Proximity Threshold (Already Active)**
- Added 0.5% threshold to prevent flipping when MAs are too close
- Would have prevented your $2,000 loss from excessive trading
- Only trades when MA4 and MA20 differ by >$570 at current prices

### 2. **Paper Trading Mode**
- When `do_live_trades: false`, system will:
  - Process real market data
  - Calculate all signals
  - Log "🧪 PAPER TRADE" messages
  - NOT place any real orders
  - Track theoretical P&L

### 3. **5-Minute Candle Support**
- Added `candle_interval` configuration option
- Can test with 5-minute bars (12x faster feedback)
- Default remains hourly for production
- Allows rapid bug detection

### 4. **Enhanced Logging**
- Clear warnings when in paper mode
- Shows exactly what trades WOULD happen
- Calculates fees and theoretical impact
- Tracks proximity blocks

## Test Configurations Created

### Stage 1: 5-Minute Testing (`/tmp/stage1_5min_test.json`)
```json
{
  "Short_Window": 4,
  "Long_Window": 20,
  "do_live_trades": false,    // PAPER MODE
  "candle_interval": "5min",   // 5-MINUTE BARS
  "proximity_threshold": 0.5,
  "max_trades_per_day": 100
}
```

### Stage 2: Hourly Testing (`/tmp/stage2_hourly_test.json`)
```json
{
  "Short_Window": 4,
  "Long_Window": 20,
  "do_live_trades": false,    // STILL PAPER MODE
  "candle_interval": "1h",     // HOURLY BARS
  "proximity_threshold": 0.5,
  "max_trades_per_day": 10
}
```

## Deployment Steps

### 1. Pull Changes on Server
```bash
ssh ck 'cd /home/chris/projects/bitstamp && git pull'
```

### 2. Deploy Stage 1 Config (5-min test)
```bash
scp /tmp/stage1_5min_test.json ck:/home/chris/projects/bitstamp/best_strategy.json
```

### 3. Start Server
```bash
ssh ck
cd /home/chris/projects/bitstamp
screen -S paper-test python src/tdr_server.py
```

### 4. Resume with Your Position
```bash
# On your Mac
cd /Users/chris/projects/python/btc
./env/bin/python src/tdr.py
TDR> resume_auto_trade 0btc short 113793
```

### 5. Monitor
```bash
# Watch paper trades
./claude-bin/monitor_5min_test.sh

# Or check status
./claude-bin/check_ma_status.sh
```

## What You'll See

### In Server Logs
```
🧪 PAPER TRADING MODE - No real trades will be executed
🕐 NEW 5-MIN CANDLE: 2025-08-05 10:15:00
📊 SIGNAL_EVAL v2: MA4=112800 MA20=113200 Diff=-400 Prox=0.35% Sig=-1 Pos=-1 Action=NO_TRADE_PROXIMITY
💡 MAs too close: 0.35% <= 0.5% threshold
```

### When Trade Would Happen
```
🧪 PAPER TRADE: Would SELL 1.21574849 BTC @ $112,500.00
🧪 Reason: MA Crossover: short below long
🧪 Would receive: $136,396.13 USD (after fees)
```

## Safety Guarantees

1. ✅ **NO REAL MONEY AT RISK** - Paper mode only
2. ✅ **Proximity threshold active** - Prevents bad flips
3. ✅ **Clear warnings** - Can't miss that it's paper mode
4. ✅ **Backward compatible** - Can revert anytime

## Questions to Answer Before Going Live

After running paper tests:
1. Did proximity threshold prevent excessive flipping?
2. Were there any unexpected errors?
3. Did 5-minute testing reveal any bugs?
4. Is the signal logic working correctly?
5. Are you comfortable with the trade frequency?

## Emergency Stop

If anything goes wrong:
```bash
ssh ck 'pkill -f python.*tdr_server'
```

---

**Ready to deploy?** The system is set up for completely safe paper trading.