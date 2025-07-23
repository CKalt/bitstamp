# TDR Trading System Critical Details

## 🚨 CRITICAL REMINDERS - DON'T FORGET THESE!

### 1. NO EXITS - ONLY FLIPS!
- The system NEVER exits to cash/neutral
- ALWAYS either 100% LONG (holding BTC) or 100% SHORT (holding USD)
- Every "exit" is actually a FLIP to the opposite position
- Stop saying "exit" when you mean "flip"!

### 2. HTTP API is PRIMARY Interface
- **NO NEED for `enable_commands` anymore!**
- The server has full HTTP API at port 4000
- Use curl or direct HTTP requests:
  ```bash
  curl -X POST http://localhost:4000/api/command \
    -H "Content-Type: application/json" \
    -d '{"command": "status"}'
  ```

### 3. Position Tracking States
- **LONG**: position = 1, holding BTC, balance_btc > 0.0001
- **SHORT**: position = -1, holding USD, balance_btc < 0.0001
- System can get confused if both BTC and USD balances exist

## 📡 HTTP API Endpoints

### Core Endpoints
```bash
# Check server health
curl http://localhost:4000/api/ping

# Get/Update strategy configuration
curl http://localhost:4000/api/best_strategy
curl -X POST http://localhost:4000/api/best_strategy -d @strategy.json

# Execute any command
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "status long"}'

# Get logs
curl "http://localhost:4000/api/logs?lines=50"

# Get trades
curl "http://localhost:4000/api/trades?limit=10"
```

## 🔧 Fixing Broken States

### Position Mismatch (System thinks wrong position)

**Symptoms**: 
- Status shows SHORT but you have BTC
- Status shows LONG but you only have USD
- MA signal matches position so no trades happen

**Fix Process**:
1. Stop auto-trader first:
   ```bash
   curl -X POST http://localhost:4000/api/command -d '{"command": "stop_auto_trade"}'
   ```

2. Fix position tracking:
   ```bash
   # For LONG position
   curl -X POST http://localhost:4000/api/command \
     -d '{"command": "fix_position 1.5btc long 119000"}'
   
   # For SHORT position  
   curl -X POST http://localhost:4000/api/command \
     -d '{"command": "fix_position 170000usd short 118000"}'
   ```

3. Resume auto-trading:
   ```bash
   curl -X POST http://localhost:4000/api/command \
     -d '{"command": "resume_auto_trade"}'
   ```

### Manual Trade Recovery

When you execute manual trades outside the system:

1. Create and run fix script (example for manual SELL):
   ```python
   # fix_manual_sell.py
   manual_sell = {
       "timestamp": "2025-07-23T13:00:00Z",
       "type": "sell",
       "amount": btc_amount,
       "price": sell_price,
       "trade_id": "manual_sell_001",
       "source": "manual_fix"
   }
   # Add to trades.json
   # Create resume-auto-trade.json
   ```

2. Update resume-auto-trade.json BEFORE starting server
3. Then start server - it will read the resume file

## 🚀 Proper Startup Sequence

### CRITICAL: Order Matters!

1. **Update resume-auto-trade.json FIRST** (if needed)
   ```bash
   scp resume-auto-trade.json chriskoin:/home/chris/projects/bitstamp/
   ```

2. **Start server**:
   ```bash
   cd /home/chris/projects/bitstamp
   git pull origin stable-added-adaptive-trad-n-chart-more
   python src/tdr.py --server
   ```

3. **Wait for**: "Historical data: 100% complete"

4. **Resume trading** (if not auto-resumed):
   ```bash
   curl -X POST http://localhost:4000/api/command \
     -d '{"command": "resume_auto_trade"}'
   ```

## 📊 Status Checking Commands

### Essential Status Checks
```bash
# Basic status
curl -X POST http://localhost:4000/api/command -d '{"command": "status"}'

# Detailed status with pivot info
curl -X POST http://localhost:4000/api/command -d '{"command": "status long"}'

# Strategy diagnostics
curl -X POST http://localhost:4000/api/command -d '{"command": "strategy_diagnostics"}'

# Recent trades
curl -X POST http://localhost:4000/api/command -d '{"command": "trades"}'

# Check logs for errors
curl -X POST http://localhost:4000/api/command -d '{"command": "logs 50"}'
```

### What to Look For
1. **Position matches reality** (LONG with BTC, SHORT with USD)
2. **Entry price is correct**
3. **MA signal makes sense** (if SHORT, MA4 < MA20)
4. **No position mismatches** in diagnostics

## 🐛 Common Bugs & Fixes

### 1. Position Tracking Bug
**Problem**: System loses track of actual position
**Cause**: validate_position_tracking() incorrectly determines position
**Fix**: Applied in commit b87803a - checks BTC balance first

### 2. Phantom Trades
**Problem**: Trades appear in system but not on exchange
**Fix**: Always verify with actual exchange balance
**Prevention**: Check trades.json matches exchange history

### 3. MA Signal Confusion
**Problem**: System shows wrong MA signal
**Check**: 
```python
# MA4 > MA20 = LONG signal
# MA4 < MA20 = SHORT signal
```

## 💡 Strategy Configuration

### Current Pure MA Strategy
```json
{
  "Strategy": "MA",
  "Short_Window": 4,
  "Long_Window": 20,
  "enable_pivot_protection": false,
  "enable_regime_detection": false,
  "strategy_type": "MA"
}
```

### What's Disabled
- ❌ Adaptive strategy switching
- ❌ Pivot protection
- ❌ Regime detection
- ❌ RSI signals
- ❌ Bollinger bands

## 🔴 Emergency Procedures

### If System is Making Wrong Trades
1. **IMMEDIATELY STOP**:
   ```bash
   curl -X POST http://localhost:4000/api/command -d '{"command": "stop_auto_trade"}'
   ```

2. **Check actual exchange** - what's your real position?

3. **Compare with system**:
   ```bash
   curl -X POST http://localhost:4000/api/command -d '{"command": "status long"}'
   ```

4. **Fix mismatches** before restarting

### If Losing Money Fast
1. **Manual intervention** on exchange if needed
2. **Stop auto-trader**
3. **Document the manual trade**
4. **Run fix script** before restarting

## 📝 Key Architecture Points

### Data Flow
```
Live Trades → WebSocket → DataManager → MA Calculation → Signal → Trade Decision
                                            ↓
                                     (Only at candle close!)
```

### MA Decision Timing
- Decisions made ONLY at hourly candle close (XX:00)
- NOT on every tick
- This creates lag but is by design

### Position State Files
- `trades.json` - Actual trade history (source of truth)
- `resume-auto-trade.json` - Position state for restart
- `best_strategy.json` - Strategy configuration

## ⚠️ Things That Break Claude's Understanding

1. **Thinking there are "exits"** - There aren't! Only flips!
2. **Forgetting HTTP API exists** - Stop using enable_commands!
3. **Not checking actual exchange** - Always verify reality
4. **Assuming instant trades** - MA only checks hourly
5. **Ignoring position mismatches** - These cascade into disasters

## 🎯 Daily Checklist

1. Check position matches exchange
2. Verify entry prices are sensible  
3. Look for position mismatch warnings in logs
4. Ensure MA signals align with position
5. Monitor unrealized P&L

## 🚀 Quick Recovery Templates

### After Manual Buy
```bash
python fix_manual_buy.py  # Create this with trade details
# Then start server
```

### After Manual Sell
```bash
python fix_manual_sell.py  # Updates trades.json and resume
# Then start server
```

### After Server Crash
```bash
# Server auto-resumes from resume-auto-trade.json
# Just restart - no manual intervention needed
```

## 📌 Remember

1. **ALWAYS** verify with exchange before trusting system state
2. **NEVER** assume the system position without checking
3. **DOCUMENT** all manual interventions
4. **UPDATE** trades.json for manual trades
5. **CHECK** logs after any unusual behavior

---

**Last Updated**: 2025-07-23
**Critical**: Keep this document updated with new learnings!