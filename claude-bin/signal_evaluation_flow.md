# Signal Evaluation Flow - Detailed

## Every 30 Seconds:

### 1. Data Collection
- Get latest price from WebSocket feed
- Resample data to hourly bars (1H frequency)
- Calculate MA4 and MA20 using latest prices

### 2. Signal Calculation
```
MA4 = Average of last 4 hourly bars
MA20 = Average of last 20 hourly bars
Proximity = |MA4 - MA20| / MA20 * 100
Signal = LONG if MA4 > MA20, SHORT if MA4 < MA20
```

### 3. Trade Decision Logic
```
IF current_position != signal AND proximity <= 0.3% THEN
    CHECK additional conditions:
    - Not in startup grace period
    - Daily trade limit not exceeded (max 5)
    - Hourly trade limit not exceeded (max 2)
    - Min time between trades elapsed (30 min)
    - do_live_trades = true
    
    IF all conditions pass THEN
        EXECUTE TRADE
    ELSE
        LOG why trade was skipped
END IF
```

### 4. Trade Execution
When a trade executes:
1. SELL all BTC (if going SHORT from LONG)
2. BUY with all USD (if going LONG from SHORT)
3. Update position tracking
4. Write to trades.json
5. Update resume-auto-trade.json

## Critical Points to Monitor:

### A. Signal Evaluation Points (every 30 seconds)
Look for log entries:
- `SIGNAL_EVAL: MA4=X MA20=Y Diff=Z Prox=W% Sig=S Pos=P`
- `CHECK_FOR_SIGNALS: signal=X, price=Y, position=Z`

### B. Decision Points
Watch for:
- "Buy signal triggered" or "Sell signal triggered"
- "Reached daily trade limit"
- "Skipping - same signal time"
- "MA Crossover Proximity: X%"

### C. Trade Execution
Confirm:
- "Executing trade: MA Crossover"
- "Executed LIVE BUY/SELL order"
- Position change in status

## Verification Checklist:

Every 30 seconds when proximity < 0.4%:
- [ ] Check server log shows SIGNAL_EVAL entry
- [ ] Verify MA values are reasonable
- [ ] Confirm proximity calculation is correct
- [ ] If proximity <= 0.3%, verify trade decision
- [ ] If trade should execute, confirm it does
- [ ] If trade shouldn't execute, understand why

## Common Bug Scenarios to Watch For:

1. **Signal Not Evaluated**
   - No SIGNAL_EVAL logs every 30 seconds
   - Indicates evaluation loop issue

2. **Wrong Proximity Calculation**
   - MA values don't match proximity %
   - Formula: |MA4-MA20|/MA20 * 100

3. **Trade Should Execute But Doesn't**
   - Proximity <= 0.3%
   - Position != Signal
   - But no trade occurs
   - Check: limits, grace period, configuration

4. **Trade Executes at Wrong Time**
   - Proximity > 0.3%
   - Trade still happens
   - Indicates threshold check failure

5. **Position Tracking Mismatch**
   - Status shows different position than trades.json
   - Resume file not updated
   - Balance calculations wrong