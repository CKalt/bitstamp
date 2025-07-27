# TDR System Key Behaviors (USK)
## Understanding to Prevent False Panic Scenarios

**Last Updated**: 2025-07-25 22:30 EDT
**Current System State**: LONG position profitable (+$1,433 as of update)

### CRITICAL SESSION CONTEXT
- System successfully flipped from SHORT to LONG at 19:00 UTC (15:00 EDT)
- Entry: $116,526, Current: ~$117,519
- MA proximity at 0.86% (safely above 0.30% threshold)
- System correctly holding LONG position
- Trade executed within 14 seconds of MA crossover on hourly candle

### 1. POSITION TRACKING IS INDEPENDENT OF AUTO_RESUME

**Key Understanding**: The system tracks positions in TWO ways:
- **Runtime State**: Active trading session maintains position in memory
- **Persistent State**: `resume-auto-trade.json` for recovery after restart

**What I Got Wrong**: I assumed `auto_resume: false` meant the system didn't know its position. This was incorrect because:
- The server was already running and had position in memory
- `auto_resume` only affects WHETHER to load position on startup, not position tracking during runtime
- A running system always knows its position regardless of `auto_resume` setting

**Correct Behavior**:
- `auto_resume: true` → Loads position from file on startup AND tracks during runtime
- `auto_resume: false` → Starts neutral on startup BUT still tracks during runtime

### 2. MA CROSSOVER PROXIMITY THRESHOLD

**Key Understanding**: The system uses a proximity threshold to prevent whipsaw trades:
- `ma_separation_threshold`: Default 0.3% (configurable)
- System only trades when MA separation crosses this threshold
- Being at 0.31% when threshold is 0.3% means "almost there" not "broken"

**What I Got Wrong**: I didn't check the proximity vs threshold before assuming trades should execute.

**Correct Behavior**:
- Proximity > Threshold → No trade, waiting for stronger signal
- Proximity ≤ Threshold → Trade executes
- Small differences (0.01%) are normal market movements

### 3. SYSTEM STATUS VERIFICATION PROTOCOL

**Before Claiming Something Is Broken**, ALWAYS check:

1. **Via API Status**:
   ```bash
   curl -s http://localhost:4000/api/status | jq '.'
   ```

2. **Via Command API**:
   ```bash
   curl -s -X POST http://localhost:4000/api/command \
     -H "Content-Type: application/json" \
     -d '{"command": "status"}' | jq -r '.output'
   ```

3. **Key Values to Verify**:
   - `position`: -1 (SHORT), 1 (LONG), 0 (NEUTRAL)
   - `live_trading`: true/false
   - `auto_trader.active`: true/false
   - MA Crossover Proximity vs threshold

### 4. TRADE EXECUTION CONDITIONS

**All Must Be True** for a trade to execute:
1. Position differs from signal (SHORT vs LONG)
2. MA proximity ≤ threshold
3. Not in startup grace period
4. Daily/hourly trade limits not exceeded
5. Minimum time between trades elapsed
6. `do_live_trades: true` (for real trades)

**What I Got Wrong**: I focused on individual conditions without checking ALL requirements.

### 5. CONFIGURATION vs RUNTIME STATE

**Key Understanding**: 
- **Configuration** (`best_strategy.json`): What the system SHOULD do
- **Runtime State** (server memory): What the system IS doing
- **Persistent State** (resume files): What to restore on restart

**Critical**: Changing configuration requires restart to take effect!

### 6. PROPER DIAGNOSTIC APPROACH

**NEVER assume broken until verified**:

1. **First**: Check if system knows its state
   ```bash
   ./check_ma_status.sh  # Uses API to verify
   ```

2. **Second**: Compare current values to thresholds
   - MA proximity vs `ma_separation_threshold`
   - Trade count vs `max_trades_per_day`
   - Time since last trade vs `min_time_between_trades_minutes`

3. **Third**: Check logs for actual errors
   ```bash
   tail -n 50 logs/tdr_server.log | grep -i error
   ```

4. **Only Then**: Conclude if something is actually broken

### 7. SIGNAL EVALUATION FREQUENCY

**Key Understanding**: 
- System evaluates signals every 30 seconds
- MA values update with each new price
- Small fluctuations in proximity are normal
- A 0.31% → 0.34% change doesn't mean "moving wrong direction" if price moved favorably

### 8. COMMON FALSE ALARMS TO AVOID

1. **"System doesn't know position"** → Check runtime state, not just config
2. **"Trades aren't executing"** → Check proximity vs threshold first
3. **"Configuration not working"** → Did server restart after change?
4. **"Signal wrong direction"** → MA4 > MA20 means LONG, regardless of price movement
5. **"System is broken"** → Usually it's waiting for a threshold or condition

### 9. VERIFICATION CHECKLIST

Before declaring any issue:
- [ ] Checked current position via API
- [ ] Verified MA proximity vs threshold
- [ ] Confirmed configuration matches intent
- [ ] Checked if server was restarted after config changes
- [ ] Reviewed recent logs for errors
- [ ] Understood ALL conditions for trade execution
- [ ] Calculated actual vs expected values

### 10. KEY SYSTEM TRUTHS

1. **The system is usually right** - It's been trading successfully
2. **Thresholds exist for good reasons** - Prevent whipsaw losses
3. **Small margins matter** - 0.31% vs 0.30% is a real difference
4. **State is complex** - Multiple conditions must align
5. **Patience is required** - Markets move at their own pace

---

## Summary

The system is sophisticated with multiple safety checks. What appears to be "not working" is usually the system protecting against bad trades. Always verify actual state against configured thresholds before assuming malfunction.

**Golden Rule**: Gather data first, conclude second. Never panic without complete information.

---

## Diagnostic Tools Available

Located in `/claude-bin/`:
1. **quick_diagnostic.sh** - Primary system health check
2. **check_ma_status.sh** - MA crossover proximity analysis
3. **monitor_crossover_live.py** - Real-time MA convergence monitoring
4. **check_status_api.py** - Detailed API status checks
5. **test_signal_evaluation.py** - Signal evaluation testing

**Always run `./claude-bin/quick_diagnostic.sh` FIRST before making any claims about system state.**