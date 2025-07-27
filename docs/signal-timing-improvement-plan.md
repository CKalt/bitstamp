# Signal Timing Improvement Plan

## Current Problem
- System only evaluates signals on hourly candle closes
- Can be up to 59 minutes late detecting crossovers
- Today's trade at 19:00:14 was pure luck - crossover happened exactly at hour boundary

## Safety Requirements
1. **MUST NOT break existing system** - Run parallel, not replace
2. **MUST NOT cause false signals** - Need smoothing/confirmation
3. **MUST test extensively** before live trading
4. **MUST be able to rollback** instantly if issues

## Proposed Solution: Hybrid Approach

### Phase 1: Parallel Monitoring (NO TRADING)
1. Keep existing hourly MA system running unchanged
2. Add parallel calculation using 5-minute candles
3. Log both signals side-by-side for comparison
4. Measure:
   - How much earlier we detect crossovers
   - How many false signals we'd get
   - Optimal confirmation period

### Phase 2: Testing Framework
```python
# Create test scenarios:
1. Historical comparison - backtest both approaches
2. Paper trading - track what would happen
3. Signal quality metrics:
   - Time to detection
   - False positive rate
   - Profitability difference
```

### Phase 3: Gradual Implementation
1. Start with alerts only (no trades)
2. Add "early warning" system
3. Implement with small position sizes
4. Full deployment only after proven results

## Technical Approach

### Option A: Higher Frequency MA Calculation
```python
# Calculate MA6/MA34 on 5-minute candles
# MA6 on 5min = 30 minutes of data
# MA34 on 5min = 170 minutes of data
# More responsive but more noise
```

### Option B: Real-time Price MA
```python
# Calculate MAs on streaming tick data
# Smooth with time-weighted average
# Most responsive but needs careful filtering
```

### Option C: Dual Timeframe Confirmation
```python
# Use 15-min for signal detection
# Confirm with hourly trend
# Balance of speed and reliability
```

## Testing Checklist
- [ ] Implement parallel calculation without touching trade logic
- [ ] Log all signal differences with timestamps
- [ ] Run for 24-48 hours to collect data
- [ ] Analyze false signals and whipsaws
- [ ] Create unit tests for edge cases
- [ ] Test connection failures and recovery
- [ ] Verify no memory leaks or performance issues
- [ ] Test rollback procedure

## Risk Mitigation
1. **Circuit breaker**: Disable if >3 trades per hour
2. **Sanity checks**: Signal must persist for X minutes
3. **Position limits**: Start with 10% position size
4. **Manual override**: Kill switch always available
5. **Alerting**: Notify on unusual behavior

## Success Metrics
- Detect crossovers within 5 minutes (vs current 30min average)
- False signal rate <10%
- Improved entry prices by >0.1%
- No increase in losing trades

## Implementation Order
1. Create diagnostic tool to analyze current delays
2. Build parallel monitoring system
3. Collect 1 week of comparison data
4. Design confirmation algorithm
5. Implement with extensive tests
6. Paper trade for 1 week
7. Live trade with small size
8. Full deployment

## Code Structure
```
src/tdr_core/
  strategies.py          # Keep unchanged
  strategies_rt.py       # New real-time version
  signal_comparison.py   # Compare both approaches
  
tests/
  test_signal_timing.py  # Comprehensive tests
  test_ma_calculation.py # Verify MA accuracy
  test_whipsaw.py       # Test false signal filtering
```

## Rollback Plan
1. Feature flag: `enable_realtime_signals: false`
2. All new code in separate modules
3. Original system untouched
4. One-line change to disable

This plan ensures we can improve signal timing without risking the working system.