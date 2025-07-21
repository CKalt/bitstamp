# Enhanced Monitoring Guide

## Overview

The trading system now supports two monitoring methods:
1. **JSON File Commands** (original) - Good for complex commands and compatibility
2. **Direct HTTP API** (new) - Faster, more responsive, better for real-time monitoring

## New Monitoring Commands

### Via Client (JSON Files)

```bash
# In the TDR client:
tdr> signal_monitor              # Current signal status
tdr> signal_monitor history      # Last hour of signals
tdr> signal_monitor missed       # Check for missed opportunities
```

### Via curl (Direct HTTP)

```bash
# Quick status check (minimal data for frequent polling)
curl http://localhost:4000/api/quick_status

# Comprehensive signal status
curl http://localhost:4000/api/signal_status

# Signal history
curl "http://localhost:4000/api/signal_history?hours=2"

# Check for missed signals
curl http://localhost:4000/api/missed_signals

# Execute any command via HTTP
curl -X POST http://localhost:4000/api/execute_command \
  -H "Content-Type: application/json" \
  -d '{"command": "status long", "source": "claude_monitor"}'
```

## Signal Monitor Output

The `signal_monitor` command provides:

- **Current Position**: Side, entry price, P&L
- **Moving Averages**: Current values and difference
- **Signal Status**: 
  - Current signal (BUY/SELL)
  - Whether it matches position
  - Distance to signal flip
  - Estimated bars until signal
- **Confirmation Status**: Bars confirmed vs required
- **Alerts**: Warnings about approaching signals or conflicts

## Claude Integration Options

### Option 1: Keep JSON File System (Current)
**Pros:**
- Already working and integrated
- Good audit trail
- Handles complex commands well
- No changes needed

**Cons:**
- ~1-2 second latency from file monitoring
- More complex for simple queries

### Option 2: Switch to curl/HTTP (New)
**Pros:**
- Near-instant response (<100ms)
- Better for frequent monitoring
- Simpler for status checks
- Can still execute complex commands

**Cons:**
- Need to update Claude's command interface
- Less integrated with existing workflow

### Recommendation: Hybrid Approach

Use **both** systems based on the task:

1. **Use curl for**:
   - Frequent status monitoring (every minute)
   - Quick signal checks
   - Real-time monitoring during critical periods

2. **Use JSON files for**:
   - Complex trading commands
   - System configuration changes
   - Batch operations

## Example Monitoring Script for Claude

```bash
#!/bin/bash
# claude_monitor.sh - Real-time monitoring

while true; do
    # Quick status check
    STATUS=$(curl -s http://localhost:4000/api/quick_status)
    
    # Extract key values
    PRICE=$(echo $STATUS | jq -r '.price')
    PNL=$(echo $STATUS | jq -r '.pnl')
    MA_DIFF=$(echo $STATUS | jq -r '.ma_diff_pct')
    SIGNAL_MATCH=$(echo $STATUS | jq -r '.signal_matches')
    
    # Check for alerts
    if [ "$SIGNAL_MATCH" = "false" ]; then
        echo "⚠️ ALERT: Signal mismatch detected!"
        # Get detailed status
        curl -s http://localhost:4000/api/signal_status | jq
    fi
    
    # Brief status line
    echo "$(date '+%H:%M:%S') | Price: $PRICE | P&L: $PNL | MA Diff: $MA_DIFF%"
    
    sleep 60  # Check every minute
done
```

## Critical Monitoring Points

1. **Signal Approaching**: When MA difference < 0.5%
2. **Signal Conflict**: When current signal doesn't match position
3. **Missed Signals**: When signal persists for 3+ bars without execution
4. **Confirmation Progress**: Track bars confirmed vs required

## Setting Up Alerts

The system now provides alerts for:
- Signal approaching (within 3 bars)
- Signal/position mismatch
- Sustained signals without execution

## Best Practices

1. **Monitor MA difference percentage** - Key indicator of signal strength
2. **Watch confirmation progress** - Know when trade will trigger
3. **Check missed signals hourly** - Identify any system issues
4. **Use quick_status for dashboards** - Minimal overhead for frequent checks