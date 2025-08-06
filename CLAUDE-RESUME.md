# CLAUDE-RESUME.md - Paper Trading Test Session

## Current Situation
- **Testing:** Proximity threshold fix (0.5% minimum MA spread) to prevent excessive flipping
- **Mode:** Paper trading on `gg tst` server (no real money at risk)
- **Speed:** 1-minute candles for rapid testing (60x faster than hourly)
- **Branch:** stable-added-adaptive-trad-n-chart-more
- **Start Time:** ~01:50 on 2025-08-06

## Problem We're Solving
- User lost money due to excessive position flipping when MAs were too close
- Previous trade: Sold at $113,793 (flipped too early)
- Solution: Added proximity threshold - only trade when MA4 and MA20 differ by >0.5%

## Key Files Modified
1. `/src/tdr_core/strategies.py`
   - Added proximity threshold check (0.5%)
   - Added 1-minute candle support for testing
   - Fixed datetime.now() for real-time candles

2. `/src/tdr_core/signal_monitor_enhanced.py`
   - Fixed JSON serialization bug (numpy int64 -> Python int)

3. Configuration on test server:
   ```json
   {
     "Short_Window": 4,
     "Long_Window": 20,
     "do_live_trades": false,
     "candle_interval": "1min",
     "proximity_threshold": 0.5
   }
   ```

## Test Scripts Created
All in `/claude-bin/`:
- `monitor_test.sh` - Main monitoring dashboard
- `verify_1min_fixed.sh` - Verify 1-minute candles working
- `test_auto_trading_complete.sh` - Test all auto trading aspects
- `test_auto_resume_thoroughly.sh` - Test auto-resume functionality
- `detect_specific_bugs.py` - Automated bug detection
- `check_pnl_with_fees.sh` - P&L analysis with fees
- `comprehensive_bug_monitor.sh` - Detailed bug monitoring

## Known Issues Fixed
1. ✅ 1-minute candles triggering correctly (was stuck at hour boundaries)
2. ✅ JSON serialization error in signal_monitor_enhanced
3. ✅ Auto-resume working with correct position/entry price
4. ✅ Server stability (no more crashes)

## Next Steps After Reboot

### 1. Check Test Results
```bash
# Check how many trades were executed vs blocked
ssh ck 'grep -c "PAPER TRADE:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log'
ssh ck 'grep -c "NO_TRADE_PROXIMITY" /home/chris/projects/bitstamp-testing/logs/tdr_server.log'

# See trade history
ssh ck 'grep "PAPER TRADE:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log'

# Check final position and P&L
ssh ck 'tail -100 /home/chris/projects/bitstamp-testing/logs/tdr_server.log | grep -E "(Position|P&L)"'
```

### 2. Analyze Proximity Threshold Effectiveness
```bash
# Get proximity values when trades were blocked
ssh ck 'grep "SIGNAL_EVAL v2:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | grep "NO_TRADE_PROXIMITY"'

# Compare to when trades executed
ssh ck 'grep -B5 "PAPER TRADE:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | grep "SIGNAL_EVAL"'
```

### 3. Decision Points
- If proximity threshold worked well (fewer trades, less flipping):
  - Consider testing with real money but small amounts
  - Maybe adjust threshold (0.5% might be too strict or too loose)
  
- If still too many trades:
  - Increase proximity threshold to 0.7% or 1.0%
  - Add additional filters (momentum, volume, etc.)

### 4. Resume Testing After Reboot
```bash
# SSH to test server
ssh ck
cd /home/chris/projects/bitstamp-testing

# Start server in screen
screen -S server-tst -dm ./env/bin/python src/tdr_server.py

# Monitor from local machine
./claude-bin/monitor_test.sh
```

## Important Context
- User has $156,574 USD available for trading
- System uses 3-part trades to stay under 90% rule
- Paper trading shows what would happen without risking money
- Test server (`gg tst`) is separate from production (`gg btc`)

## Current Position (as of last check)
- Position: SHORT -1.37 BTC
- Entry: $113,633
- System auto-resumes position on restart

## Commands to Remember
- Monitor: `./claude-bin/monitor_test.sh`
- Restart server: `ssh ck 'cd /home/chris/projects/bitstamp-testing && screen -S server-tst -dm ./env/bin/python src/tdr_server.py'`
- Check logs: `ssh ck 'tail -f /home/chris/projects/bitstamp-testing/logs/tdr_server.log'`