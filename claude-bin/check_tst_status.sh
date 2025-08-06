#!/bin/bash
# Check status of test instance

echo "📊 CHECKING TEST INSTANCE STATUS"
echo "================================"
echo ""

# 1. Check if server is running
echo "1. Server Status:"
if ssh ck 'screen -ls | grep -q server-tst'; then
    echo "   ✓ Test server is running"
    ssh ck 'screen -ls | grep server-tst'
else
    echo "   ✗ Test server is NOT running"
fi

# 2. Check config
echo ""
echo "2. Current Config:"
ssh ck 'cd /home/chris/projects/bitstamp-testing && cat best_strategy.json 2>/dev/null | grep -E "(do_live_trades|candle_interval|proximity_threshold)" | sed "s/^/   /"' || echo "   ✗ No config found"

# 3. Check recent activity
echo ""
echo "3. Recent Activity:"
LOG_FILE="/home/chris/projects/bitstamp-testing/logs/tdr_server.log"
if ssh ck "test -f $LOG_FILE"; then
    SIGNALS=$(ssh ck "grep -c 'SIGNAL_EVAL' $LOG_FILE 2>/dev/null" || echo "0")
    BLOCKS=$(ssh ck "grep -c 'NO_TRADE_PROXIMITY' $LOG_FILE 2>/dev/null" || echo "0")
    PAPERS=$(ssh ck "grep -c 'PAPER TRADE:' $LOG_FILE 2>/dev/null" || echo "0")
    
    echo "   Signal evaluations: $SIGNALS"
    echo "   Proximity blocks: $BLOCKS"
    echo "   Paper trades: $PAPERS"
    
    echo ""
    echo "4. Last 5 Activities:"
    ssh ck "grep -E '(SIGNAL_EVAL|PAPER TRADE:|NO_TRADE_PROXIMITY)' $LOG_FILE | tail -5" | sed 's/^/   /'
else
    echo "   ✗ No log file found"
fi

# 4. Check git status
echo ""
echo "5. Code Version:"
ssh ck 'cd /home/chris/projects/bitstamp-testing && git log --oneline -1' | sed 's/^/   /'

echo ""
echo "6. Test vs Production:"
echo "   Test dir: /home/chris/projects/bitstamp-testing/"
echo "   Prod dir: /home/chris/projects/bitstamp/"
echo "   ⚠️  Make sure you're using the right one!"