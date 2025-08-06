#!/bin/bash
# Verify live trading matches backtest expectations

echo "🔍 LIVE TRADING vs BACKTEST VERIFICATION"
echo "========================================"
echo ""

# Count blocks and trades in current session
echo "📊 CURRENT SESSION METRICS:"
echo "--------------------------"

# Find when current session started
start_time=$(ssh ck 'grep "Starting TDR Server" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -1 | cut -d" " -f1-2')
echo "Session started: $start_time"

# Count blocks since session start
blocks_since_start=$(ssh ck "grep -A1000 \"$start_time\" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | grep -c 'BLOCKING TRADE'" 2>/dev/null || echo "0")
echo "Trades blocked: $blocks_since_start"

# Count actual trades since session start  
trades_since_start=$(ssh ck "grep -A1000 \"$start_time\" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | grep -c 'PAPER TRADE:'" 2>/dev/null || echo "0")
echo "Trades executed: $trades_since_start"

# Calculate block rate
total=$((blocks_since_start + trades_since_start))
if [ $total -gt 0 ]; then
    block_rate=$(echo "scale=1; $blocks_since_start * 100 / $total" | bc)
    echo "Block rate: ${block_rate}%"
else
    echo "Block rate: N/A (no signals yet)"
fi

echo ""
echo "🎯 EXPECTED BACKTEST BEHAVIOR:"
echo "------------------------------"
echo "With 0.3% proximity threshold:"
echo "  - Block rate: 95-99%"
echo "  - Trades: 1-3 per day (hourly)"
echo "  - Trades: 20-50 per day (1-min test)"
echo "  - No trades within 30 min of each other"

echo ""
echo "✅ VERIFICATION POINTS:"
echo "----------------------"

# Check if blocking is working
if [ $blocks_since_start -gt 0 ]; then
    echo "✅ Proximity blocking IS working"
    
    # Show sample of blocked trades
    echo ""
    echo "Sample blocked trades (last 3):"
    ssh ck "grep 'BLOCKING TRADE' /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -3" | while read line; do
        timestamp=$(echo "$line" | cut -d' ' -f1-2)
        proximity=$(echo "$line" | grep -o "([0-9.]*%" | tr -d '(')
        echo "  $timestamp - Blocked at $proximity"
    done
else
    echo "⚠️ No blocks detected - MAs may be diverged"
fi

# Check for rapid flips
echo ""
echo "🔄 CHECKING FOR RAPID FLIPS:"
echo "----------------------------"
ssh ck 'grep "PAPER TRADE:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -5' | while read line; do
    timestamp=$(echo "$line" | cut -d' ' -f1-2)
    action=$(echo "$line" | grep -o "Would [a-z]*" | cut -d' ' -f2)
    price=$(echo "$line" | grep -o "@ \$[0-9,]*" | cut -d'$' -f2)
    echo "  $timestamp: $action @ \$$price"
done

echo ""
echo "📝 MATCHING SCORE:"
echo "-----------------"
if [ $block_rate ]; then
    if (( $(echo "$block_rate > 90" | bc -l) )); then
        echo "✅ Block rate matches backtest expectation"
    else
        echo "⚠️ Block rate lower than expected"
    fi
fi

if [ $trades_since_start -eq 0 ] && [ $blocks_since_start -gt 10 ]; then
    echo "✅ Correctly blocking trades when MAs close"
elif [ $trades_since_start -gt 0 ] && [ $blocks_since_start -gt 0 ]; then
    echo "✅ System is both blocking and trading as expected"
else
    echo "⏳ Need more data to verify match"
fi