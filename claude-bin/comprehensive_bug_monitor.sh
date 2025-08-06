#!/bin/bash
# Comprehensive bug monitoring system

echo "🔍 COMPREHENSIVE BUG MONITORING SYSTEM"
echo "====================================="
echo ""

# Define what we're monitoring for
echo "📋 MONITORING CHECKLIST:"
echo "------------------------"
echo "1. ⏱️  1-minute candles (60/hour expected)"
echo "2. 🚫 Proximity threshold blocks"
echo "3. 🧪 Paper trades execution"
echo "4. ❌ Errors and exceptions"
echo "5. 💰 Position tracking accuracy"
echo "6. 📊 Signal evaluation consistency"
echo "7. 🔄 Multi-part trade handling"
echo "8. 💾 Memory/performance issues"
echo ""

# Initialize counters
LAST_CANDLE_COUNT=0
LAST_ERROR_COUNT=0
LAST_TRADE_COUNT=0
ITERATION=0

# Main monitoring loop
while true; do
    ITERATION=$((ITERATION + 1))
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔄 Monitoring Cycle #$ITERATION - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 1. CHECK CANDLE FREQUENCY
    echo -e "\n1️⃣ CANDLE FREQUENCY CHECK:"
    CANDLE_COUNT=$(ssh ck 'grep -c "NEW 1-MIN CANDLE" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null' || echo "0")
    NEW_CANDLES=$((CANDLE_COUNT - LAST_CANDLE_COUNT))
    echo "   New candles since last check: $NEW_CANDLES"
    
    # Get last 3 candle times to verify timing
    echo "   Recent candle times:"
    ssh ck 'grep "NEW 1-MIN CANDLE" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -3 | grep -o "[0-9][0-9]:[0-9][0-9]:00"' | sed 's/^/      /'
    
    # Check for timing issues
    if [ "$ITERATION" -gt 1 ] && [ "$NEW_CANDLES" -eq 0 ]; then
        echo "   ⚠️  WARNING: No new candles in last cycle!"
    fi
    LAST_CANDLE_COUNT=$CANDLE_COUNT
    
    # 2. CHECK PROXIMITY THRESHOLD
    echo -e "\n2️⃣ PROXIMITY THRESHOLD CHECK:"
    PROXIMITY_BLOCKS=$(ssh ck 'grep -c "NO_TRADE_PROXIMITY" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null' || echo "0")
    echo "   Total proximity blocks: $PROXIMITY_BLOCKS"
    
    # Show recent proximity values
    echo "   Recent proximity values:"
    ssh ck 'grep "SIGNAL_EVAL v2:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -3 | grep -o "Prox=[0-9.]*%" | sed "s/^/      /"'
    
    # 3. CHECK PAPER TRADES
    echo -e "\n3️⃣ PAPER TRADE EXECUTION:"
    TRADE_COUNT=$(ssh ck 'grep -c "PAPER TRADE:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null' || echo "0")
    NEW_TRADES=$((TRADE_COUNT - LAST_TRADE_COUNT))
    
    if [ "$NEW_TRADES" -gt 0 ]; then
        echo "   🎯 NEW TRADES DETECTED: $NEW_TRADES"
        ssh ck 'grep "PAPER TRADE:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -$NEW_TRADES'
    else
        echo "   No new trades (Total: $TRADE_COUNT)"
    fi
    LAST_TRADE_COUNT=$TRADE_COUNT
    
    # 4. CHECK FOR ERRORS
    echo -e "\n4️⃣ ERROR DETECTION:"
    ERROR_COUNT=$(ssh ck 'grep -ci "error\|exception\|traceback" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null' || echo "0")
    NEW_ERRORS=$((ERROR_COUNT - LAST_ERROR_COUNT))
    
    if [ "$NEW_ERRORS" -gt 0 ]; then
        echo "   🚨 NEW ERRORS FOUND: $NEW_ERRORS"
        ssh ck 'grep -i "error\|exception" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -$NEW_ERRORS | head -5'
    else
        echo "   ✅ No new errors (Total: $ERROR_COUNT)"
    fi
    LAST_ERROR_COUNT=$ERROR_COUNT
    
    # 5. CHECK POSITION CONSISTENCY
    echo -e "\n5️⃣ POSITION TRACKING:"
    POSITION_INFO=$(ssh ck 'curl -s http://localhost:4000/api/status 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    pos = data[\"position\"][\"position\"]
    btc = data[\"position\"].get(\"btc_balance\", 0)
    usd = data[\"position\"].get(\"usd_balance\", 0)
    price = data.get(\"last_price\", 0)
    print(f\"Position: {pos} | BTC: {btc:.8f} | USD: {usd:.2f} | Price: {price:.0f}\")
except: 
    print(\"Error reading position\")
"')
    echo "   $POSITION_INFO"
    
    # 6. CHECK SIGNAL EVALUATION RATE
    echo -e "\n6️⃣ SIGNAL EVALUATION RATE:"
    EVAL_COUNT=$(ssh ck 'grep -c "SIGNAL_EVAL v2:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -100 2>/dev/null' || echo "0")
    echo "   Evaluations in last 100 lines: ~$EVAL_COUNT"
    
    # Expected ~1 per minute
    EXPECTED_RATE="60/hour"
    echo "   Expected rate: $EXPECTED_RATE"
    
    # 7. CHECK FOR MULTI-PART TRADE BUGS
    echo -e "\n7️⃣ MULTI-PART TRADE CHECK:"
    MULTI_PARTS=$(ssh ck 'grep -B1 -A1 "Three-part\|multi.*part" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -10' 2>/dev/null)
    if [ ! -z "$MULTI_PARTS" ]; then
        echo "   Multi-part trades detected"
    else
        echo "   No multi-part trade activity"
    fi
    
    # 8. PERFORMANCE CHECK
    echo -e "\n8️⃣ PERFORMANCE CHECK:"
    LOG_SIZE=$(ssh ck 'ls -lh /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null | awk "{print \$5}"' || echo "N/A")
    echo "   Log file size: $LOG_SIZE"
    
    # Check if process is still running
    if ssh ck 'ps aux | grep -q "[t]dr_server.*testing"'; then
        echo "   Process: ✅ Running"
    else
        echo "   Process: ❌ NOT RUNNING - CRASHED!"
        break
    fi
    
    # SUMMARY
    echo -e "\n📊 SUMMARY:"
    echo "   Monitoring duration: $((ITERATION * 30)) seconds"
    echo "   Next check in 30 seconds..."
    
    # Sleep before next iteration
    sleep 30
done

echo -e "\n❌ MONITORING STOPPED - Server appears to have crashed!"