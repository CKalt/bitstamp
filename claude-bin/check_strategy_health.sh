#!/bin/bash
# Simple script to check if strategy is evaluating signals

echo "Checking strategy health..."
echo "========================="

# Get last 5 heartbeats
echo -e "\nLast 5 heartbeats:"
ssh ck "grep 'HEARTBEAT' ~/projects/bitstamp/logs/tdr_server.log | tail -5"

# Get last signal evaluation
echo -e "\nLast signal evaluation:"
ssh ck "grep 'SIGNAL_EVAL' ~/projects/bitstamp/logs/tdr_server.log | tail -1"

# Check time since last heartbeat
LAST_HB=$(ssh ck "grep 'HEARTBEAT' ~/projects/bitstamp/logs/tdr_server.log | tail -1 | awk '{print \$2}'" 2>/dev/null)
if [ -n "$LAST_HB" ]; then
    CURRENT=$(date +%s)
    LAST=$(date -d "$LAST_HB" +%s 2>/dev/null || echo 0)
    DIFF=$((CURRENT - LAST))
    
    echo -e "\nTime since last heartbeat: $DIFF seconds"
    
    if [ $DIFF -gt 120 ]; then
        echo "⚠️  WARNING: No heartbeat for over 2 minutes!"
    elif [ $DIFF -gt 60 ]; then
        echo "⚠️  CAUTION: No heartbeat for over 1 minute"
    else
        echo "✅ Strategy loop is healthy"
    fi
else
    echo "❌ No heartbeats found in log!"
fi

# Check current position and price
echo -e "\nCurrent status:"
curl -s http://localhost:4000/api/status | jq '{last_price, position: .position.position, entry_price: .position.entry_price}'