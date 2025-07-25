#!/bin/bash
# Screen 3: Trade Activity Monitor
# Watches trades.json for new entries

echo "💰 SCREEN 3: TRADE ACTIVITY MONITOR"
echo "==================================="
echo "Monitoring trades.json for new trades"
echo ""

# Get initial trade count
if [ -f trades.json ]; then
    INITIAL_COUNT=$(grep -c '"type"' trades.json 2>/dev/null || echo 0)
else
    INITIAL_COUNT=0
fi

echo "Starting with $INITIAL_COUNT trades in history"
echo "Watching for new trades..."
echo ""

# Monitor for changes
while true; do
    if [ -f trades.json ]; then
        CURRENT_COUNT=$(grep -c '"type"' trades.json 2>/dev/null || echo 0)
        
        if [ $CURRENT_COUNT -gt $INITIAL_COUNT ]; then
            echo ""
            echo "🚨🚨🚨 NEW TRADE DETECTED! 🚨🚨🚨"
            echo "Time: $(date)"
            
            # Show the last trade
            LAST_TRADE=$(tail -20 trades.json | grep -B5 -A5 '"type"' | tail -10)
            echo "Trade details:"
            echo "$LAST_TRADE"
            echo ""
            
            INITIAL_COUNT=$CURRENT_COUNT
        fi
    fi
    
    # Also check server logs for trade execution
    RECENT_TRADE=$(tail -5 logs/tdr_server.log | grep -E "Executing trade|Executed LIVE" | tail -1)
    if [ -n "$RECENT_TRADE" ]; then
        echo "[$(date +%H:%M:%S)] Recent trade activity: $RECENT_TRADE"
    fi
    
    sleep 10
done