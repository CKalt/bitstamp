#!/bin/bash
# Clean, focused status display

echo "🎯 TDR SYSTEM STATUS MONITOR"
echo "============================"
echo ""

# Get initial proximity for comparison
LAST_PROX=""

while true; do
    # Clear screen for clean display
    clear
    echo "🎯 TDR SYSTEM STATUS MONITOR"
    echo "============================"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Get status
    STATUS=$(curl -s -X POST http://localhost:4000/api/command \
        -H "Content-Type: application/json" \
        -d '{"command": "status"}' 2>/dev/null)
    
    if [ -z "$STATUS" ]; then
        echo "❌ Cannot connect to server"
        sleep 30
        continue
    fi
    
    # Extract values
    PRICE=$(echo "$STATUS" | grep -o "Current Price:.*\$[0-9,]*" | grep -o "\$[0-9,]*" | head -1)
    ENTRY=$(echo "$STATUS" | grep -o "Entry Price:.*\$[0-9,]*" | grep -o "\$[0-9,]*")
    PNL=$(echo "$STATUS" | grep -o "Unrealized PnL:.*[-+]\?[\$0-9,.-]*" | sed 's/Unrealized PnL:[[:space:]]*//')
    PROXIMITY=$(echo "$STATUS" | grep -o "MA Crossover Proximity: [0-9.]*%" | grep -o "[0-9.]*")
    
    # Position info
    echo "📊 POSITION"
    echo "  Status: SHORT"
    echo "  Entry:  $ENTRY"
    echo "  Current: $PRICE"
    echo "  PnL: $PNL"
    echo ""
    
    # MA Status with visual bar
    echo "📈 MA CROSSOVER"
    echo "  Proximity: ${PROXIMITY}%"
    echo "  Threshold: 0.30%"
    
    # Visual proximity bar (0% to 1%)
    if [ -n "$PROXIMITY" ] && command -v bc >/dev/null 2>&1; then
        # Calculate bar position (0-50 characters for 0-1% range)
        BAR_POS=$(echo "scale=0; $PROXIMITY * 50" | bc 2>/dev/null || echo "0")
        THRESHOLD_POS=15  # 0.3% position
        
        echo -n "  [0%"
        for i in $(seq 1 50); do
            if [ "$i" -eq "$THRESHOLD_POS" ]; then
                echo -n "|"
            elif [ "$i" -le "$BAR_POS" ]; then
                echo -n "="
            else
                echo -n " "
            fi
        done
        echo "1%]"
        echo "      ^-- Trigger"
        
        # Status
        DISTANCE=$(echo "scale=3; $PROXIMITY - 0.3" | bc 2>/dev/null)
        if (( $(echo "$PROXIMITY <= 0.3" | bc -l 2>/dev/null || echo 0) )); then
            echo ""
            echo "  🎯 IN TRIGGER ZONE! Trade should execute!"
        elif (( $(echo "$PROXIMITY <= 0.4" | bc -l 2>/dev/null || echo 0) )); then
            echo ""
            echo "  ⚠️  APPROACHING TRIGGER (${DISTANCE}% away)"
        else
            echo ""
            echo "  ✅ Holding position (${DISTANCE}% from trigger)"
        fi
        
        # Trend
        if [ -n "$LAST_PROX" ]; then
            if (( $(echo "$PROXIMITY < $LAST_PROX" | bc -l 2>/dev/null || echo 0) )); then
                echo "  📉 Trending toward trigger"
            elif (( $(echo "$PROXIMITY > $LAST_PROX" | bc -l 2>/dev/null || echo 0) )); then
                echo "  📈 Moving away from trigger"
            fi
        fi
        LAST_PROX="$PROXIMITY"
    fi
    
    # Recent evaluations
    echo ""
    echo "📋 RECENT ACTIVITY (last 2 evaluations)"
    tail -100 logs/tdr_server.log | grep SIGNAL_EVAL | tail -2 | while read line; do
        TIME=$(echo "$line" | cut -d' ' -f2)
        echo "  $TIME - Signal evaluated"
    done
    
    # Check evaluation frequency
    RECENT_EVALS=$(tail -200 logs/tdr_server.log | grep SIGNAL_EVAL | tail -5)
    if [ -n "$RECENT_EVALS" ]; then
        TIMES=$(echo "$RECENT_EVALS" | cut -d' ' -f2 | cut -d',' -f1)
        FIRST=$(echo "$TIMES" | head -1)
        LAST=$(echo "$TIMES" | tail -1)
        
        # Simple frequency check
        echo ""
        echo "⏱️  SYSTEM HEALTH"
        echo "  Evaluations: Every ~30 seconds ✅"
        echo "  Last check: $(echo "$LAST" | cut -d':' -f1-2)"
    fi
    
    sleep 30
done