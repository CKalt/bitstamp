#!/bin/bash
# Screen 2: Quick Status Check Loop
# Checks MA proximity every 30 seconds

echo "📈 SCREEN 2: MA PROXIMITY MONITOR"
echo "================================="
echo "Checking MA crossover proximity every 30 seconds"
echo "Threshold: 0.3%"
echo ""

while true; do
    # Get status via API
    STATUS=$(curl -s -X POST http://localhost:4000/api/command \
        -H "Content-Type: application/json" \
        -d '{"command": "status"}' 2>/dev/null)
    
    # Extract key values
    PROXIMITY=$(echo "$STATUS" | grep -o "MA Crossover Proximity: [0-9.]*%" | grep -o "[0-9.]*")
    PRICE=$(echo "$STATUS" | grep -o "Current Price:.*\$[0-9,]*" | grep -o "\$[0-9,]*")
    PNL=$(echo "$STATUS" | grep -o "Unrealized PnL:.*[-$0-9,.]" | grep -o "[-$0-9,.]*$")
    
    # Display with timestamp
    echo "[$(date +%H:%M:%S)] Price: $PRICE | Proximity: ${PROXIMITY}% | PnL: $PNL"
    
    # Alert if approaching trigger
    if [ -n "$PROXIMITY" ] && command -v bc >/dev/null 2>&1; then
        if (( $(echo "$PROXIMITY <= 0.4" | bc -l) )); then
            echo "⚠️  APPROACHING TRIGGER ZONE!"
        fi
        if (( $(echo "$PROXIMITY <= 0.3" | bc -l) )); then
            echo "🎯 IN TRIGGER ZONE! TRADE SHOULD EXECUTE!"
        fi
    fi
    
    sleep 30
done