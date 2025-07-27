#!/bin/bash
# Quick check of MA proximity with visual indicator

STATUS=$(curl -s -X POST http://localhost:4000/api/command \
    -H "Content-Type: application/json" \
    -d '{"command": "status"}' 2>/dev/null)

PROXIMITY=$(echo "$STATUS" | grep -o "MA Crossover Proximity: [0-9.]*%" | grep -o "[0-9.]*")
PRICE=$(echo "$STATUS" | grep -o "Current Price:.*[0-9]" | grep -o "\\$[0-9.]*" | tail -1)
PNL=$(echo "$STATUS" | grep -o "Unrealized PnL:.*[0-9.]" | grep -o "[-$0-9.]*$")

echo "📊 MA PROXIMITY CHECK"
echo "===================="
echo "Current Price: $PRICE"
echo "Proximity: ${PROXIMITY}%"
echo "Threshold: 0.3%"
echo "PnL: $PNL"
echo ""

# Visual proximity bar
if command -v bc >/dev/null 2>&1 && [ -n "$PROXIMITY" ]; then
    # Create visual representation (0% to 2%)
    SCALED=$(echo "scale=0; $PROXIMITY * 50" | bc)
    THRESHOLD_POS=$(echo "scale=0; 0.3 * 50" | bc)
    
    echo -n "0% ["
    for i in $(seq 1 100); do
        if [ $i -eq $THRESHOLD_POS ]; then
            echo -n "|"
        elif [ $i -le $SCALED ]; then
            echo -n "="
        else
            echo -n " "
        fi
    done
    echo "] 2%"
    echo "    ^"
    echo "    Trigger at 0.3%"
    
    DISTANCE=$(echo "scale=3; $PROXIMITY - 0.3" | bc)
    if (( $(echo "$DISTANCE > 0" | bc -l) )); then
        echo ""
        echo "📏 Distance to trigger: ${DISTANCE}%"
    else
        echo ""
        echo "🎯 IN TRIGGER ZONE!"
    fi
fi