#!/bin/bash
# Enhanced monitoring for when approaching trigger zone
# Alerts when getting close and when trade executes

THRESHOLD=0.3
WARNING_ZONE=0.1  # Alert when within 0.1% of threshold
CHECK_INTERVAL=30  # Check every 30 seconds
LAST_POSITION=""
LAST_PROXIMITY=""

echo "🎯 TRIGGER ZONE MONITORING"
echo "=========================="
echo "Threshold: ${THRESHOLD}%"
echo "Warning zone: Within ${WARNING_ZONE}% of threshold"
echo "Checking every ${CHECK_INTERVAL} seconds"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    # Get current status
    STATUS=$(curl -s http://localhost:4000/api/status 2>/dev/null)
    if [ $? -ne 0 ]; then
        echo "❌ Error connecting to server"
        sleep $CHECK_INTERVAL
        continue
    fi
    
    # Extract values
    POSITION=$(echo "$STATUS" | grep -o '"position":[^,}]*' | grep -o '[-0-9]*$' | head -1)
    PRICE=$(echo "$STATUS" | grep -o '"last_price":[^,}]*' | grep -o '[0-9.]*$')
    
    # Get MA proximity from status command
    CMD_RESPONSE=$(curl -s -X POST http://localhost:4000/api/command \
        -H "Content-Type: application/json" \
        -d '{"command": "status"}' 2>/dev/null)
    
    PROXIMITY=$(echo "$CMD_RESPONSE" | grep -o "MA Crossover Proximity: [0-9.]*%" | grep -o "[0-9.]*")
    PNL=$(echo "$CMD_RESPONSE" | grep -o "Unrealized PnL:.*[0-9.]" | grep -o "[-$0-9.]*$")
    
    # Format position name
    if [ "$POSITION" = "-1" ]; then
        POS_NAME="SHORT"
    elif [ "$POSITION" = "1" ]; then
        POS_NAME="LONG"
    else
        POS_NAME="NEUTRAL"
    fi
    
    # Check if position changed (TRADE EXECUTED!)
    if [ -n "$LAST_POSITION" ] && [ "$POSITION" != "$LAST_POSITION" ]; then
        echo ""
        echo "🚨🚨🚨 TRADE EXECUTED! 🚨🚨🚨"
        echo "Position changed from $LAST_POSITION to $POSITION"
        echo "New position: $POS_NAME at $PRICE"
        echo ""
        # Could add notification here (sound, email, etc)
    fi
    
    # Display current status
    TIMESTAMP=$(date +"%H:%M:%S")
    echo -ne "\r[$TIMESTAMP] $POS_NAME | Price: \$$PRICE | MA Proximity: ${PROXIMITY}% | PnL: $PNL"
    
    # Check if in warning zone
    if [ -n "$PROXIMITY" ] && command -v bc >/dev/null 2>&1; then
        DISTANCE=$(echo "scale=3; $PROXIMITY - $THRESHOLD" | bc)
        
        # Alert if very close to threshold
        if (( $(echo "$DISTANCE >= 0 && $DISTANCE <= $WARNING_ZONE" | bc -l) )); then
            echo -ne " ⚠️  APPROACHING TRIGGER! (${DISTANCE}% away)"
        elif (( $(echo "$PROXIMITY <= $THRESHOLD" | bc -l) )); then
            echo -ne " 🎯 IN TRIGGER ZONE!"
        fi
    fi
    
    # Store for next iteration
    LAST_POSITION="$POSITION"
    LAST_PROXIMITY="$PROXIMITY"
    
    sleep $CHECK_INTERVAL
done