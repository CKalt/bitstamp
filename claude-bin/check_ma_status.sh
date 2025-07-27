#!/bin/bash
# Check MA crossover status using available API endpoints

echo "📊 CHECKING MA CROSSOVER STATUS"
echo "========================================"
echo "Time: $(date)"
echo "----------------------------------------"

# Get current status
echo -e "\n🔹 SYSTEM STATUS:"
STATUS=$(curl -s http://localhost:4000/api/status)

# Extract key values using grep and sed
POSITION=$(echo "$STATUS" | grep -o '"position":[^,]*' | head -1 | sed 's/.*://;s/[^-0-9]//g')
LAST_PRICE=$(echo "$STATUS" | grep -o '"last_price":[^,]*' | sed 's/.*://;s/[^0-9.]//g')
LIVE_TRADING=$(echo "$STATUS" | grep -o '"live_trading":[^,]*' | sed 's/.*://;s/[^a-z]//g')
TRADES_TODAY=$(echo "$STATUS" | grep -o '"trades_today":[^,]*' | sed 's/.*://;s/[^0-9]//g')

if [ "$POSITION" = "-1" ]; then
    POS_NAME="SHORT"
elif [ "$POSITION" = "1" ]; then
    POS_NAME="LONG"
else
    POS_NAME="NEUTRAL"
fi

echo "  Position: $POS_NAME"
echo "  Current Price: \$$LAST_PRICE"
echo "  Live Trading: $LIVE_TRADING"
echo "  Trades Today: ${TRADES_TODAY:-0}"

# Send status command to get MA info
echo -e "\n🔹 SENDING STATUS COMMAND:"
RESPONSE=$(curl -s -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "status"}')

# Extract MA proximity if available
if echo "$RESPONSE" | grep -q "MA Crossover Proximity"; then
    echo -e "\n🔹 MA CROSSOVER INFO:"
    echo "$RESPONSE" | grep -A5 "MA Crossover Proximity" | head -6
    
    # Extract proximity percentage
    PROXIMITY=$(echo "$RESPONSE" | grep -o "MA Crossover Proximity: [0-9.]*%" | sed 's/[^0-9.]//g')
    
    # Get threshold from config
    THRESHOLD=$(grep ma_separation_threshold best_strategy.json | sed 's/[^0-9.]//g')
    
    if [ -n "$PROXIMITY" ] && [ -n "$THRESHOLD" ]; then
        echo -e "\n🔹 TRIGGER ANALYSIS:"
        echo "  Current Proximity: ${PROXIMITY}%"
        echo "  Threshold: ${THRESHOLD}%"
        
        # Use bc for floating point comparison
        if (( $(echo "$PROXIMITY <= $THRESHOLD" | bc -l) )); then
            echo "  🎯 IN TRIGGER ZONE! Trade should execute!"
        else
            DIFF=$(echo "scale=3; $PROXIMITY - $THRESHOLD" | bc)
            echo "  Distance to trigger: ${DIFF}%"
            
            if (( $(echo "$DIFF < 0.05" | bc -l) )); then
                echo "  ⚠️  VERY CLOSE to trigger!"
            fi
        fi
    fi
fi

echo -e "\n========================================"