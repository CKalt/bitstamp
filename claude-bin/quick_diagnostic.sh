#!/bin/bash
# Quick diagnostic to verify system state before making any claims

echo "🔍 QUICK SYSTEM DIAGNOSTIC"
echo "=========================="
echo "Running at: $(date)"
echo ""

# 1. Get system status
STATUS=$(curl -s http://localhost:4000/api/status)
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Cannot connect to server"
    exit 1
fi

# 2. Extract and display key values
echo "✅ SERVER STATUS:"
echo "  Connected successfully"

# Parse JSON manually for key fields
POSITION=$(echo "$STATUS" | grep -o '"position":[^,}]*' | grep -o '[-0-9]*$')
LIVE_TRADING=$(echo "$STATUS" | grep -o '"live_trading":[^,}]*' | grep -o '[a-z]*$')
AUTO_ACTIVE=$(echo "$STATUS" | grep -o '"active":[^,}]*' | head -1 | grep -o '[a-z]*$')

echo "  Position: $POSITION ($([ "$POSITION" = "-1" ] && echo "SHORT" || [ "$POSITION" = "1" ] && echo "LONG" || echo "NEUTRAL"))"
echo "  Live Trading: $LIVE_TRADING"
echo "  Auto-Trader: $AUTO_ACTIVE"

# 3. Get MA proximity from status command
echo ""
echo "📊 MA CROSSOVER STATUS:"
CMD_RESPONSE=$(curl -s -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "status"}' 2>/dev/null)

if echo "$CMD_RESPONSE" | grep -q "MA Crossover Proximity"; then
    PROXIMITY=$(echo "$CMD_RESPONSE" | grep -o "MA Crossover Proximity: [0-9.]*%" | grep -o "[0-9.]*")
    echo "  Proximity: ${PROXIMITY}%"
    
    # Get threshold from config
    if [ -f "best_strategy.json" ]; then
        THRESHOLD=$(grep ma_separation_threshold best_strategy.json | grep -o "[0-9.]*" | head -1)
        echo "  Threshold: ${THRESHOLD}%"
        
        # Check if we can do float comparison
        if command -v bc >/dev/null 2>&1; then
            if (( $(echo "$PROXIMITY <= $THRESHOLD" | bc -l) )); then
                echo "  Status: 🎯 IN TRIGGER ZONE"
            else
                DIFF=$(echo "scale=3; $PROXIMITY - $THRESHOLD" | bc)
                echo "  Status: Waiting (${DIFF}% above threshold)"
            fi
        fi
    fi
else
    echo "  ⚠️  Could not get MA proximity"
fi

# 4. Check for recent errors
echo ""
echo "🔍 RECENT ERRORS (last 5 minutes):"
if [ -f "logs/tdr_server.log" ]; then
    ERRORS=$(tail -n 1000 logs/tdr_server.log | grep -i error | tail -5)
    if [ -z "$ERRORS" ]; then
        echo "  ✅ No recent errors"
    else
        echo "$ERRORS" | while IFS= read -r line; do
            echo "  ⚠️  $line"
        done
    fi
else
    echo "  ⚠️  Log file not found"
fi

# 5. Summary
echo ""
echo "=========================="
echo "DIAGNOSTIC COMPLETE"

# Exit with status based on findings
if [ "$AUTO_ACTIVE" = "true" ] && [ "$LIVE_TRADING" = "true" ]; then
    echo "✅ System appears to be running normally"
    exit 0
else
    echo "⚠️  Check configuration - auto-trader or live trading may be off"
    exit 1
fi