#!/bin/bash
# Quick MA Strategy Check - Shows what each strategy would do right now

echo "🔮 Quick MA Strategy Check"
echo "=========================="
echo ""

# Get current position from API
echo "📍 Getting current position..."
CURRENT_STATUS=$(curl -s http://localhost:4000/api/status 2>/dev/null)

if [ -n "$CURRENT_STATUS" ]; then
    CURRENT_PRICE=$(echo "$CURRENT_STATUS" | jq -r '.last_price' 2>/dev/null || echo "0")
    CURRENT_POS=$(echo "$CURRENT_STATUS" | jq -r '.position.position' 2>/dev/null || echo "0")
    POSITION_SIZE=$(echo "$CURRENT_STATUS" | jq -r '.position.position_size' 2>/dev/null || echo "0")
    
    echo "Current Price: \$$CURRENT_PRICE"
    echo -n "Current Position: $CURRENT_POS "
    
    if [ "$CURRENT_POS" = "1" ]; then
        echo "(LONG)"
    elif [ "$CURRENT_POS" = "-1" ]; then
        echo "(SHORT)"
    else
        echo "(NEUTRAL)"
    fi
    
    if [ "$POSITION_SIZE" != "null" ] && [ "$POSITION_SIZE" != "0" ]; then
        echo "Position Size: $POSITION_SIZE BTC"
    fi
else
    echo "⚠️  Could not connect to API"
    CURRENT_POS="1"
    echo "Assuming: LONG position"
fi

echo ""
echo "🔍 Checking MA strategies on server..."
echo ""

# Function to check a specific MA strategy
check_ma_strategy() {
    local SHORT=$1
    local LONG=$2
    local DESC=$3
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "MA $SHORT/$LONG - $DESC"
    
    # Get approximate MA values from server
    # This is a rough calculation - just for preview
    RESULT=$(ssh ck "cd ~/projects/bitstamp && python3 -c \"
import json
import sys
from datetime import datetime, timedelta

# Read recent prices
prices = []
timestamps = []
count = 0

# Read from end of file backwards
import subprocess
result = subprocess.run(['tail', '-n', '5000', 'btcusd.log'], capture_output=True, text=True)

for line in result.stdout.strip().split('\n'):
    try:
        data = json.loads(line)
        if 'data' in data and 'price' in data['data']:
            prices.append(float(data['data']['price']))
            timestamps.append(float(data['data']['timestamp']))
            count += 1
            if count > $LONG * 120:  # Rough: 120 entries per hour
                break
    except:
        continue

if len(prices) < $LONG * 60:
    print('NOT_ENOUGH_DATA')
else:
    prices.reverse()
    
    # Simple MA calculation
    ma_short = sum(prices[-$SHORT*60:]) / ($SHORT*60)
    ma_long = sum(prices[-$LONG*60:]) / ($LONG*60)
    
    signal = 1 if ma_short > ma_long else -1
    diff = ma_short - ma_long
    diff_pct = (diff / ma_long) * 100
    
    print(f'MA{$SHORT}:{ma_short:.0f}|MA{$LONG}:{ma_long:.0f}|SIGNAL:{signal}|DIFF:{diff:.0f}|PCT:{diff_pct:.2f}')
\"" 2>/dev/null)
    
    if [ "$RESULT" = "NOT_ENOUGH_DATA" ]; then
        echo "❌ Not enough data for this MA period"
        return
    fi
    
    if [ -z "$RESULT" ]; then
        echo "❌ Could not calculate MAs"
        return
    fi
    
    # Parse the result
    MA_SHORT_VAL=$(echo "$RESULT" | cut -d'|' -f1 | cut -d':' -f2)
    MA_LONG_VAL=$(echo "$RESULT" | cut -d'|' -f2 | cut -d':' -f2)
    SIGNAL=$(echo "$RESULT" | cut -d'|' -f3 | cut -d':' -f2)
    DIFF=$(echo "$RESULT" | cut -d'|' -f4 | cut -d':' -f2)
    PCT=$(echo "$RESULT" | cut -d'|' -f5 | cut -d':' -f2)
    
    echo "MA$SHORT: \$$MA_SHORT_VAL"
    echo "MA$LONG: \$$MA_LONG_VAL"
    echo "Difference: \$$DIFF ($PCT%)"
    echo -n "Signal: $SIGNAL "
    
    if [ "$SIGNAL" = "1" ]; then
        echo "(LONG)"
    else
        echo "(SHORT)"
    fi
    
    # Check if it would trade
    echo ""
    if [ "$SIGNAL" = "1" ] && [ "$CURRENT_POS" = "-1" ]; then
        echo "⚠️  WOULD TRADE IMMEDIATELY: Flip from SHORT to LONG"
    elif [ "$SIGNAL" = "-1" ] && [ "$CURRENT_POS" = "1" ]; then
        echo "⚠️  WOULD TRADE IMMEDIATELY: Flip from LONG to SHORT"
    elif [ "$SIGNAL" = "$CURRENT_POS" ]; then
        echo "✅ SAFE: Signal matches your position"
    else
        echo "🔄 Signal different but no immediate trade"
    fi
}

# Check different strategies
check_ma_strategy 4 20 "Aggressive"
echo ""
check_ma_strategy 12 48 "Conservative"
echo ""
check_ma_strategy 6 34 "Moderate"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 SUMMARY"
echo ""
echo "Choose a strategy where:"
echo "• ✅ Signal matches your position (SAFE)"
echo "• ⚠️  Signal differs if you WANT to trade"
echo ""
echo "To apply: ssh ck && gg btc && nano best_strategy.json"