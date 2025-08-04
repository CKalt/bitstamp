#!/bin/bash
# Check what signals are currently active to help choose strategy

echo "🔍 Current Signal Check"
echo "======================"
echo ""

# Get current position
echo "📍 Your Current Position:"
POS_INFO=$(curl -s http://localhost:4000/api/status 2>/dev/null | jq -r '.position.position' 2>/dev/null)

if [ "$POS_INFO" = "1" ]; then
    echo "   LONG (holding BTC)"
elif [ "$POS_INFO" = "-1" ]; then
    echo "   SHORT (holding USD)"
else
    echo "   Unknown (check manually)"
    POS_INFO="1"  # Assume LONG for safety
fi

echo ""
echo "📊 Checking Recent MA Signals on Server..."
echo ""

# Get last few signal evaluations from different configs
echo "Recent Signal Evaluations:"
echo "-------------------------"

# Check server logs for recent evaluations
ssh ck 'grep "SIGNAL_EVAL" ~/projects/bitstamp/logs/tdr_server.log | tail -10' 2>/dev/null | while read -r line; do
    # Extract key info
    if [[ $line =~ MA([0-9]+)=([0-9]+).*MA([0-9]+)=([0-9]+).*Sig=(-?[0-9]+) ]]; then
        MA_SHORT="${BASH_REMATCH[1]}"
        MA_SHORT_VAL="${BASH_REMATCH[2]}"
        MA_LONG="${BASH_REMATCH[3]}"
        MA_LONG_VAL="${BASH_REMATCH[4]}"
        SIGNAL="${BASH_REMATCH[5]}"
        
        # Extract timestamp
        TIMESTAMP=$(echo "$line" | awk '{print $1, $2}')
        
        echo "[$TIMESTAMP] MA$MA_SHORT/$MA_LONG: Signal=$SIGNAL"
    fi
done

echo ""
echo "🎯 What Each Strategy Would Do:"
echo "------------------------------"

# Function to explain action
explain_action() {
    local SIGNAL=$1
    local POSITION=$2
    
    if [ "$SIGNAL" = "1" ] && [ "$POSITION" = "-1" ]; then
        echo "⚠️  WOULD TRADE: Flip SHORT → LONG"
    elif [ "$SIGNAL" = "-1" ] && [ "$POSITION" = "1" ]; then
        echo "⚠️  WOULD TRADE: Flip LONG → SHORT"
    elif [ "$SIGNAL" = "$POSITION" ]; then
        echo "✅ NO TRADE: Signal matches position"
    else
        echo "🔄 Position mismatch but depends on config"
    fi
}

# Try to get specific MA configs from recent data
echo ""
echo "Based on your position ($POS_INFO):"
echo ""

# Check MA 4/20
LAST_MA420=$(ssh ck 'grep "MA4=" ~/projects/bitstamp/logs/tdr_server.log | tail -1' 2>/dev/null)
if [[ $LAST_MA420 =~ Sig=(-?[0-9]+) ]]; then
    echo -n "MA 4/20 (Aggressive):  "
    explain_action "${BASH_REMATCH[1]}" "$POS_INFO"
fi

# Check MA 12/48
LAST_MA1248=$(ssh ck 'grep "MA12=" ~/projects/bitstamp/logs/tdr_server.log | tail -1' 2>/dev/null)
if [[ $LAST_MA1248 =~ Sig=(-?[0-9]+) ]]; then
    echo -n "MA 12/48 (Stable):     "
    explain_action "${BASH_REMATCH[1]}" "$POS_INFO"
fi

# Manual calculation helper
echo ""
echo "💡 Quick Decision Guide:"
echo "----------------------"
echo "1. If you see ✅ NO TRADE - that strategy is SAFE to use"
echo "2. If you see ⚠️  WOULD TRADE - only use if you WANT to flip positions"
echo "3. Can't decide? Use MA 12/48 for stability"
echo ""
echo "📝 To Set Your Choice:"
echo "ssh ck"
echo "gg btc"
echo "nano best_strategy.json"
echo "# Edit Short_Window and Long_Window"
echo "# Save and restart server"