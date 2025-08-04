#!/bin/bash
# Safe Server Start - Prevents accidental trades

echo "🛡️  SAFE SERVER START CHECKLIST"
echo "=============================="
echo ""

# Step 1: Show current position
echo "📍 Step 1: Know Your Position"
CURRENT_POS=$(curl -s http://localhost:4000/api/status 2>/dev/null | jq -r '.position.position' 2>/dev/null || echo "unknown")

if [ "$CURRENT_POS" = "1" ]; then
    echo "   You are currently: LONG (holding BTC)"
    POS_NAME="LONG"
elif [ "$CURRENT_POS" = "-1" ]; then
    echo "   You are currently: SHORT (holding USD)" 
    POS_NAME="SHORT"
else
    echo "   ⚠️  Cannot determine position!"
    echo "   Check manually with: curl http://localhost:4000/api/status | jq '.position'"
    exit 1
fi

echo ""
echo "📋 Step 2: Check Server Config"
echo "   Checking server configuration..."

SERVER_CONFIG=$(ssh ck "cat ~/projects/bitstamp/best_strategy.json" 2>/dev/null)
if [ -z "$SERVER_CONFIG" ]; then
    echo "   ❌ Could not read server config!"
    echo "   Fix: ssh ck && cat ~/projects/bitstamp/best_strategy.json"
    exit 1
fi

SHORT_WIN=$(echo "$SERVER_CONFIG" | jq -r '.Short_Window' 2>/dev/null)
LONG_WIN=$(echo "$SERVER_CONFIG" | jq -r '.Long_Window' 2>/dev/null)
LIVE_TRADES=$(echo "$SERVER_CONFIG" | jq -r '.do_live_trades' 2>/dev/null)

echo "   Current server config:"
echo "   • MA Strategy: $SHORT_WIN/$LONG_WIN"
echo "   • Live Trading: $LIVE_TRADES"

if [ "$LIVE_TRADES" != "true" ]; then
    echo "   ⚠️  WARNING: Live trading is DISABLED!"
fi

echo ""
echo "🔮 Step 3: Preview What Will Happen"
echo "   Checking what MA $SHORT_WIN/$LONG_WIN would do..."

# Try to get recent signal for this MA
RECENT_SIGNAL=$(ssh ck "grep \"MA${SHORT_WIN}=\" ~/projects/bitstamp/logs/tdr_server.log | tail -1" 2>/dev/null)

if [[ $RECENT_SIGNAL =~ Sig=(-?[0-9]+) ]]; then
    SIGNAL="${BASH_REMATCH[1]}"
    
    echo -n "   Latest Signal: $SIGNAL "
    if [ "$SIGNAL" = "1" ]; then
        echo "(LONG)"
    else
        echo "(SHORT)"
    fi
    
    echo ""
    echo "   🎯 PREDICTION:"
    
    if [ "$SIGNAL" = "$CURRENT_POS" ]; then
        echo "   ✅ SAFE TO START - Signal matches your position"
        echo "   No immediate trade will occur"
    else
        echo "   ⚠️  WARNING - WILL TRADE IMMEDIATELY!"
        if [ "$SIGNAL" = "1" ] && [ "$CURRENT_POS" = "-1" ]; then
            echo "   Will flip from SHORT to LONG on resume"
        elif [ "$SIGNAL" = "-1" ] && [ "$CURRENT_POS" = "1" ]; then
            echo "   Will flip from LONG to SHORT on resume"
        fi
        
        echo ""
        echo "   Your options:"
        echo "   1. Change config to different MA values"
        echo "   2. Accept the trade"
        echo "   3. Wait for market conditions to change"
    fi
else
    echo "   ⚠️  Could not determine signal"
    echo "   Will need to watch carefully on startup"
fi

echo ""
echo "📝 Step 4: Start Server Safely"
echo ""
echo "If SAFE to proceed:"
echo "1. ssh ck"
echo "2. gg btc"
echo "3. screen -r server"
echo "4. Ctrl-C to stop (if running)"
echo "5. python src/tdr.py --server"
echo "6. WATCH the first few SIGNAL_EVAL lines!"
echo "7. Only resume if signal matches expectation"
echo ""
echo "To resume trading:"
if [ "$CURRENT_POS" = "1" ]; then
    echo "curl -X POST http://localhost:4000/api/command \\"
    echo "  -H \"Content-Type: application/json\" \\"
    echo "  -d '{\"command\": \"auto_trade 1.25btc long\"}'"
else
    echo "curl -X POST http://localhost:4000/api/command \\"
    echo "  -H \"Content-Type: application/json\" \\"
    echo "  -d '{\"command\": \"auto_trade 161000usd short\"}'"
fi

echo ""
echo "🚨 EMERGENCY STOP:"
echo "curl -X POST http://localhost:4000/api/command \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"command\": \"stop_auto_trade\"}'"