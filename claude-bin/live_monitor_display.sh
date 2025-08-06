#!/bin/bash
# Live monitoring display - shows exactly what I'm checking

echo "🔍 LIVE MONITORING DISPLAY"
echo "=========================="
echo "Showing what I check every cycle..."
echo ""

while true; do
    clear
    echo "🔍 LIVE MONITORING - $(date '+%H:%M:%S')"
    echo "================================"
    
    # 1. Check if server is still running
    echo -e "\n1️⃣ SERVER HEALTH:"
    echo -n "   Process running: "
    if ssh ck 'ps aux | grep -q "[t]dr_server.*testing"'; then
        echo "✅ YES"
    else
        echo "❌ NO - SERVER CRASHED!"
    fi
    
    # 2. Check for errors
    echo -e "\n2️⃣ ERROR CHECK:"
    error_count=$(ssh ck 'grep -ci "error\|exception" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null' || echo "0")
    echo "   Total errors found: $error_count"
    if [ "$error_count" -gt "0" ]; then
        echo "   Last error:"
        ssh ck 'grep -i "error\|exception" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -1'
    fi
    
    # 3. Check current price and position
    echo -e "\n3️⃣ CURRENT STATUS:"
    status=$(ssh ck 'curl -s http://localhost:4000/api/status 2>/dev/null')
    if [ ! -z "$status" ]; then
        echo "$status" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(f'   Price: \${data.get(\"last_price\", 0):,.0f}')
    print(f'   Position: {data[\"position\"][\"position\"]} ({\"LONG\" if data[\"position\"][\"position\"] > 0 else \"SHORT\" if data[\"position\"][\"position\"] < 0 else \"NEUTRAL\"})')
    print(f'   Live Trading: {data.get(\"live_trading\", \"Unknown\")}')
except: pass"
    fi
    
    # 4. Check for paper trades
    echo -e "\n4️⃣ PAPER TRADES:"
    paper_count=$(ssh ck 'grep -c "PAPER TRADE:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null' || echo "0")
    echo "   Total paper trades: $paper_count"
    if [ "$paper_count" -gt "0" ]; then
        echo "   Last paper trade:"
        ssh ck 'grep "PAPER TRADE:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -1 | cut -d" " -f5-'
    fi
    
    # 5. Check proximity blocks
    echo -e "\n5️⃣ PROXIMITY THRESHOLD:"
    blocks=$(ssh ck 'grep -c "NO_TRADE_PROXIMITY\|MAs too close" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null' || echo "0")
    echo "   Trades blocked by proximity: $blocks"
    
    # 6. Check for 1-minute candle bug
    echo -e "\n6️⃣ 1-MIN CANDLE STATUS:"
    candle_count=$(ssh ck 'grep -c "NEW 1-MIN CANDLE" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null' || echo "0")
    echo "   Candle transitions: $candle_count"
    if [ "$candle_count" -eq "0" ]; then
        echo "   ⚠️  Bug confirmed: No minute transitions"
    fi
    
    # 7. Activity rate
    echo -e "\n7️⃣ ACTIVITY:"
    eval_count=$(ssh ck 'grep -c "Strategy evaluation" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null' || echo "0")
    echo "   Total evaluations: $eval_count"
    echo "   Last activity: $(ssh ck 'tail -1 /home/chris/projects/bitstamp-testing/logs/tdr_server.log | grep -o "^[0-9-]* [0-9:]*"')"
    
    echo -e "\n⏱️  Refreshing in 10 seconds... (Ctrl+C to stop)"
    sleep 10
done