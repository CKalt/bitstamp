#!/bin/bash
# Simple monitoring script for paper trading test

echo "📊 PAPER TRADING TEST MONITOR"
echo "============================="
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "📊 PAPER TRADING TEST MONITOR - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    
    # 1. SERVER STATUS
    echo -e "\n1️⃣ SERVER STATUS:"
    if ssh ck 'ps aux | grep -q "[t]dr_server.*testing"'; then
        echo "   ✅ Server is running"
        uptime=$(ssh ck 'ps aux | grep "[t]dr_server.*testing" | awk "{print \$9}"')
        echo "   Started at: $uptime"
    else
        echo "   ❌ SERVER IS DOWN!"
        echo "   Run: ssh ck 'cd /home/chris/projects/bitstamp-testing && screen -S server-tst -dm ./env/bin/python src/tdr_server.py'"
    fi
    
    # 2. POSITION & P&L
    echo -e "\n2️⃣ POSITION & P&L:"
    ssh ck 'curl -s http://localhost:4000/api/status 2>/dev/null' | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    pos = d['position']['position']
    size = d['position'].get('position_size', 0)
    entry = d['position']['entry_price']
    current = d.get('last_price', 0)
    
    if pos == 1:
        dir = 'LONG'
        pnl = (current - entry) * size
    elif pos == -1:
        dir = 'SHORT'
        pnl = (entry - current) * size
    else:
        dir = 'NEUTRAL'
        pnl = 0
    
    print(f'   Position: {dir}')
    print(f'   Size: {size:.8f} BTC')
    print(f'   Entry: \${entry:,.0f}')
    print(f'   Current: \${current:,.0f}')
    print(f'   P&L: \${pnl:,.2f}')
except:
    print('   ❌ Cannot get position data')
"
    
    # 3. RECENT TRADES
    echo -e "\n3️⃣ RECENT TRADES:"
    trade_count=$(ssh ck 'grep -c "PAPER TRADE:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null' || echo "0")
    echo "   Total paper trades: $trade_count"
    
    last_trade=$(ssh ck 'grep "PAPER TRADE:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null | tail -1')
    if [ ! -z "$last_trade" ]; then
        echo "   Last trade:"
        echo "     $last_trade" | sed 's/.*PAPER TRADE://'
    fi
    
    # 4. PROXIMITY THRESHOLD
    echo -e "\n4️⃣ PROXIMITY THRESHOLD STATUS:"
    recent_prox=$(ssh ck 'grep "SIGNAL_EVAL v2:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null | tail -5 | grep -o "Prox=[0-9.]*%"')
    if [ ! -z "$recent_prox" ]; then
        echo "   Recent proximity values:"
        echo "$recent_prox" | sed 's/^/     /'
    fi
    
    blocks=$(ssh ck 'grep -c "NO_TRADE_PROXIMITY" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null' || echo "0")
    echo "   Total trades blocked by proximity: $blocks"
    
    # 5. CANDLE FREQUENCY
    echo -e "\n5️⃣ CANDLE FREQUENCY:"
    candles_last_5min=$(ssh ck 'grep "NEW 1-MIN CANDLE" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null | tail -5 | wc -l')
    echo "   Candles in last 5 entries: $candles_last_5min (expect ~5)"
    
    # 6. ERRORS
    echo -e "\n6️⃣ ERROR CHECK:"
    error_count=$(ssh ck 'grep -ci "error\|exception" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null' || echo "0")
    if [ "$error_count" -gt "0" ]; then
        echo "   ⚠️ Total errors found: $error_count"
        echo "   Last error:"
        ssh ck 'grep -i "error\|exception" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null | tail -1' | sed 's/^/     /'
    else
        echo "   ✅ No errors found"
    fi
    
    echo -e "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Refreshing in 30 seconds... (Press Ctrl+C to stop)"
    
    sleep 30
done