#\!/bin/bash
# Monitor adaptive strategy performance

echo "🎯 ADAPTIVE STRATEGY MONITOR - $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
echo ""

# 1. Current Status
echo "📊 CURRENT STATUS:"
status=$(ssh ck 'curl -s http://localhost:4001/api/status 2>/dev/null')
echo "$status" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    pos = d.get('position', {}).get('position', 0)
    print(f'   Position: {\"LONG\" if pos == 1 else \"SHORT\" if pos == -1 else \"NEUTRAL\"}')
    print(f'   Entry: \${d.get(\"position\", {}).get(\"entry_price\", 0):,.0f}')
    print(f'   Current: \${d.get(\"last_price\", 0):,.0f}')
    print(f'   Strategy: {d.get(\"auto_trader\", {}).get(\"strategy\", \"Unknown\")}')
except: pass
"
echo ""

# 2. Market Regime
echo "🌡️ MARKET REGIME DETECTION:"
ssh ck 'tail -500 /home/chris/projects/bitstamp-testing/logs/tdr_server.log | grep -E "regime:|REGIME|Market identified as" | tail -3' | while read line; do
    echo "   $(echo "$line" | cut -d' ' -f5-)"
done
echo ""

# 3. Recent Trades
echo "📈 RECENT PAPER TRADES:"
trade_count=$(ssh ck 'grep -c "PAPER TRADE" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null' || echo "0")
echo "   Total trades: $trade_count"
ssh ck 'grep "PAPER TRADE" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -3' | while read line; do
    timestamp=$(echo "$line" | cut -d' ' -f1-2)
    action=$(echo "$line" | grep -o "Would [buy|sell]* [0-9.]* BTC @ \$[0-9,]*")
    echo "   $timestamp: $action"
done
echo ""

# 4. Strategy Switches
echo "🔄 STRATEGY SWITCHES:"
ssh ck 'grep -E "Switching to|Strategy switched|Active strategy" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -3' | while read line; do
    echo "   $(echo "$line" | cut -d' ' -f5-)"
done
echo ""

# 5. Current Indicators
echo "📊 CURRENT INDICATORS:"
ssh ck 'tail -100 /home/chris/projects/bitstamp-testing/logs/tdr_server.log | grep -E "MA4=|RSI=|MACD=|Volume=" | tail -1' | while read line; do
    echo "   $(echo "$line" | grep -o "MA4=[0-9]* MA20=[0-9]*\|RSI=[0-9.]*\|MACD=[0-9.]*")"
done
echo ""

echo "✅ Adaptive strategy is running with market regime detection"
echo "   • TRENDING → Uses MA crossover"
echo "   • RANGING → Uses RSI mean reversion"
echo "   • VOLATILE → Uses MACD breakout"
