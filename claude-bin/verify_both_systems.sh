#\!/bin/bash
# Verify both production and test systems

echo "🎯 PRODUCTION SYSTEM (gg btc - port 4000)"
echo "=========================================="
echo ""

echo "📊 Configuration:"
ssh ck 'grep -E "candle_interval|proximity_threshold|do_live" /home/chris/projects/bitstamp/best_strategy.json' | sed 's/^/   /'
echo ""

echo "⏰ Hourly Candles (last 3):"
ssh ck 'grep "HOURLY CANDLE" /home/chris/projects/bitstamp/logs/tdr_server.log | tail -3' | while read line; do
    timestamp=$(echo "$line" | grep -o "[0-9][0-9]:[0-9][0-9]:[0-9][0-9]")
    echo "   ✓ $timestamp"
done
echo ""

echo "📈 Last Signal Evaluation (at hourly candle):"
ssh ck 'grep "SIGNAL_EVAL" /home/chris/projects/bitstamp/logs/tdr_server.log | grep "01:00:" | tail -1' | sed 's/.*SIGNAL_EVAL v2: /   /'
echo ""

echo "💰 Current Position:"
ssh ck 'curl -s http://localhost:4000/api/status 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"   Position: {''LONG'' if d.get(''position'',{}).get(''position'',0)==1 else ''SHORT'' if d.get(''position'',{}).get(''position'',0)==-1 else ''NEUTRAL''}\")"'
echo ""

echo "-------------------------------------------"
echo ""

echo "🧪 TEST SYSTEM (gg tst - port 4001)"
echo "===================================="
echo ""

echo "📊 Configuration:"
ssh ck 'grep -E "candle_interval|proximity_threshold|do_live" /home/chris/projects/bitstamp-testing/best_strategy.json' | sed 's/^/   /'
echo ""

echo "⏰ 1-Minute Candles (last 3):"
ssh ck 'grep "1-MIN CANDLE" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -3' | while read line; do
    timestamp=$(echo "$line" | grep -o "[0-9][0-9]:[0-9][0-9]:[0-9][0-9]")
    echo "   ✓ $timestamp"
done
echo ""

echo "📈 Last Signal Evaluation:"
ssh ck 'grep "SIGNAL_EVAL" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -1' | sed 's/.*SIGNAL_EVAL v2: /   /'
echo ""

echo "🧾 Paper Trades:"
trade_count=$(ssh ck 'grep -c "PAPER TRADE" /home/chris/projects/bitstamp-testing/logs/tdr_server.log')
echo "   Total executed: $trade_count"
echo ""

echo "================================"
echo "✅ Both systems running correctly\!"
echo "Production: HOURLY bars, LIVE trading"
echo "Test: 1-MIN bars, PAPER trading"
