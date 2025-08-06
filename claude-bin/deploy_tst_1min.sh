#!/bin/bash
# Deploy 1-minute testing to gg tst instance

echo "🚀 DEPLOYING 1-MIN TESTING TO GG TST"
echo "===================================="
echo ""

# 1. Stop any existing test server
echo "1. Stopping existing test server..."
ssh ck 'screen -S server-tst -X quit 2>/dev/null || true'
echo "   ✓ Stopped"

# 2. Pull latest code
echo ""
echo "2. Pulling latest code to test instance..."
ssh ck 'cd /home/chris/projects/bitstamp-testing && git pull'

# 3. Create 1-minute config directly on server
echo ""
echo "3. Creating 1-minute test config..."
ssh ck 'cat > /home/chris/projects/bitstamp-testing/best_strategy.json << EOF
{
  "Short_Window": 4,
  "Long_Window": 20,
  "do_live_trades": false,
  "strategy_type": "MA",
  "enable_adaptive_strategy": false,
  "max_trades_per_day": 500,
  "proximity_threshold": 0.5,
  "candle_interval": "1min",
  "_comment": "1-MINUTE BARS for ultra-fast paper testing on gg tst"
}
EOF'
echo "   ✓ Config created"

# 4. Show the config
echo ""
echo "4. Verifying config:"
ssh ck 'cd /home/chris/projects/bitstamp-testing && cat best_strategy.json | grep -E "(do_live_trades|candle_interval|_comment)"'

# 5. Clear old logs for fresh start
echo ""
echo "5. Clearing old test logs..."
ssh ck 'cd /home/chris/projects/bitstamp-testing && rm -f logs/tdr_server.log && touch logs/tdr_server.log'
echo "   ✓ Logs cleared"

# 6. Start the server
echo ""
echo "6. Starting test server..."
ssh ck 'cd /home/chris/projects/bitstamp-testing && screen -dmS server-tst python src/tdr_server.py'
sleep 3

# 7. Verify it started
echo ""
echo "7. Verifying server started..."
if ssh ck 'screen -ls | grep -q server-tst'; then
    echo "   ✓ Server running in screen 'server-tst'"
else
    echo "   ✗ Server failed to start!"
    exit 1
fi

# 8. Check initial logs
echo ""
echo "8. Initial log output:"
ssh ck 'sleep 2 && tail -20 /home/chris/projects/bitstamp-testing/logs/tdr_server.log | grep -E "(PAPER TRADING|1-min|Starting)"'

echo ""
echo "✅ DEPLOYMENT COMPLETE!"
echo ""
echo "Next steps:"
echo "1. Connect client from btc-testing directory:"
echo "   cd /Users/chris/projects/python/btc-testing"
echo "   ./env/bin/python src/tdr.py"
echo ""
echo "2. Resume with short position:"
echo "   TDR> resume_auto_trade 0btc short 113793"
echo ""
echo "3. Monitor with:"
echo "   ./claude-bin/monitor_1min_test.sh"
echo ""
echo "Server is running on gg tst with 1-minute bars!"
echo "Production (gg btc) remains untouched."