#!/bin/bash
# Quick test server check

echo "🔍 TEST SERVER STATUS CHECK"
echo "=========================="
echo ""

# 1. Config
echo "1. Configuration:"
ssh ck 'cd /home/chris/projects/bitstamp-testing && grep -E "(do_live_trades|candle_interval|proximity)" best_strategy.json' | sed 's/^/   /'

# 2. Process
echo ""
echo "2. Server Process:"
ssh ck 'ps aux | grep -E "tdr_server.*testing" | grep -v grep | wc -l' | xargs -I {} echo "   Processes running: {}"

# 3. Recent activity
echo ""
echo "3. Recent Activity (last 2 minutes):"
ssh ck 'tail -100 /home/chris/projects/bitstamp-testing/logs/tdr_server.log | grep "$(date -u +"%Y-%m-%d %H:%M" -d "1 minute ago")\|$(date -u +"%Y-%m-%d %H:%M")" | grep -E "(SIGNAL_EVAL|PAPER|1-MIN|proximity)" | tail -5' | sed 's/^/   /'

# 4. Position
echo ""
echo "4. Current Position:"
ssh ck 'curl -s http://localhost:4000/api/status | python3 -m json.tool | grep -E "(position|balance)"' | sed 's/^/   /'

echo ""
echo "✅ Paper trading test is running on gg tst"
echo "📊 Waiting for 1-minute candle transitions..."