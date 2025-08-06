#!/bin/bash
# Verify production is running correctly with proximity threshold

echo "🔍 PRODUCTION VERIFICATION"
echo "========================="
echo ""

# Check server status
echo "1️⃣ SERVER STATUS:"
if ssh ck 'screen -ls | grep -q "server[^-]"'; then
    echo "   ✅ Production server is running"
else
    echo "   ❌ Server NOT running!"
    exit 1
fi

# Check position
echo ""
echo "2️⃣ POSITION CHECK:"
ssh ck 'curl -s http://localhost:4000/api/status 2>/dev/null' | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    pos = d['position']['position']
    if pos == -1:
        print('   ✅ Position: SHORT (correct)')
    elif pos == 1:
        print('   ⚠️ Position: LONG (should be SHORT!)')
    else:
        print('   ⚠️ Position: NEUTRAL (should be SHORT!)')
    
    print(f'   Balance: \${d[\"position\"].get(\"usd_balance\", 0):,.2f} USD')
    print(f'   Auto Trading: {d.get(\"auto_trader\", {}).get(\"active\", False)}')
except Exception as e:
    print(f'   ❌ Error checking status: {e}')
"

# Check for proximity blocking
echo ""
echo "3️⃣ PROXIMITY THRESHOLD:"
recent_blocks=$(ssh ck 'tail -1000 /home/chris/projects/bitstamp/logs/tdr_server.log | grep -c "BLOCKING TRADE" 2>/dev/null' || echo "0")
if [ "$recent_blocks" -gt "0" ]; then
    echo "   ✅ Proximity blocking is active ($recent_blocks blocks in recent logs)"
else
    echo "   ⏳ No blocks yet (MAs may be diverged or just started)"
fi

echo ""
echo "4️⃣ MONITORING COMMANDS:"
echo "   Watch logs:  ssh ck 'tail -f /home/chris/projects/bitstamp/logs/tdr_server.log'"
echo "   Check status: curl http://localhost:4000/api/status"
echo ""
echo "⚠️ IMPORTANT: Monitor for first few hours to ensure:"
echo "   - Position stays SHORT until signal changes"
echo "   - Proximity threshold blocks trades when MAs < 0.3%"
echo "   - No excessive flipping"