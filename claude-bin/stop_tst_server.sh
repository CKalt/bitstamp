#!/bin/bash
# Stop test server cleanly

echo "🛑 STOPPING TEST SERVER"
echo "======================="
echo ""

# 1. Check if running
if ssh ck 'screen -ls | grep -q server-tst'; then
    echo "Found test server running..."
    
    # 2. Send quit command
    ssh ck 'screen -S server-tst -X quit'
    echo "✓ Sent stop command"
    
    # 3. Wait and verify
    sleep 2
    if ssh ck 'screen -ls | grep -q server-tst'; then
        echo "⚠️  Server still running, force killing..."
        ssh ck 'screen -S server-tst -X kill'
    else
        echo "✓ Server stopped successfully"
    fi
else
    echo "No test server found running"
fi

# 4. Show final log entries
echo ""
echo "Last log entries:"
ssh ck 'tail -10 /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null | grep -E "(Shutting down|ERROR|PAPER TRADE)"' || echo "No recent logs"

echo ""
echo "✅ Test server stopped"
echo "Production server (gg btc) was not affected"