#!/bin/bash
# Screen 1: Server Log Monitoring
# Shows SIGNAL_EVAL entries every 30 seconds

echo "📊 SCREEN 1: SERVER LOG MONITORING"
echo "=================================="
echo "This monitors signal evaluations on the server"
echo "You should see SIGNAL_EVAL entries every ~30 seconds"
echo ""
echo "Starting log monitor..."
echo ""

tail -f logs/tdr_server.log | grep -E "SIGNAL_EVAL|CHECK_FOR_SIGNALS|trigger|Executing trade|Buy signal|Sell signal" | while read line; do
    # Add timestamp for clarity
    echo "[$(date +%H:%M:%S)] $line"
done