#!/bin/bash
# Screen 1: Server Log Monitoring
# Shows SIGNAL_EVAL entries every 30 seconds

# Check if we're in the right directory
if [ ! -f logs/tdr_server.log ]; then
    echo "❌ ERROR: logs/tdr_server.log not found!"
    echo "Make sure you're in /home/chris/projects/bitstamp"
    exit 1
fi

echo "📊 SCREEN 1: SERVER LOG MONITORING"
echo "=================================="
echo "This monitors signal evaluations on the server"
echo "You should see SIGNAL_EVAL entries every ~30 seconds"
echo ""

# Show last 5 matching entries first
echo "Recent entries:"
echo "---------------"
tail -100 logs/tdr_server.log | grep -E "SIGNAL_EVAL|CHECK_FOR_SIGNALS|trigger|Executing trade|Buy signal|Sell signal" | tail -5
echo ""
echo "Now monitoring for new entries..."
echo "==============================================="

# Use stdbuf to disable all buffering
stdbuf -o0 -e0 tail -f logs/tdr_server.log | stdbuf -o0 -e0 grep -E "SIGNAL_EVAL|CHECK_FOR_SIGNALS|trigger|Executing trade|Buy signal|Sell signal"