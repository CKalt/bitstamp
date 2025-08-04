#!/bin/bash
# Monitor signal evaluations in real-time

echo "🔍 Real-time Signal Monitoring"
echo "============================="
echo ""

# Function to check signal monitor status
check_signal_status() {
    echo -e "\n📊 Signal Monitor Status:"
    curl -s http://localhost:4000/api/signal_monitor/status 2>/dev/null | jq '.' || echo "❌ Could not fetch status"
}

# Function to get recent evaluations
get_recent_evaluations() {
    echo -e "\n📈 Recent Signal Evaluations:"
    curl -s http://localhost:4000/api/signal_monitor/recent 2>/dev/null | jq '.' || echo "❌ Could not fetch evaluations"
}

# Function to check via logs
check_via_logs() {
    echo -e "\n📜 Latest Signal Checks from Log:"
    ssh ck "grep -E 'SIGNAL_CHECK|SIGNAL_EVAL|HEARTBEAT' ~/projects/bitstamp/logs/tdr_server.log | tail -10"
}

# Main monitoring loop
while true; do
    clear
    echo "🔍 Real-time Signal Monitoring - $(date)"
    echo "============================="
    
    # Check current price and position
    echo -e "\n💰 Current Status:"
    curl -s http://localhost:4000/api/status 2>/dev/null | jq '{
        price: .last_price,
        position: .position.position,
        entry: .position.entry_price,
        pnl: .position.unrealized_pnl
    }' || echo "❌ Could not fetch status"
    
    # Check signal monitor status
    check_signal_status
    
    # Show recent evaluations
    get_recent_evaluations
    
    # Check heartbeats
    echo -e "\n💓 Last 3 Heartbeats:"
    ssh ck "grep 'HEARTBEAT' ~/projects/bitstamp/logs/tdr_server.log | tail -3"
    
    # Check for missed signals
    echo -e "\n⚠️  Checking for Missed Signals:"
    SIGNAL_LOG="/home/chris/projects/bitstamp/logs/signal_monitor_$(date +%Y%m%d).json"
    ssh ck "if [ -f $SIGNAL_LOG ]; then jq '.missed_count' $SIGNAL_LOG 2>/dev/null || echo '0'; else echo 'No signal log found'; fi"
    
    echo -e "\n[Press Ctrl+C to exit, refreshing in 30 seconds...]"
    sleep 30
done