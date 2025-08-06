#!/bin/bash
# Comprehensive bug monitoring for paper trading

echo "🐛 PAPER TRADING BUG MONITOR"
echo "============================"
echo "Starting at: $(date)"
echo ""

# Function to check for errors
check_errors() {
    local errors=$(ssh ck 'grep -i "error\|exception\|traceback" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -5')
    if [ ! -z "$errors" ]; then
        echo "❌ ERRORS DETECTED:"
        echo "$errors"
        return 1
    fi
    return 0
}

# Function to monitor candle transitions
monitor_candles() {
    echo -e "\n📊 Candle Transitions:"
    ssh ck 'grep "NEW 1-MIN CANDLE" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -5' || echo "   No candle transitions yet"
}

# Function to check signal evaluations
check_signals() {
    echo -e "\n📈 Recent Signal Evaluations:"
    ssh ck 'grep "SIGNAL_EVAL v2:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -3'
}

# Function to check proximity blocks
check_proximity() {
    echo -e "\n🚫 Proximity Threshold Activity:"
    local blocks=$(ssh ck 'grep -c "NO_TRADE_PROXIMITY" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null' || echo "0")
    echo "   Total proximity blocks: $blocks"
    ssh ck 'grep "MAs too close" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -3'
}

# Function to check paper trades
check_paper_trades() {
    echo -e "\n🧪 Paper Trades:"
    ssh ck 'grep "PAPER TRADE:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -3' || echo "   No paper trades yet"
}

# Function to check system health
check_health() {
    echo -e "\n💓 System Health:"
    
    # Check if process is running
    local proc_count=$(ssh ck 'ps aux | grep "tdr_server.*testing" | grep -v grep | wc -l')
    echo -n "   Server process: "
    if [ "$proc_count" -gt 0 ]; then
        echo "✅ Running"
    else
        echo "❌ NOT RUNNING!"
        return 1
    fi
    
    # Check last heartbeat
    local last_heartbeat=$(ssh ck 'grep "HEARTBEAT" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -1 | grep -o "[0-9][0-9]:[0-9][0-9]:[0-9][0-9]"')
    echo "   Last heartbeat: $last_heartbeat"
    
    # Check data flow
    local last_log_time=$(ssh ck 'tail -1 /home/chris/projects/bitstamp-testing/logs/tdr_server.log | grep -o "^[0-9-]* [0-9:]*"')
    echo "   Last log entry: $last_log_time"
}

# Main monitoring loop
iteration=0
while true; do
    clear
    echo "🐛 PAPER TRADING BUG MONITOR - Iteration $((++iteration))"
    echo "============================"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Test Instance: gg tst (paper mode)"
    
    # Run all checks
    check_health || echo "⚠️  HEALTH CHECK FAILED"
    echo ""
    
    if ! check_errors; then
        echo -e "\n🚨 ERRORS FOUND - CHECK IMMEDIATELY!"
    else
        echo -e "\n✅ No errors detected"
    fi
    
    monitor_candles
    check_signals
    check_proximity
    check_paper_trades
    
    # Summary
    echo -e "\n📊 Summary:"
    local total_evals=$(ssh ck 'grep -c "Strategy evaluation" /home/chris/projects/bitstamp-testing/logs/tdr_server.log' || echo "0")
    echo "   Total evaluations: $total_evals"
    echo "   Monitoring for: $((iteration * 30)) seconds"
    
    echo -e "\nRefreshing in 30 seconds... (Ctrl+C to stop)"
    sleep 30
done