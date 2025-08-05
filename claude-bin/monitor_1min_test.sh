#!/bin/bash
# Monitor 1-minute testing - ultra fast feedback

echo "⚡ 1-MINUTE TEST MONITOR (gg tst)"
echo "================================="
echo "Expecting ~60 signals per hour!"
echo ""

# Function to show last 20 signals (more for 1-min)
show_signals() {
    echo "📊 Recent Signals (1-min bars):"
    ssh ck 'tail -500 /home/chris/projects/bitstamp-testing/logs/tdr_server.log | grep "SIGNAL_EVAL v2:" | tail -20' | \
        awk '{print $1 " " $2 " " substr($0, index($0,"Prox"))}'
}

# Function to count trades per hour
count_activity() {
    echo -e "\n📈 Activity Rate:"
    local current_hour=$(date +"%Y-%m-%d %H")
    local signals=$(ssh ck "grep -c 'SIGNAL_EVAL v2:' /home/chris/projects/bitstamp-testing/logs/tdr_server.log | grep '$current_hour'" || echo "0")
    local blocks=$(ssh ck "grep -c 'NO_TRADE_PROXIMITY' /home/chris/projects/bitstamp-testing/logs/tdr_server.log" || echo "0")
    local paper_trades=$(ssh ck "grep -c 'PAPER TRADE:' /home/chris/projects/bitstamp-testing/logs/tdr_server.log" || echo "0")
    
    echo "Signals this hour: ~$signals (expect ~60)"
    echo "Proximity blocks total: $blocks"
    echo "Paper trades total: $paper_trades"
}

# Main loop
while true; do
    clear
    echo "⚡ 1-MINUTE TEST MONITOR (gg tst)"
    echo "================================="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Mode: PAPER TRADING (1-min bars)"
    echo "Instance: TEST (not production)"
    echo ""
    
    show_signals
    count_activity
    
    echo -e "\n🧪 Last Paper Trade:"
    ssh ck 'grep "PAPER TRADE:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -3'
    
    echo -e "\nRefreshing every 10 seconds..."
    sleep 10
done