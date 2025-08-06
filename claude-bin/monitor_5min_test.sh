#!/bin/bash
# Monitor 5-minute bar testing for rapid feedback

echo "🔬 5-MINUTE BAR TEST MONITOR"
echo "============================"
echo "Purpose: Rapid bug detection"
echo ""

# Check if we need to modify the candle interval check
echo "📝 First, let's add 5-minute candle support to strategies.py..."
echo ""

# Function to show recent signals
show_recent_signals() {
    echo "📊 Last 10 Signal Evaluations (5-min bars):"
    ssh ck 'tail -200 /home/chris/projects/bitstamp/logs/tdr_server.log | grep "SIGNAL_EVAL v2:" | tail -10' | \
        awk '{print $1 " " $2 " - " substr($0, index($0,"MA4"))}'
}

# Function to show proximity blocks
show_proximity_blocks() {
    echo -e "\n🚫 Proximity Blocks (last 24h):"
    local count=$(ssh ck 'grep -c "NO_TRADE_PROXIMITY" /home/chris/projects/bitstamp/logs/tdr_server.log')
    echo "Total blocks: $count"
    ssh ck 'grep "NO_TRADE_PROXIMITY" /home/chris/projects/bitstamp/logs/tdr_server.log | tail -5'
}

# Function to show would-have trades
show_paper_trades() {
    echo -e "\n💰 Paper Trades (would have executed):"
    ssh ck 'grep -E "(WOULD.*EXECUTE|WILL_BUY|WILL_SELL)" /home/chi/projects/bitstamp/logs/tdr_server.log | tail -5'
}

# Function to check for errors
check_errors() {
    echo -e "\n⚠️  Recent Errors/Warnings:"
    ssh ck 'grep -E "(ERROR|WARN|Exception)" /home/chris/projects/bitstamp/logs/tdr_server.log | tail -5' || echo "No errors found ✅"
}

# Main monitoring loop
echo "Starting monitoring..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "🔬 5-MINUTE BAR TEST MONITOR"
    echo "============================"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Mode: PAPER TRADING (5-min bars)"
    echo ""
    
    show_recent_signals
    show_proximity_blocks
    show_paper_trades
    check_errors
    
    echo -e "\n📈 Signal Rate:"
    echo "Expecting ~12 signals/hour with 5-min bars"
    echo "vs 1 signal/hour with hourly bars"
    
    sleep 30
done