#!/bin/bash
# Real-time paper trading dashboard

echo "📊 PAPER TRADING DASHBOARD"
echo "========================="
echo "Mode: TESTING (no real trades)"
echo ""

# Function to show current status
show_status() {
    echo -e "\n🔍 Current Status:"
    ssh ck 'tail -100 /home/chris/projects/bitstamp/logs/tdr_server.log | grep "SIGNAL_EVAL v2:" | tail -1' | \
        sed 's/.*SIGNAL_EVAL v2://' | \
        awk '{
            print "  MA4: $" $1 " | MA20: $" $3
            print "  Proximity: " $5
            print "  Signal: " $6 " | Position: " $7
            print "  Action: " $8
        }' | sed 's/MA[0-9]*=//g' | sed 's/Diff=[-0-9]*//g' | sed 's/Prox=//g' | sed 's/Sig=//g' | sed 's/Pos=//g' | sed 's/Action=//g'
}

# Function to show recent proximity blocks
show_blocks() {
    echo -e "\n🚫 Recent Proximity Blocks:"
    ssh ck 'grep "MAs too close" /home/chris/projects/bitstamp/logs/tdr_server.log | tail -5' | \
        awk '{print "  " $1 " " $2 " - " $10 " " $11 " " $12 " " $13}'
}

# Function to show theoretical trades
show_trades() {
    echo -e "\n💰 Theoretical Trades (if threshold was off):"
    ssh ck 'grep -E "(WOULD EXECUTE|Simulated trade)" /home/chris/projects/bitstamp/logs/tdr_server.log | tail -5' | \
        awk '{print "  " $1 " " $2 " - " substr($0, index($0,$3))}'
}

# Main loop
while true; do
    clear
    echo "📊 PAPER TRADING DASHBOARD"
    echo "========================="
    echo "Mode: TESTING (no real trades)"
    echo "Proximity Threshold: 0.5%"
    echo "Last Update: $(date '+%H:%M:%S')"
    
    show_status
    show_blocks
    show_trades
    
    echo -e "\n📈 Statistics:"
    ssh ck 'cd /home/chris/projects/bitstamp && python3 /tmp/monitor_paper_trades.py logs/tdr_server.log 2>/dev/null | grep -E "(blocked|would execute|savings)"' || echo "  Calculating..."
    
    echo -e "\nPress Ctrl+C to exit"
    sleep 30
done