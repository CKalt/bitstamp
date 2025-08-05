#!/bin/bash
# Watch for the next hourly candle evaluation in real-time

echo "🕐 WATCHING FOR NEXT HOURLY CANDLE"
echo "=================================="
echo ""

# Calculate when next candle should appear
next_hour=$(date -d '+1 hour' +"%H:00")
minutes_left=$((60 - $(date +%-M)))
current_time=$(date +"%H:%M:%S")

echo "Current time: $current_time"
echo "Next candle: $next_hour:00 (in ~$minutes_left minutes)"
echo ""
echo "Watching logs for the transition..."
echo "Press Ctrl+C to stop"
echo ""

# Watch for the hourly candle message
ssh ck "tail -f /home/chris/projects/bitstamp/logs/tdr_server.log" | while read line; do
    # Highlight hourly candle messages
    if [[ $line =~ "NEW HOURLY CANDLE" ]]; then
        echo ""
        echo "🎉 ============ HOURLY CANDLE DETECTED ============"
        echo "$line" | grep --color=always "NEW HOURLY CANDLE"
        echo "=================================================="
        echo ""
    elif [[ $line =~ "Waiting for new hourly candle" ]]; then
        # Show waiting messages in gray
        echo -e "\033[90m$line\033[0m"
    elif [[ $line =~ "SIGNAL_EVAL v2" ]]; then
        # Highlight signal evaluations
        echo ""
        echo "📊 Signal Evaluation:"
        echo "$line" | sed 's/.*SIGNAL_EVAL v2://' | sed 's/,/\n   /g'
        echo ""
    elif [[ $line =~ "TRADE EXECUTED" ]] || [[ $line =~ "Executing trade" ]]; then
        # Alert on trades
        echo ""
        echo "🚨 ========== TRADE ALERT =========="
        echo "$line" | grep --color=always -E "(TRADE EXECUTED|Executing trade)"
        echo "==================================="
        echo ""
    fi
done