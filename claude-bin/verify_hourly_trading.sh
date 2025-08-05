#!/bin/bash
# Verify hourly trading is working correctly

echo "🔍 HOURLY TRADING VERIFICATION"
echo "=============================="
echo ""

# 1. Check recent logs for hourly candle messages
echo "📊 Recent Hourly Candle Checks:"
echo "-------------------------------"
ssh ck "grep -E '(NEW HOURLY CANDLE|Waiting for new hourly candle)' /home/chris/projects/bitstamp/logs/tdr_server.log | tail -10"

echo ""
echo "⏰ Trade Timing Analysis:"
echo "------------------------"
# Check if trades only happen at hour boundaries
ssh ck "grep 'TRADE EXECUTED' /home/chris/projects/bitstamp/logs/tdr_server.log | tail -5" | while read line; do
    timestamp=$(echo "$line" | grep -o '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\} [0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}')
    if [ ! -z "$timestamp" ]; then
        minute=$(echo "$timestamp" | cut -d: -f2)
        echo "Trade at $timestamp - Minute: $minute"
    fi
done

echo ""
echo "📈 Signal Evaluation Frequency:"
echo "------------------------------"
# Count evaluations per hour
current_hour=$(date +"%Y-%m-%d %H")
echo "Evaluations in current hour ($current_hour:xx):"
ssh ck "grep 'SIGNAL_EVAL v2' /home/chris/projects/bitstamp/logs/tdr_server.log | grep \"$current_hour\" | wc -l"

echo ""
echo "🎯 Next Evaluation Time:"
echo "-----------------------"
# Calculate next hourly evaluation
next_hour=$(date -d '+1 hour' +"%H:00")
minutes_left=$((60 - $(date +%-M)))
echo "Next hourly candle: $next_hour (in ~$minutes_left minutes)"

echo ""
echo "✅ System Health Checks:"
echo "-----------------------"
# Check if we're getting regular heartbeats
recent_heartbeat=$(ssh ck "grep HEARTBEAT /home/chris/projects/bitstamp/logs/tdr_server.log | tail -1")
if [ ! -z "$recent_heartbeat" ]; then
    echo "✓ Heartbeats active"
else
    echo "✗ No recent heartbeats!"
fi

# Check if we're waiting between evaluations
waiting_msgs=$(ssh ck "grep 'Waiting for new hourly candle' /home/chris/projects/bitstamp/logs/tdr_server.log | tail -1")
if [ ! -z "$waiting_msgs" ]; then
    echo "✓ Properly waiting between hourly checks"
else
    echo "✗ Not seeing wait messages!"
fi

echo ""
echo "📊 Current Status:"
echo "-----------------"
# Get latest signal evaluation
ssh ck "grep 'SIGNAL_EVAL v2' /home/chris/projects/bitstamp/logs/tdr_server.log | tail -1" | \
    sed 's/.*MA4=/MA4=/' | \
    sed 's/,/\n   /g'