#!/bin/bash
# Verify 1-minute candles are working

echo "⏱️  VERIFYING 1-MINUTE CANDLES"
echo "============================="
echo "Should see new candle every 60 seconds..."
echo ""

# Monitor for 3 minutes
start_time=$(date +%s)
last_count=0

while [ $(($(date +%s) - start_time)) -lt 180 ]; do
    current_time=$(date '+%H:%M:%S')
    
    # Count candle transitions
    candle_count=$(ssh ck 'grep -c "NEW 1-MIN CANDLE" /home/chris/projects/bitstamp-testing/logs/tdr_server.log 2>/dev/null' || echo "0")
    
    # Check if new candle appeared
    if [ "$candle_count" -gt "$last_count" ]; then
        echo "✅ [$current_time] NEW CANDLE! Total: $candle_count"
        
        # Show the last candle message
        ssh ck 'grep "NEW 1-MIN CANDLE" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -1'
    else
        echo "   [$current_time] Waiting... (Total candles: $candle_count)"
    fi
    
    last_count=$candle_count
    sleep 10
done

echo ""
echo "Test complete. Expected ~3 candles in 3 minutes."