#!/bin/bash
# Screen 4: Error and Warning Monitor
# Watches for any errors or issues

echo "🚨 SCREEN 4: ERROR & WARNING MONITOR"
echo "===================================="
echo "Monitoring for errors, warnings, and issues"
echo ""

tail -f logs/tdr_server.log | grep -E "ERROR|WARNING|CRITICAL|Failed|Exception|error|warning" | while read line; do
    # Skip known non-issues
    if echo "$line" | grep -q "validate_position_tracking"; then
        continue  # Skip if we see the old error
    fi
    
    # Highlight different severity levels
    if echo "$line" | grep -q "ERROR\|CRITICAL\|Failed"; then
        echo "❌ [$(date +%H:%M:%S)] $line"
    elif echo "$line" | grep -q "WARNING\|warning"; then
        echo "⚠️  [$(date +%H:%M:%S)] $line"
    else
        echo "[$(date +%H:%M:%S)] $line"
    fi
done