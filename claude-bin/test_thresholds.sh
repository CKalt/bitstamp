#!/bin/bash
# Test different proximity thresholds

echo "🧪 TESTING DIFFERENT THRESHOLDS"
echo "==============================="

# Test various thresholds
for threshold in 0.3 0.5 0.75 1.0; do
    echo -e "\n📊 Testing ${threshold}% threshold:"
    
    # Update the threshold in strategies.py
    ssh ck "cd /home/chris/projects/bitstamp && sed -i 's/PROXIMITY_THRESHOLD = [0-9.]*/PROXIMITY_THRESHOLD = $threshold/' src/tdr_core/strategies.py"
    
    echo "✓ Updated to ${threshold}%"
    echo "  Run for 1-2 hours and check:"
    echo "  - How many trades blocked"
    echo "  - How many flips prevented"
    echo "  - Theoretical P&L"
done