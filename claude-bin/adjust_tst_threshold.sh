#!/bin/bash
# Adjust proximity threshold on test server without restart

if [ $# -ne 1 ]; then
    echo "Usage: $0 <threshold>"
    echo "Example: $0 0.75"
    echo ""
    echo "Current thresholds to try:"
    echo "  0.3  - More sensitive (more trades)"
    echo "  0.5  - Current default"
    echo "  0.75 - Less sensitive"
    echo "  1.0  - Very conservative (fewer trades)"
    exit 1
fi

NEW_THRESHOLD=$1

echo "🔧 ADJUSTING PROXIMITY THRESHOLD"
echo "================================"
echo "New threshold: ${NEW_THRESHOLD}%"
echo ""

# 1. Update config
echo "1. Updating config..."
ssh ck "cd /home/chris/projects/bitstamp-testing && \
    sed -i 's/\"proximity_threshold\": [0-9.]*/\"proximity_threshold\": $NEW_THRESHOLD/' best_strategy.json"

# 2. Show new config
echo ""
echo "2. New config:"
ssh ck 'cd /home/chris/projects/bitstamp-testing && grep proximity_threshold best_strategy.json'

# 3. Update running code (hot patch)
echo ""
echo "3. Hot-patching running server..."
ssh ck "cd /home/chris/projects/bitstamp-testing && \
    sed -i 's/PROXIMITY_THRESHOLD = [0-9.]*/PROXIMITY_THRESHOLD = $NEW_THRESHOLD/' src/tdr_core/strategies.py"

echo ""
echo "✅ Threshold updated to ${NEW_THRESHOLD}%"
echo ""
echo "Note: This changes the hardcoded value temporarily."
echo "For permanent change, update the code and commit."
echo ""
echo "Monitor results with: ./claude-bin/monitor_1min_test.sh"