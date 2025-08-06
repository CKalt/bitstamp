#!/bin/bash
# Test proximity threshold with current data

echo "🧪 TESTING PROXIMITY THRESHOLD"
echo "=============================="
echo ""

# Get current MA values
echo "1. Checking current MA values..."
response=$(ssh ck 'curl -s -X POST http://localhost:4000/api/command -H "Content-Type: application/json" -d "{\"command\": \"status\"}"')

# Parse the response to look for MA info
echo "$response" | python3 -c "
import json
import sys
data = json.load(sys.stdin)
output = data.get('output', '')
print('Current status output:')
for line in output.split('\n'):
    if 'MA' in line or 'Crossover' in line or 'proximity' in line:
        print(f'  {line.strip()}')
"

echo ""
echo "2. Checking for proximity blocks in logs..."
ssh ck 'grep -c "NO_TRADE_PROXIMITY\|MAs too close" /home/chris/projects/bitstamp-testing/logs/tdr_server.log' | xargs -I {} echo "   Total proximity blocks: {}"

echo ""
echo "3. Most recent proximity check:"
ssh ck 'grep -E "(Prox=|proximity)" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -3'

echo ""
echo "Note: Even though 1-minute candles have a bug,"
echo "      the proximity threshold should still work"
echo "      when signals are evaluated."