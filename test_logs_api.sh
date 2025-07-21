#!/bin/bash
# Test log API endpoints

SERVER_URL="http://localhost:4000"

echo "=== Testing Log API Endpoints ==="
echo

echo "1. Tail last 10 lines:"
curl -s "${SERVER_URL}/api/logs/tail?lines=10" | jq .

echo -e "\n2. Search for ENTRY_PRICE_DEBUG:"
curl -s "${SERVER_URL}/api/logs/grep?pattern=ENTRY_PRICE_DEBUG&lines=5" | jq .

echo -e "\n3. Get recent errors/warnings:"
curl -s "${SERVER_URL}/api/logs/errors?lines=5" | jq .

echo -e "\n4. Stream logs from last 2 minutes:"
curl -s "${SERVER_URL}/api/logs/stream?minutes=2&level=INFO" | jq '.logs[-5:]'

echo -e "\n=== Done ==="