#!/bin/bash
# Test auto-resume behavior on test server

TEST_SERVER="http://localhost:4001"
TEST_DIR="/home/chris/projects/bitstamp-testing"

echo "🧪 AUTO-RESUME TEST SCRIPT"
echo "=========================="
echo ""

# Function to check server status
check_status() {
    ssh ck "curl -s $TEST_SERVER/api/status" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Server: {d.get(\"server\")}')
print(f'Auto-resume config: {d.get(\"auto_resume\")}')
print(f'Position: {d.get(\"position\", {}).get(\"position\", 0)}')
print(f'BTC: {d.get(\"position\", {}).get(\"btc_balance\", 0)}')
print(f'USD: {d.get(\"position\", {}).get(\"usd_balance\", 0)}')
print(f'Auto-trading: {d.get(\"auto_trader\", {}).get(\"active\", False)}')
"
}

# Function to check for resume file
check_resume_file() {
    echo "Checking for resume file..."
    ssh ck "ls -la $TEST_DIR/resume-auto-trade.json 2>/dev/null" || echo "No resume file found"
}

# Function to create test resume file
create_test_resume() {
    echo "Creating test resume file..."
    ssh ck "cat > $TEST_DIR/resume-auto-trade.json << 'EOF'
{
  \"timestamp\": \"$(date -Iseconds)\",
  \"position\": \"SHORT\",
  \"amount\": 10000,
  \"unit\": \"usd\",
  \"entry_price\": 113000,
  \"current_price\": 115000,
  \"unrealized_pnl\": -176.99,
  \"command\": \"resume_auto_trade 10000usd short 113000\",
  \"strategy\": {
    \"type\": \"MACrossoverStrategy\",
    \"short_window\": 4,
    \"long_window\": 20
  },
  \"balances\": {
    \"btc\": 0.0,
    \"usd\": 10000
  },
  \"trades_executed\": 0,
  \"last_trade_time\": null,
  \"trade_references\": [],
  \"pivot_protection\": {
    \"enabled\": false,
    \"tracker\": {}
  }
}
EOF"
}

# Function to restart server
restart_server() {
    echo "Restarting test server..."
    ssh ck "screen -S server-tst -X quit 2>/dev/null"
    sleep 2
    ssh ck "cd $TEST_DIR && screen -d -m -S server-tst bash -c 'source source-venv.sh && python src/tdr.py --server --port 4001'"
    echo "Waiting for server to start..."
    sleep 10
}

# Function to set auto_resume
set_auto_resume() {
    local value=$1
    echo "Setting auto_resume to $value..."
    ssh ck "cd $TEST_DIR && python3 -c \"
import json
with open('best_strategy.json', 'r') as f:
    config = json.load(f)
config['auto_resume'] = $value
with open('best_strategy.json', 'w') as f:
    json.dump(config, f, indent=2)
print(f'auto_resume set to {config[\\\"auto_resume\\\"]}')
\""
}

# Function to check server logs
check_logs() {
    echo "Recent server logs about resume:"
    ssh ck "tail -50 $TEST_DIR/logs/tdr_server.log | grep -E '(auto_resume|resume|Resume|AUTO-RESUME)' | tail -10"
}

# Main test sequence
echo "TEST 1: auto_resume = false with resume file"
echo "----------------------------------------------"
set_auto_resume "false"
create_test_resume
check_resume_file
restart_server
echo "Status after restart:"
check_status
echo ""
check_logs
echo ""

read -p "Press Enter to continue to TEST 2..."

echo "TEST 2: auto_resume = true with resume file"
echo "--------------------------------------------"
set_auto_resume "true"
# Resume file already exists from TEST 1
check_resume_file
restart_server
echo "Status after restart:"
check_status
echo ""
check_logs
echo ""

read -p "Press Enter to continue to TEST 3..."

echo "TEST 3: auto_resume = false, no resume file"
echo "--------------------------------------------"
set_auto_resume "false"
ssh ck "rm -f $TEST_DIR/resume-auto-trade.json"
check_resume_file
restart_server
echo "Status after restart:"
check_status
echo ""
check_logs
echo ""

echo "✅ Test complete!"
echo ""
echo "Summary:"
echo "- TEST 1: Should NOT auto-resume (auto_resume=false, file exists)"
echo "- TEST 2: Should auto-resume (auto_resume=true, file exists)"
echo "- TEST 3: Should NOT auto-resume (auto_resume=false, no file)"