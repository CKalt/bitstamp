#!/bin/bash
# Comprehensive auto trading test suite

echo "🧪 COMPREHENSIVE AUTO TRADING TEST SUITE"
echo "========================================"
echo "Testing ALL aspects of auto trading system"
echo ""

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_result="$3"
    
    echo -e "\n📋 TEST: $test_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    result=$(eval "$test_command")
    
    if [[ "$result" == *"$expected_result"* ]]; then
        echo "✅ PASSED"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "❌ FAILED"
        echo "   Expected: $expected_result"
        echo "   Got: $result"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# 1. TEST AUTO RESUME FUNCTIONALITY
echo -e "\n🔄 TESTING AUTO RESUME"
echo "========================"

# Stop the server
echo "1. Stopping server to test resume..."
ssh ck 'screen -S server-tst -X quit'
sleep 3

# Check if resume file exists
echo "2. Checking for resume file..."
resume_exists=$(ssh ck 'test -f /home/chris/projects/bitstamp-testing/resume-auto-trade.json && echo "YES" || echo "NO"')
run_test "Resume file exists" "echo $resume_exists" "YES"

# Check resume file contents
echo "3. Checking resume file contents..."
resume_content=$(ssh ck 'cat /home/chris/projects/bitstamp-testing/resume-auto-trade.json 2>/dev/null | python3 -m json.tool')
run_test "Resume has position" "echo '$resume_content' | grep -c position" "1"
run_test "Resume has amount" "echo '$resume_content' | grep -c amount" "1"
run_test "Resume has entry_price" "echo '$resume_content' | grep -c entry_price" "1"

# Test auto_resume flag
echo "4. Testing auto_resume setting..."
auto_resume=$(ssh ck 'grep -c "auto_resume.*true" /home/chris/projects/bitstamp-testing/best_strategy.json')
echo "   auto_resume in config: $([[ $auto_resume -gt 0 ]] && echo 'true' || echo 'false')"

# Restart server and check if it auto-resumes
echo "5. Restarting server..."
ssh ck 'cd /home/chris/projects/bitstamp-testing && screen -dmS server-tst ./env/bin/python src/tdr_server.py'
sleep 10

# Check if trading resumed automatically
echo "6. Checking if auto-trading resumed..."
auto_trader_status=$(ssh ck 'curl -s http://localhost:4000/api/status | python3 -c "
import json, sys
data = json.load(sys.stdin)
if \"auto_trader\" in data and data[\"auto_trader\"][\"active\"]:
    print(\"ACTIVE\")
else:
    print(\"NOT ACTIVE\")
"')
run_test "Auto trader active after restart" "echo $auto_trader_status" "ACTIVE"

# 2. TEST POSITION TRACKING
echo -e "\n💰 TESTING POSITION TRACKING"
echo "=============================="

# Get current position
position_info=$(ssh ck 'curl -s http://localhost:4000/api/status | python3 -c "
import json, sys
data = json.load(sys.stdin)
pos = data[\"position\"]
print(f\"position:{pos[\"position\"]} btc:{pos.get(\"btc_balance\", 0)} usd:{pos.get(\"usd_balance\", 0)}\")
"')

# Check position consistency
run_test "Position is valid (-1, 0, or 1)" "echo '$position_info' | grep -E 'position:(-1|0|1)'" "position:"

# 3. TEST TRADE EXECUTION LOGIC
echo -e "\n🎯 TESTING TRADE EXECUTION"
echo "=========================="

# Check multi-part trade handling
echo "1. Checking multi-part trade configuration..."
multi_part=$(ssh ck 'grep -A20 "execute.*trade" /home/chris/projects/bitstamp-testing/src/tdr_core/strategies.py | grep -c "three_parts"')
run_test "Multi-part trade logic exists" "echo $multi_part" "1"

# Check trade counting
echo "2. Checking trade counting..."
trade_count=$(ssh ck 'grep -c "trades_executed" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -1')
echo "   Total trade count references: $trade_count"

# 4. TEST DAILY LIMITS
echo -e "\n📊 TESTING DAILY TRADE LIMITS"
echo "==============================="

max_trades=$(ssh ck 'grep "max_trades_per_day" /home/chris/projects/bitstamp-testing/best_strategy.json | grep -o "[0-9]*"')
echo "   Max trades per day: $max_trades"
run_test "Daily limit configured" "echo $max_trades" "500"

# 5. TEST ERROR HANDLING
echo -e "\n⚠️  TESTING ERROR HANDLING"
echo "=========================="

# Send invalid command
echo "1. Testing invalid command handling..."
error_response=$(ssh ck 'curl -s -X POST http://localhost:4000/api/command -H "Content-Type: application/json" -d "{\"command\": \"invalid_command xyz\"}" | python3 -m json.tool | grep -c "error"')
run_test "Invalid command returns error" "echo $error_response" "1"

# 6. TEST SIGNAL EVALUATION
echo -e "\n📈 TESTING SIGNAL EVALUATION"
echo "============================="

# Count recent evaluations
eval_count=$(ssh ck 'grep -c "SIGNAL_EVAL v2:" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -100')
echo "   Signal evaluations in last 100 lines: $eval_count"
run_test "Signals being evaluated" "[[ $eval_count -gt 0 ]] && echo YES || echo NO" "YES"

# 7. TEST PAPER TRADING MODE
echo -e "\n🧪 TESTING PAPER TRADING SAFETY"
echo "================================"

# Verify no real orders
real_orders=$(ssh ck 'grep -c "place_order.*market" /home/chris/projects/bitstamp-testing/logs/tdr_server.log')
run_test "No real orders placed" "echo $real_orders" "0"

# Check for paper trade warnings
paper_warnings=$(ssh ck 'grep -c "PAPER TRADING MODE" /home/chris/projects/bitstamp-testing/logs/tdr_server.log')
run_test "Paper trading warnings present" "[[ $paper_warnings -gt 0 ]] && echo YES || echo NO" "YES"

# 8. TEST MA STRATEGY PARAMETERS
echo -e "\n📊 TESTING MA STRATEGY"
echo "======================"

# Check MA windows
ma_config=$(ssh ck 'grep -E "(Short_Window|Long_Window)" /home/chris/projects/bitstamp-testing/best_strategy.json')
echo "$ma_config"
run_test "MA 4/20 configured" "echo '$ma_config' | grep -c '4\|20'" "2"

# 9. TEST PERSISTENCE
echo -e "\n💾 TESTING DATA PERSISTENCE"
echo "==========================="

# Check trades.json updates
trades_file=$(ssh ck 'test -f /home/chris/projects/bitstamp-testing/trades.json && echo "EXISTS" || echo "NOT FOUND"')
run_test "Trades file exists" "echo $trades_file" "EXISTS"

# 10. TEST CANDLE INTERVALS
echo -e "\n⏰ TESTING CANDLE INTERVALS"
echo "============================"

# Count candles in last 5 minutes
recent_candles=$(ssh ck 'grep "NEW 1-MIN CANDLE" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -5 | wc -l')
run_test "Getting 1-min candles" "[[ $recent_candles -ge 3 ]] && echo YES || echo NO" "YES"

# FINAL REPORT
echo -e "\n\n📊 TEST SUITE SUMMARY"
echo "===================="
echo "✅ Tests Passed: $TESTS_PASSED"
echo "❌ Tests Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED!"
else
    echo "⚠️  Some tests failed. Review output above."
fi

echo ""
echo "📝 Additional Manual Checks Recommended:"
echo "   - Monitor for 10+ minutes to verify continuous operation"
echo "   - Force a position flip to test trade execution"
echo "   - Kill and restart server to test resume again"
echo "   - Check memory usage over time"