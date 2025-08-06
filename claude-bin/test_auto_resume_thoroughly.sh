#!/bin/bash
# Thorough auto-resume testing

echo "🔄 THOROUGH AUTO-RESUME TESTING"
echo "==============================="
echo ""

# Function to check server state
check_state() {
    local desc="$1"
    echo -e "\n📊 $desc"
    echo "-------------------"
    
    # Get status
    status=$(ssh ck 'curl -s http://localhost:4000/api/status 2>/dev/null')
    
    if [ -z "$status" ]; then
        echo "❌ Server not responding"
        return 1
    fi
    
    # Parse status
    echo "$status" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    pos = data['position']
    at = data.get('auto_trader', {})
    
    print(f'Server: ✅ Running')
    print(f'Position: {pos[\"position\"]} ({\"LONG\" if pos[\"position\"] > 0 else \"SHORT\" if pos[\"position\"] < 0 else \"NEUTRAL\"})')
    print(f'BTC: {pos.get(\"btc_balance\", 0):.8f}')
    print(f'USD: {pos.get(\"usd_balance\", 0):.2f}')
    print(f'Auto Trading: {\"✅ Active\" if at.get(\"active\") else \"❌ Inactive\"}')
    print(f'Live Trading: {\"❌ NO (Paper)\" if not data.get(\"live_trading\") else \"✅ YES\"}')
except Exception as e:
    print(f'❌ Error parsing status: {e}')
"
    
    # Check resume file
    echo -e "\nResume File:"
    ssh ck 'if [ -f /home/chris/projects/bitstamp-testing/resume-auto-trade.json ]; then
        echo "✅ Exists"
        cat /home/chris/projects/bitstamp-testing/resume-auto-trade.json | python3 -m json.tool | head -10
    else
        echo "❌ Not found"
    fi'
}

# TEST 1: Current State
echo "TEST 1: Checking current state"
check_state "INITIAL STATE"

# TEST 2: Stop server gracefully
echo -e "\n\nTEST 2: Stopping server gracefully"
echo "=================================="
ssh ck 'curl -s -X POST http://localhost:4000/api/command -H "Content-Type: application/json" -d "{\"command\": \"stop\"}"' > /dev/null 2>&1
sleep 2
ssh ck 'screen -S server-tst -X quit' 2>/dev/null
sleep 3

echo "✅ Server stopped"

# TEST 3: Check what was saved
echo -e "\n\nTEST 3: Checking saved state"
echo "============================"
echo "Resume file contents:"
ssh ck 'cat /home/chris/projects/bitstamp-testing/resume-auto-trade.json | python3 -m json.tool'

echo -e "\nBackup files:"
ssh ck 'ls -la /home/chris/projects/bitstamp-testing/resume-auto-trade.json* | tail -5'

# TEST 4: Restart and check auto-resume
echo -e "\n\nTEST 4: Testing auto-resume on restart"
echo "======================================"

# Check auto_resume setting
echo "Checking auto_resume setting:"
auto_resume_enabled=$(ssh ck 'grep -o "\"auto_resume\": *[^,}]*" /home/chris/projects/bitstamp-testing/best_strategy.json')
echo "   Config: $auto_resume_enabled"

# Start server
echo -e "\nStarting server..."
ssh ck 'cd /home/chris/projects/bitstamp-testing && screen -dmS server-tst ./env/bin/python src/tdr_server.py'
echo "Waiting for initialization..."
sleep 10

# Check if it resumed
check_state "STATE AFTER RESTART"

# TEST 5: Manual resume test
echo -e "\n\nTEST 5: Testing manual resume command"
echo "====================================="

# First stop auto trading
echo "Stopping auto trading..."
ssh ck 'curl -s -X POST http://localhost:4000/api/command -H "Content-Type: application/json" -d "{\"command\": \"stop\"}"' | python3 -m json.tool | grep -E "(success|output)"

sleep 2

# Now resume manually
echo -e "\nResuming manually..."
ssh ck 'curl -s -X POST http://localhost:4000/api/command -H "Content-Type: application/json" -d "{\"command\": \"resume_auto_trade\"}"' | python3 -m json.tool | grep -E "(success|output|error)"

sleep 3
check_state "STATE AFTER MANUAL RESUME"

# TEST 6: Test resume with position change
echo -e "\n\nTEST 6: Testing resume with different position"
echo "=============================================="

# Stop trading
ssh ck 'curl -s -X POST http://localhost:4000/api/command -H "Content-Type: application/json" -d "{\"command\": \"stop\"}"' > /dev/null 2>&1

# Start with different position
echo "Starting with opposite position..."
current_pos=$(ssh ck 'curl -s http://localhost:4000/api/status | python3 -c "import json,sys; print(json.load(sys.stdin)[\"position\"][\"position\"])"')

if [ "$current_pos" == "1" ]; then
    new_pos="short"
    new_cmd="resume_auto_trade 156574usd short 113800"
else
    new_pos="long"
    new_cmd="resume_auto_trade 1.37btc long 113800"
fi

echo "Resuming as $new_pos (was $current_pos)..."
ssh ck "curl -s -X POST http://localhost:4000/api/command -H \"Content-Type: application/json\" -d '{\"command\": \"$new_cmd\"}'" | python3 -m json.tool | grep -E "(success|output)"

sleep 3
check_state "STATE AFTER POSITION CHANGE"

# TEST 7: Kill test (ungraceful shutdown)
echo -e "\n\nTEST 7: Testing ungraceful shutdown recovery"
echo "==========================================="

# Kill the process
echo "Killing server process..."
ssh ck 'pkill -9 -f "python.*tdr_server.*testing"'
sleep 3

# Restart
echo "Restarting after kill..."
ssh ck 'cd /home/chris/projects/bitstamp-testing && screen -dmS server-tst ./env/bin/python src/tdr_server.py'
sleep 10

check_state "STATE AFTER KILL/RESTART"

# SUMMARY
echo -e "\n\n📊 AUTO-RESUME TEST SUMMARY"
echo "==========================="
echo ""
echo "✅ Tests Completed:"
echo "   1. Initial state check"
echo "   2. Graceful shutdown"
echo "   3. State persistence"
echo "   4. Auto-resume on restart"
echo "   5. Manual resume command"
echo "   6. Resume with position change"
echo "   7. Ungraceful shutdown recovery"
echo ""
echo "🔍 Key Points to Verify:"
echo "   - Resume file is created/updated"
echo "   - Position is maintained across restarts"
echo "   - Auto-trading resumes if configured"
echo "   - Manual resume works correctly"
echo "   - System handles crashes gracefully"