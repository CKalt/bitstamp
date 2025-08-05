#!/bin/bash
# Comprehensive auto-trading verification script

echo "🔍 AUTO-TRADING VERIFICATION CHECKLIST"
echo "======================================"
echo "Time: $(date)"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check functions
check_pass() {
    echo -e "${GREEN}✅ $1${NC}"
}

check_fail() {
    echo -e "${RED}❌ $1${NC}"
    FAILURES=$((FAILURES + 1))
}

check_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

FAILURES=0

# 1. Server Status
echo "1. SERVER STATUS"
echo "----------------"
STATUS=$(curl -s http://localhost:4000/api/status 2>/dev/null)
if [ -n "$STATUS" ]; then
    check_pass "Server is running"
    
    # Extract key values
    AUTO_ACTIVE=$(echo "$STATUS" | grep -o '"active":[^,}]*' | head -1 | grep -o 'true\|false')
    LIVE_TRADING=$(echo "$STATUS" | grep -o '"live_trading":[^,}]*' | grep -o 'true\|false')
    POSITION=$(echo "$STATUS" | grep -o '"position":[^,}]*' | head -1 | grep -o '[-0-9]*')
    
    if [ "$AUTO_ACTIVE" = "true" ]; then
        check_pass "Auto-trader is active"
    else
        check_fail "Auto-trader is NOT active"
    fi
    
    if [ "$LIVE_TRADING" = "true" ]; then
        check_pass "Live trading is enabled"
    else
        check_fail "Live trading is DISABLED (dry-run mode)"
    fi
else
    check_fail "Cannot connect to server"
    echo "Server may be down!"
    exit 1
fi

# 2. Configuration Check
echo ""
echo "2. CONFIGURATION"
echo "----------------"
if [ -f best_strategy.json ]; then
    DO_LIVE=$(grep do_live_trades best_strategy.json | grep -o 'true\|false')
    AUTO_RESUME=$(grep auto_resume best_strategy.json | grep -o 'true\|false')
    THRESHOLD=$(grep ma_separation_threshold best_strategy.json | grep -o '[0-9.]*')
    
    if [ "$DO_LIVE" = "true" ]; then
        check_pass "do_live_trades: true"
    else
        check_fail "do_live_trades: false (trades won't execute!)"
    fi
    
    if [ "$AUTO_RESUME" = "true" ]; then
        check_pass "auto_resume: true"
    else
        check_warn "auto_resume: false (position won't reload on restart)"
    fi
    
    check_pass "MA threshold: ${THRESHOLD}%"
else
    check_fail "best_strategy.json not found"
fi

# 3. Position Tracking
echo ""
echo "3. POSITION TRACKING"
echo "-------------------"
if [ -f resume-auto-trade.json ]; then
    check_pass "Resume file exists"
    RESUME_POS=$(grep position resume-auto-trade.json | head -1 | grep -o 'SHORT\|LONG')
    echo "  Position in resume: $RESUME_POS"
else
    check_warn "No resume file (OK if never traded)"
fi

if [ "$POSITION" = "-1" ]; then
    check_pass "Current position: SHORT"
elif [ "$POSITION" = "1" ]; then
    check_pass "Current position: LONG"
else
    check_warn "Current position: NEUTRAL"
fi

# 4. Recent Activity
echo ""
echo "4. RECENT ACTIVITY"
echo "-----------------"
RECENT_EVALS=$(tail -100 logs/tdr_server.log | grep -c "SIGNAL_EVAL")
if [ $RECENT_EVALS -gt 2 ]; then
    check_pass "Signal evaluations running (found $RECENT_EVALS in recent logs)"
else
    check_fail "No recent signal evaluations!"
fi

# Check evaluation frequency
LAST_EVAL_TIME=$(tail -100 logs/tdr_server.log | grep "SIGNAL_EVAL" | tail -1 | cut -d' ' -f2)
if [ -n "$LAST_EVAL_TIME" ]; then
    echo "  Last evaluation: $LAST_EVAL_TIME"
fi

# 5. Trade Execution Capability
echo ""
echo "5. TRADE EXECUTION"
echo "-----------------"
# Check for recent errors
RECENT_ERRORS=$(tail -200 logs/tdr_server.log | grep -c "ERROR")
if [ $RECENT_ERRORS -eq 0 ]; then
    check_pass "No recent errors"
else
    check_warn "Found $RECENT_ERRORS errors in recent logs"
fi

# Check trade limits
TRADES_TODAY=$(echo "$STATUS" | grep -o '"trades_today":[^,}]*' | grep -o '[0-9]*')
echo "  Trades today: ${TRADES_TODAY:-0}/5"

# 6. Current MA Status
echo ""
echo "6. MA CROSSOVER STATUS"
echo "---------------------"
CMD_STATUS=$(curl -s -X POST http://localhost:4000/api/command \
    -H "Content-Type: application/json" \
    -d '{"command": "status"}' 2>/dev/null)

PROXIMITY=$(echo "$CMD_STATUS" | grep -o "MA Crossover Proximity: [0-9.]*%" | grep -o "[0-9.]*")
if [ -n "$PROXIMITY" ]; then
    echo "  Current proximity: ${PROXIMITY}%"
    echo "  Trigger threshold: ${THRESHOLD}%"
    
    if command -v bc >/dev/null 2>&1 && [ -n "$THRESHOLD" ]; then
        if (( $(echo "$PROXIMITY <= $THRESHOLD" | bc -l) )); then
            check_warn "IN TRIGGER ZONE - Trade may execute soon!"
        else
            DISTANCE=$(echo "scale=2; $PROXIMITY - $THRESHOLD" | bc)
            echo "  Distance to trigger: ${DISTANCE}%"
        fi
    fi
fi

# 7. Critical Services
echo ""
echo "7. CRITICAL SERVICES"
echo "-------------------"
# Check if systemd service is active
SYSTEMD_STATUS=$(systemctl is-active tdr-server 2>/dev/null)
if [ "$SYSTEMD_STATUS" = "active" ]; then
    check_pass "TDR service is active (will auto-restart on failure)"
else
    check_warn "TDR not running as systemd service"
fi

# Check WebSocket connection
WS_STATUS=$(echo "$STATUS" | grep -o '"websocket":"[^"]*"' | cut -d'"' -f4)
if [ "$WS_STATUS" = "connected" ]; then
    check_pass "WebSocket connected (receiving price updates)"
else
    check_fail "WebSocket not connected!"
fi

# Summary
echo ""
echo "======================================"
if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED${NC}"
    echo "Auto-trading should continue correctly while you're away."
    echo ""
    echo "RECOMMENDATIONS:"
    echo "- Leave screen sessions running for monitoring"
    echo "- Check back periodically via: screen -r tdr-proximity"
    echo "- System will evaluate every 30 seconds and trade when proximity ≤ ${THRESHOLD}%"
else
    echo -e "${RED}❌ FOUND $FAILURES CRITICAL ISSUES${NC}"
    echo "Please fix these before leaving the system unattended!"
fi

echo ""
echo "USEFUL COMMANDS WHILE AWAY:"
echo "- Check status: curl -s http://localhost:4000/api/status | jq"
echo "- View screens: screen -ls"
echo "- Check logs: tail -f logs/tdr_server.log | grep SIGNAL_EVAL"