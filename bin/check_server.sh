#!/bin/bash
# Quick server check script with timeouts and better diagnostics

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "🔍 TDR Server Status Check"
echo "========================="

# 1. Check for process
echo -n "📋 Process check: "
PID=$(pgrep -f "python.*tdr_server.py" | head -1)
if [ -z "$PID" ]; then
    echo -e "${RED}No server process found${NC}"
    echo ""
    echo "💡 Start the server with: bin/server_control.sh start"
    exit 1
else
    echo -e "${GREEN}Found${NC} (PID: $PID)"
fi

# 2. Check process details
echo -n "📊 Process info: "
PS_INFO=$(ps -p $PID -o %cpu,%mem,etime 2>/dev/null | tail -1)
if [ ! -z "$PS_INFO" ]; then
    CPU=$(echo $PS_INFO | awk '{print $1}')
    MEM=$(echo $PS_INFO | awk '{print $2}')
    TIME=$(echo $PS_INFO | awk '{print $3}')
    echo "CPU: ${CPU}%, MEM: ${MEM}%, Uptime: $TIME"
else
    echo -e "${YELLOW}Unable to get process info${NC}"
fi

# 3. Check port
echo -n "🔌 Port 4000: "
if lsof -i :4000 >/dev/null 2>&1; then
    echo -e "${GREEN}Listening${NC}"
else
    echo -e "${RED}Not listening${NC}"
    echo ""
    echo "⚠️  Server process exists but not listening on port 4000"
    echo "💡 Check logs: tail -50 logs/server.log"
    exit 1
fi

# 4. Quick HTTP test
echo -n "🌐 HTTP ping test: "
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 --max-time 3 http://localhost:4000/api/ping 2>/dev/null)
if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}OK${NC} (200)"
elif [ "$HTTP_CODE" = "000" ]; then
    echo -e "${RED}No response${NC}"
    echo ""
    echo "⚠️  Server is listening but not responding to HTTP requests"
    echo "💡 The server may be loading data or stuck. Check logs:"
    echo "   tail -f logs/server.log"
    exit 1
else
    echo -e "${YELLOW}Unexpected response${NC} ($HTTP_CODE)"
fi

# 5. Try to get status
echo -n "📡 API status: "
STATUS=$(curl -s --connect-timeout 2 --max-time 3 http://localhost:4000/api/status 2>/dev/null)
if [ $? -eq 0 ] && [ ! -z "$STATUS" ]; then
    echo -e "${GREEN}Available${NC}"
    
    # Parse key fields
    AUTO_TRADER=$(echo "$STATUS" | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('auto_trader', {}).get('active', 'N/A'))" 2>/dev/null || echo "parse error")
    HISTORY=$(echo "$STATUS" | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('history_loaded', 'N/A'))" 2>/dev/null || echo "parse error")
    WS=$(echo "$STATUS" | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('websocket', 'N/A'))" 2>/dev/null || echo "parse error")
    
    echo ""
    echo "📈 Server State:"
    echo "   Auto Trading: $AUTO_TRADER"
    echo "   History Loaded: $HISTORY"
    echo "   WebSocket: $WS"
else
    echo -e "${RED}Not available${NC}"
fi

echo ""
echo "✅ Check complete"