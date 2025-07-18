#!/bin/bash
# Server control script for TDR

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go to project root (parent of bin)
cd "$SCRIPT_DIR/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if server is running
check_server() {
    curl -s --connect-timeout 2 --max-time 5 http://localhost:4000/api/ping > /dev/null 2>&1
    return $?
}

# Function to get server PID
get_server_pid() {
    pgrep -f "python.*tdr_server.py" | head -1
}

case "$1" in
    status)
        echo "🔍 Checking server status..."
        
        # First check if process exists
        PID=$(get_server_pid)
        if [ -z "$PID" ]; then
            echo -e "${RED}❌ No server process found${NC}"
            exit 1
        fi
        
        echo "📍 Found server process (PID: $PID)"
        
        # Check if it's responding to HTTP
        echo -n "🌐 Checking HTTP response..."
        if check_server; then
            echo -e " ${GREEN}✓${NC}"
            
            # Get detailed status
            STATUS=$(curl -s --connect-timeout 2 --max-time 5 http://localhost:4000/api/status 2>/dev/null)
            if [ $? -eq 0 ] && [ ! -z "$STATUS" ]; then
                echo ""
                echo "📊 Server Details:"
                echo "$STATUS" | python -m json.tool 2>/dev/null | grep -E '"auto_trader"|"history_loaded"|"websocket"' | sed 's/^/  /' || echo "  Unable to parse status"
            else
                echo -e "\n${YELLOW}⚠️  Server is running but not responding to status requests${NC}"
            fi
        else
            echo -e " ${RED}✗${NC}"
            echo -e "${YELLOW}⚠️  Server process exists but is not responding${NC}"
            echo "   The server may be starting up or stuck."
            echo "   Check logs with: $0 logs"
            exit 1
        fi
        ;;
        
    start)
        echo "🚀 Starting server..."
        if check_server; then
            echo -e "${YELLOW}⚠️  Server is already running${NC}"
            exit 0
        fi
        
        # Source virtual environment first
        source env/bin/activate
        
        # Start server
        nohup python src/tdr_server.py > logs/server.log 2>&1 &
        
        # Wait for startup
        echo -n "   Waiting for server to start"
        for i in {1..30}; do
            if check_server; then
                echo ""
                PID=$(get_server_pid)
                echo -e "${GREEN}✅ Server started successfully${NC} (PID: $PID)"
                exit 0
            fi
            echo -n "."
            sleep 1
        done
        
        echo ""
        echo -e "${RED}❌ Server failed to start${NC}"
        echo "Check logs/server.log for details"
        exit 1
        ;;
        
    stop)
        echo "🛑 Stopping server..."
        if ! check_server; then
            echo -e "${YELLOW}⚠️  Server is not running${NC}"
            exit 0
        fi
        
        # Try graceful shutdown first
        curl -s --connect-timeout 2 --max-time 5 -X POST http://localhost:4000/api/shutdown 2>/dev/null
        
        # Wait for graceful shutdown
        echo -n "   Waiting for graceful shutdown"
        for i in {1..10}; do
            if ! check_server; then
                echo ""
                echo -e "${GREEN}✅ Server stopped gracefully${NC}"
                exit 0
            fi
            echo -n "."
            sleep 1
        done
        
        # Force kill if needed
        echo ""
        echo "   Force stopping server..."
        PID=$(get_server_pid)
        if [ ! -z "$PID" ]; then
            kill -9 $PID
            echo -e "${GREEN}✅ Server stopped (forced)${NC}"
        fi
        ;;
        
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
        
    logs)
        echo "📋 Showing server logs (Ctrl+C to exit)..."
        tail -f logs/server.log
        ;;
        
    attach)
        echo "📎 Attaching to server output..."
        echo "   (This shows live output, Ctrl+C to detach)"
        PID=$(get_server_pid)
        if [ -z "$PID" ]; then
            echo -e "${RED}❌ Server is not running${NC}"
            exit 1
        fi
        tail -f logs/server.log
        ;;
        
    quick-restart)
        echo "🔄 Quick restart with git pull..."
        bin/quick_restart.sh
        ;;
        
    *)
        echo "TDR Server Control"
        echo "=================="
        echo ""
        echo "Usage: $0 {status|start|stop|restart|logs|attach|quick-restart}"
        echo ""
        echo "Commands:"
        echo "  status        - Check if server is running"
        echo "  start         - Start the server"
        echo "  stop          - Stop the server"
        echo "  restart       - Restart the server"
        echo "  logs          - Show server logs (tail -f)"
        echo "  attach        - Attach to server output"
        echo "  quick-restart - Stop, pull changes, and restart"
        echo ""
        echo "Examples:"
        echo "  $0 status     # Check server status"
        echo "  $0 logs       # Watch live logs"
        echo "  $0 restart    # Restart server"
        ;;
esac