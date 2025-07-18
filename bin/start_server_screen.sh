#!/bin/bash
# Start server in screen session with proper environment

echo "🖥️  Starting TDR Server in screen session..."

# Store current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go to project root (parent of bin)
cd "$SCRIPT_DIR/.."

# Kill any existing server
echo "📍 Stopping any existing server..."
pkill -f "python.*tdr_server.py" 2>/dev/null || true
sleep 2

# Kill any existing screen session
screen -S tdr_server -X quit 2>/dev/null || true

# Create startup script for screen
cat > /tmp/tdr_server_start.sh << 'EOF'
#!/bin/bash
cd /home/chris/projects/bitstamp
source env/bin/activate
export PYTHONPATH=/home/chris/projects/bitstamp/src:$PYTHONPATH
echo "Starting TDR Server with monitoring..."
echo "PYTHONPATH=$PYTHONPATH"
python src/tdr_server.py
EOF

chmod +x /tmp/tdr_server_start.sh

# Start in screen
screen -dmS tdr_server /tmp/tdr_server_start.sh

echo "✅ Server starting in screen session 'tdr_server'"
echo ""
echo "📋 Commands:"
echo "   screen -r tdr_server    # Attach to server"
echo "   Ctrl-A D                # Detach from screen"
echo "   screen -ls              # List screen sessions"
echo ""
echo "Waiting for server to be ready..."

# Wait for server
for i in {1..30}; do
    if curl -s http://localhost:4000/api/ping > /dev/null 2>&1; then
        echo "✅ Server is ready!"
        echo ""
        echo "📺 Attach with: screen -r tdr_server"
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "⚠️  Server may still be starting. Check with: screen -r tdr_server"