#!/bin/bash
# Quick restart script with auto-pull and data caching

echo "🔄 Quick Restart Script Starting..."

# Store current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to check if server is running
check_server() {
    curl -s http://localhost:4000/api/ping > /dev/null 2>&1
    return $?
}

# Function to stop server gracefully
stop_server() {
    echo "📍 Stopping server..."
    if check_server; then
        # Send shutdown command
        curl -X POST http://localhost:4000/api/shutdown 2>&1 || true
        
        # Wait for graceful shutdown (max 10 seconds)
        for i in {1..10}; do
            if ! check_server; then
                echo "✅ Server stopped gracefully"
                break
            fi
            sleep 1
        done
    fi
    
    # Force kill if still running
    pkill -f "python.*tdr_server.py" 2>/dev/null || true
    sleep 1
}

# Function to pull latest changes
pull_changes() {
    echo "📥 Pulling latest changes..."
    git pull origin stable-added-adaptive-trad-n-chart-more
    
    if [ $? -eq 0 ]; then
        echo "✅ Code updated successfully"
    else
        echo "❌ Failed to pull changes"
        exit 1
    fi
}

# Function to check cache validity
check_cache() {
    CACHE_FILE="data_cache/btcusd_processed.pkl"
    SOURCE_FILE="btcusd.log"
    
    if [ -f "$CACHE_FILE" ]; then
        # Get modification times
        CACHE_TIME=$(stat -f %m "$CACHE_FILE" 2>/dev/null || stat -c %Y "$CACHE_FILE" 2>/dev/null)
        SOURCE_TIME=$(stat -f %m "$SOURCE_FILE" 2>/dev/null || stat -c %Y "$SOURCE_FILE" 2>/dev/null)
        
        if [ "$CACHE_TIME" -gt "$SOURCE_TIME" ]; then
            echo "✅ Cache is valid"
            return 0
        fi
    fi
    
    echo "⚠️  Cache invalid or missing"
    return 1
}

# Function to start server with caching
start_server() {
    echo "🚀 Starting server..."
    
    # Check if we should use cache
    if check_cache; then
        export USE_DATA_CACHE=1
        echo "   Using cached data for fast startup"
    else
        export USE_DATA_CACHE=0
        echo "   Will load data from scratch (this may take time)"
    fi
    
    # Start server in background
    nohup python src/tdr_server.py > logs/server_restart.log 2>&1 &
    
    # Wait for server to be ready
    echo -n "   Waiting for server to start"
    for i in {1..30}; do
        if check_server; then
            echo ""
            echo "✅ Server started successfully"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    
    echo ""
    echo "❌ Server failed to start"
    tail -20 logs/server_restart.log
    return 1
}

# Main execution
echo "═══════════════════════════════════════════════════"
echo "Starting at $(date)"
echo "═══════════════════════════════════════════════════"

# Step 1: Stop server
stop_server

# Step 2: Pull changes
pull_changes

# Step 3: Start server
start_server

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Quick restart completed successfully!"
    echo "   Server is ready for connections"
    
    # Show recent logs
    echo ""
    echo "📋 Recent server logs:"
    tail -10 logs/server_restart.log
else
    echo "❌ Restart failed - check logs/server_restart.log"
    exit 1
fi