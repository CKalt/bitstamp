#!/bin/bash
# Setup script for development trading environment

echo "Setting up development trading environment..."
echo "=========================================="

# Check if we're on the server
if [[ ! -d "/home/chris/projects/bitstamp" ]]; then
    echo "ERROR: This script should be run on the server (ck)"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Configuration
LIVE_DIR="/home/chris/projects/bitstamp"
DEV_DIR="/home/chris/projects/bitstamp-dev"
DEV_PORT=4001

# Step 1: Create dev directory
if [[ -d "$DEV_DIR" ]]; then
    echo "⚠️  Development directory already exists: $DEV_DIR"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$DEV_DIR"
    else
        echo "Exiting..."
        exit 1
    fi
fi

echo "1. Cloning repository to $DEV_DIR..."
git clone "$LIVE_DIR" "$DEV_DIR"
cd "$DEV_DIR"

# Step 2: Create development branch
echo "2. Creating development branch..."
git checkout -b development

# Step 3: Create symlink for price feed
echo "3. Creating symlink for btcusd.log..."
ln -s ../bitstamp/btcusd.log btcusd.log
ls -la btcusd.log

# Step 4: Create necessary directories
echo "4. Creating directories..."
mkdir -p logs

# Step 5: Modify configuration
echo "5. Creating development configuration..."
if [[ -f "best_strategy.json" ]]; then
    # Backup original
    cp best_strategy.json best_strategy.json.original
    
    # Modify for dev (using Python for JSON manipulation)
    python3 << 'EOF'
import json

with open('best_strategy.json', 'r') as f:
    config = json.load(f)

# Add development settings
config['trading_mode'] = 'development'
config['max_position_btc'] = 0.001
config['log_prefix'] = 'DEV'
config['server_port'] = 4001

with open('best_strategy.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Configuration updated for development")
EOF
fi

# Step 6: Create port configuration
echo "6. Configuring development port..."
cat > port_config.py << 'EOF'
# Development server configuration
DEV_SERVER_PORT = 4001
EOF

# Step 7: Create monitoring script
echo "7. Creating monitoring script..."
cat > check-both-versions.sh << 'EOF'
#!/bin/bash
# Monitor both live and dev versions

echo "=== LIVE TRADING STATUS ==="
curl -s http://localhost:4000/api/command -H 'Content-Type: application/json' \
  -d '{"command": "status"}' 2>/dev/null | python3 -m json.tool | grep -E "position|entry_price|unrealized_pnl" || echo "Live server not responding"

echo -e "\n=== DEV TRADING STATUS ==="
curl -s http://localhost:4001/api/command -H 'Content-Type: application/json' \
  -d '{"command": "status"}' 2>/dev/null | python3 -m json.tool | grep -E "position|entry_price|unrealized_pnl" || echo "Dev server not responding"

echo -e "\n=== PRICE FEED STATUS ==="
if [[ -L "btcusd.log" ]]; then
    echo "✅ Symlink active: $(ls -la btcusd.log | awk '{print $11}')"
    echo "Latest price: $(tail -1 btcusd.log 2>/dev/null | cut -d, -f2 | cut -d: -f2)"
else
    echo "❌ Price feed symlink not found!"
fi

echo -e "\n=== RECENT ACTIVITY ==="
echo "LIVE signals:" $(tail -5 ../bitstamp/logs/tdr_server.log 2>/dev/null | grep -c SIGNAL_EVAL)
echo "DEV signals:"  $(tail -5 logs/tdr_server_dev.log 2>/dev/null | grep -c SIGNAL_EVAL)
EOF

chmod +x check-both-versions.sh

# Step 8: Create start script for dev
echo "8. Creating start script..."
cat > start-dev-trading.sh << 'EOF'
#!/bin/bash
# Start development trading server

echo "Starting development trading server on port 4001..."
echo "================================================"

# Check if price feed is available
if [[ ! -L "btcusd.log" ]]; then
    echo "ERROR: btcusd.log symlink not found!"
    echo "Make sure websock-ticker2.py is running in live directory"
    exit 1
fi

# Ensure we're using dev config
export TRADING_MODE=development
export SERVER_PORT=4001

# Start server
python src/tdr.py --server --port 4001
EOF

chmod +x start-dev-trading.sh

# Step 9: Modify logging to use separate file
echo "9. Patching logging configuration..."
cat > logging_patch.py << 'EOF'
import sys
import os

# Quick patch to use different log file for dev
if 'dev' in os.getcwd() or os.environ.get('TRADING_MODE') == 'development':
    LOG_FILE = 'logs/tdr_server_dev.log'
    TRADES_FILE = 'trades_dev.json'
else:
    LOG_FILE = 'logs/tdr_server.log'
    TRADES_FILE = 'trades.json'

print(f"Logging to: {LOG_FILE}")
print(f"Trades to: {TRADES_FILE}")
EOF

# Step 10: Create position limit safety
echo "10. Creating position limit safety check..."
cat > src/position_limiter.py << 'EOF'
"""
Position limiter for development trading
Ensures dev version only trades small amounts
"""

import logging

class PositionLimiter:
    def __init__(self, max_btc=0.001, mode='development'):
        self.max_btc = max_btc
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
    def validate_trade_size(self, amount, trade_type, current_position=0):
        """Limit trade sizes in development mode."""
        if self.mode != 'development':
            return amount
            
        if trade_type == 'buy':
            # Limit total BTC position
            max_buy = max(0, self.max_btc - current_position)
            if amount > max_buy:
                self.logger.warning(f"DEV: Limiting buy from {amount} to {max_buy} BTC (max position: {self.max_btc})")
                return max_buy
                
        return amount
        
    def check_position_limit(self, position_btc):
        """Check if position exceeds limit."""
        if self.mode == 'development' and abs(position_btc) > self.max_btc * 1.1:  # 10% tolerance
            self.logger.error(f"DEV: Position {position_btc} exceeds limit {self.max_btc}!")
            return False
        return True
EOF

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. In live directory, ensure websock-ticker2.py is running:"
echo "   cd $LIVE_DIR && screen -S ticker python src/websock-ticker2.py"
echo ""
echo "2. Start development server:"
echo "   cd $DEV_DIR && ./start-dev-trading.sh"
echo ""
echo "3. Start trading with small position:"
echo "   curl -X POST http://localhost:4001/api/command -H 'Content-Type: application/json' \\"
echo "     -d '{\"command\": \"auto_trade 0.001btc MA short=6 long=34 do_live_trades=True hist_position=long\"}'"
echo ""
echo "4. Monitor both versions:"
echo "   cd $DEV_DIR && ./check-both-versions.sh"