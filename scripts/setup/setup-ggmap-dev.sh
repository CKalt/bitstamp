#!/bin/bash
# Setup script for development environment with ggmap integration

echo "Setting up development trading environment with ggmap..."
echo "======================================================="

# Configuration based on your ggmap
LIVE_DIR="/home/chris/projects/bitstamp"              # gg btc
DEV_DIR="/home/chris/projects/bitstamp-testing"       # gg tst
DEV_PORT=4001

# Step 1: Check if dev directory exists
if [[ -d "$DEV_DIR" ]]; then
    echo "✅ Development directory exists: $DEV_DIR (gg tst)"
    cd "$DEV_DIR"
    
    # Check if it's a git repo
    if [[ ! -d ".git" ]]; then
        echo "⚠️  Not a git repository. Initializing..."
        git init
        git remote add origin $(cd "$LIVE_DIR" && git remote get-url origin)
    fi
else
    echo "Creating development directory..."
    git clone "$LIVE_DIR" "$DEV_DIR"
    cd "$DEV_DIR"
    
    # Update ggmap if not already there
    if ! grep -q "^#- tst " ~/ggmap; then
        echo "Adding 'gg tst' shortcut to ggmap..."
        ggr tst
    fi
fi

# Step 2: Create development branch
echo "Setting up development branch..."
git checkout -b development 2>/dev/null || git checkout development

# Step 3: Create symlink for price feed
echo "Creating symlink for btcusd.log..."
if [[ -L "btcusd.log" ]]; then
    echo "✅ Symlink already exists"
else
    ln -s ../bitstamp/btcusd.log btcusd.log
    echo "✅ Created symlink to live price feed"
fi

# Step 4: Create necessary directories
mkdir -p logs

# Step 5: Create development configuration
echo "Creating development configuration..."
cat > best_strategy.json << 'EOF'
{
  "Frequency": "1H",
  "Strategy": "MA",
  "Short_Window": 6,
  "Long_Window": 34,
  "Bar_Size": "1H",
  "do_live_trades": true,
  "strategy_type": "MA",
  "enable_adaptive_strategy": false,
  "auto_resume": false,
  "trading_mode": "development",
  "max_position_btc": 0.001,
  "max_position_usd": 100,
  "log_prefix": "DEV",
  "server_port": 4001
}
EOF

# Step 6: Create monitoring script that uses ggmap
cat > monitor-both.sh << 'EOF'
#!/bin/bash
# Monitor both live and dev versions using ggmap shortcuts

echo "=== LIVE TRADING STATUS (gg btc) ==="
curl -s http://localhost:4000/api/command -H 'Content-Type: application/json' \
  -d '{"command": "status"}' 2>/dev/null | grep -E "Position:|Entry Price:|Unrealized PnL:" || echo "Live server not responding"

echo -e "\n=== DEV TRADING STATUS (gg tst) ==="
curl -s http://localhost:4001/api/command -H 'Content-Type: application/json' \
  -d '{"command": "status"}' 2>/dev/null | grep -E "Position:|Entry Price:|Unrealized PnL:" || echo "Dev server not responding"

echo -e "\n=== PRICE FEED STATUS ==="
LIVE_PRICE=$(tail -1 /home/chris/projects/bitstamp/btcusd.log 2>/dev/null | cut -d',' -f2 | cut -d':' -f2)
DEV_PRICE=$(tail -1 /home/chris/projects/bitstamp-testing/btcusd.log 2>/dev/null | cut -d',' -f2 | cut -d':' -f2)
echo "Live feed: $LIVE_PRICE"
echo "Dev feed:  $DEV_PRICE (should match)"

echo -e "\n=== LOG ACTIVITY (last minute) ==="
echo "LIVE:" $(find /home/chris/projects/bitstamp/logs/tdr_server.log -mmin -1 2>/dev/null && echo "✅ Active" || echo "❌ No recent activity")
echo "DEV:"  $(find /home/chris/projects/bitstamp-testing/logs/tdr_server_dev.log -mmin -1 2>/dev/null && echo "✅ Active" || echo "❌ No recent activity")
EOF

chmod +x monitor-both.sh

# Step 7: Create start script
cat > start-dev.sh << 'EOF'
#!/bin/bash
# Start development trading server

echo "Starting DEV trading server on port 4001..."
echo "=========================================="

# Check price feed
if [[ ! -L "btcusd.log" ]]; then
    echo "❌ ERROR: btcusd.log symlink missing!"
    echo "Creating symlink..."
    ln -s ../bitstamp/btcusd.log btcusd.log
fi

# Check if price feed is active
if [[ ! -f "../bitstamp/btcusd.log" ]]; then
    echo "⚠️  WARNING: Price feed file doesn't exist!"
    echo "Make sure to run 'python src/websock-ticker2.py' in live directory (gg btc)"
fi

# Set environment to ensure dev mode
export TRADING_MODE=development
export SERVER_PORT=4001
export LOG_FILE=logs/tdr_server_dev.log

# Start server
echo "Starting server..."
python src/tdr.py --server --port 4001
EOF

chmod +x start-dev.sh

# Step 8: Create quick switch script
cat > gg-switch.sh << 'EOF'
#!/bin/bash
# Quick switch between live and dev with status

case "$1" in
    btc|live)
        echo "Switching to LIVE (gg btc)..."
        cd /home/chris/projects/bitstamp
        echo "PWD: $(pwd)"
        tail -3 logs/tdr_server.log 2>/dev/null | grep SIGNAL_EVAL
        ;;
    tst|dev)
        echo "Switching to DEV (gg tst)..."
        cd /home/chris/projects/bitstamp-testing
        echo "PWD: $(pwd)"
        tail -3 logs/tdr_server_dev.log 2>/dev/null | grep SIGNAL_EVAL
        ;;
    *)
        echo "Usage: ./gg-switch.sh [btc|tst]"
        echo "  btc/live - Switch to live trading"
        echo "  tst/dev  - Switch to dev testing"
        ;;
esac
EOF

chmod +x gg-switch.sh

# Step 9: Create position safety limiter
mkdir -p src/tdr_core
cat > src/tdr_core/position_safety.py << 'EOF'
"""
Position safety limits for development mode
"""
import logging
import os

class PositionSafety:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.is_dev = (
            config.get('trading_mode') == 'development' or
            'testing' in os.getcwd()
        )
        
    def validate_trade_size(self, amount, trade_type, current_position=0):
        """Ensure dev mode only trades tiny amounts."""
        if not self.is_dev:
            return amount
            
        max_btc = self.config.get('max_position_btc', 0.001)
        max_usd = self.config.get('max_position_usd', 100)
        
        if trade_type == 'buy':
            if amount > max_btc:
                self.logger.warning(f"DEV LIMIT: Reducing buy from {amount} to {max_btc} BTC")
                return max_btc
        else:  # sell
            # Check USD value
            current_price = self.get_current_price()
            usd_value = amount * current_price
            if usd_value > max_usd:
                safe_amount = max_usd / current_price
                self.logger.warning(f"DEV LIMIT: Reducing sell from {amount} to {safe_amount} BTC")
                return safe_amount
                
        return amount
EOF

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "Your ggmap shortcuts:"
echo "  gg btc → Live trading ($LIVE_DIR)"
echo "  gg tst → Dev testing ($DEV_DIR)"
echo ""
echo "Next steps:"
echo "1. Start price feed (if not running):"
echo "   gg btc && screen -S ticker python src/websock-ticker2.py"
echo ""
echo "2. Start dev server:"
echo "   gg tst && screen -S dev-trading ./start-dev.sh"
echo ""
echo "3. Start dev trading (0.001 BTC):"
echo "   curl -X POST http://localhost:4001/api/command -H 'Content-Type: application/json' \\"
echo "     -d '{\"command\": \"auto_trade 0.001btc MA short=6 long=34 do_live_trades=True hist_position=long\"}'"
echo ""
echo "4. Monitor both versions:"
echo "   gg tst && ./monitor-both.sh"