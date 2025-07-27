#!/bin/bash
# Setup local test environment with ggmap integration

echo "Setting up local test environment for TDR..."
echo "==========================================="

# Source ggmap to get the gg functions
source ~/ggmap

# Step 1: Clone to create test directory
echo "1. Creating test directory..."
cd /Users/chris/projects/python
git clone btc btc-testing
cd btc-testing

# Step 2: Register with ggmap
echo "2. Registering 'gg tst' shortcut..."
ggr tst

# Step 3: Switch to development branch
echo "3. Setting up development branch..."
git checkout -b development

# Step 4: Create test configuration
echo "4. Creating test configuration..."
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
  "log_prefix": "DEV_LOCAL"
}
EOF

# Step 5: Create local screen management scripts
echo "5. Creating screen management scripts..."

# Script to start client in screen
cat > start-client-tst.sh << 'EOF'
#!/bin/bash
# Start test client in screen session

source ~/ggmap

echo "Starting test client in screen session: client-tst"
echo "================================================="

# Check if screen session already exists
if screen -list | grep -q "client-tst"; then
    echo "⚠️  Screen session 'client-tst' already exists!"
    echo "To attach: screen -r client-tst"
    echo "To kill it: screen -S client-tst -X quit"
    exit 1
fi

# Start in screen session
echo "Starting client in screen session 'client-tst'..."
gg tst
screen -dmS client-tst bash -c '
source ~/ggmap
gg tst
echo "Test client starting..."
python src/tdr.py
'

echo "✅ Client started in screen session 'client-tst'"
echo "To attach: screen -r client-tst"
echo "To detach: Ctrl+A, D"
EOF

chmod +x start-client-tst.sh

# Script to monitor all local screens
cat > monitor-local-screens.sh << 'EOF'
#!/bin/bash
# Monitor local Claude and client screens

source ~/ggmap

echo "=== LOCAL SCREEN SESSIONS ==="
echo "============================"

echo -e "\nCLAUDE SESSION:"
if screen -list | grep -q "claude-tdr"; then
    echo "✅ claude-tdr - Running"
else
    echo "❌ claude-tdr - Not running"
fi

echo -e "\nCLIENT SESSIONS:"
if screen -list | grep -q "client-tdr"; then
    echo "✅ client-tdr (live) - Running"
else
    echo "❌ client-tdr (live) - Not running"
fi

if screen -list | grep -q "client-tst"; then
    echo "✅ client-tst (test) - Running"
else
    echo "❌ client-tst (test) - Not running"
fi

echo -e "\nDIRECTORIES:"
echo "Live (gg btc): $(ggdir btc)"
echo "Test (gg tst): $(ggdir tst)"

echo -e "\nUSEFUL COMMANDS:"
echo "Go to live dir:        gg btc"
echo "Go to test dir:        gg tst"
echo "Attach to claude:      screen -r claude-tdr"
echo "Attach to live client: screen -r client-tdr"
echo "Attach to test client: screen -r client-tst"
echo "List all screens:      screen -list"
EOF

chmod +x monitor-local-screens.sh

echo ""
echo "✅ Local test environment setup complete!"
echo ""
echo "Your ggmap shortcuts:"
echo "  gg btc → Live directory"
echo "  gg tst → Test directory"
echo ""
echo "Next steps:"
echo "1. Run this script: bash setup-local-test-env.sh"
echo "2. Start test client: gg tst && ./start-client-tst.sh"
echo "3. Monitor screens: ./monitor-local-screens.sh"