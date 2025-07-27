#!/bin/bash
# Setup local test environment on Mac

echo "Setting up local test environment (gg tst)..."
echo "==========================================="

# Source ggmap for the functions
source ~/ggmap

# Go to parent directory and clone
cd /Users/chris/projects/python
git clone btc btc-testing

# Go into the new directory
cd btc-testing

# Register with ggmap
echo "Registering 'gg tst' shortcut..."
ggr tst

# Switch to development branch
git checkout -b development

# Create test configuration
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

echo ""
echo "✅ Local test environment created!"
echo ""
echo "Your Mac now has:"
echo "  gg btc → /Users/chris/projects/python/btc (live)"
echo "  gg tst → /Users/chris/projects/python/btc-testing (test)"
echo ""
echo "To start test client:"
echo "  gg tst"
echo "  screen -S client-tst python src/tdr.py"