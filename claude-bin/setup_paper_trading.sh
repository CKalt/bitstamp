#!/bin/bash
# Setup paper trading mode for testing proximity threshold

echo "🧪 SETTING UP PAPER TRADING MODE"
echo "================================"
echo ""

# Create test configuration
cat > /tmp/test_best_strategy.json << 'EOF'
{
  "Short_Window": 4,
  "Long_Window": 20,
  "do_live_trades": false,
  "strategy_type": "MA",
  "enable_adaptive_strategy": false,
  "max_trades_per_day": 10,
  "proximity_threshold": 0.5,
  "_comment": "PAPER TRADING MODE - Testing 0.5% proximity threshold"
}
EOF

echo "📋 Test Configuration:"
echo "- MA 4/20 (same as live)"
echo "- do_live_trades: FALSE (paper trading)"
echo "- proximity_threshold: 0.5%"
echo "- All trades will be simulated"
echo ""

echo "To deploy on server:"
echo "1. scp /tmp/test_best_strategy.json ck:/home/chris/projects/bitstamp/best_strategy.json"
echo "2. Start server normally - it will trade on paper only"
echo ""

echo "📊 What to monitor:"
echo "- Logs will show 'NO_TRADE_PROXIMITY' when MAs < 0.5% apart"
echo "- 'WOULD EXECUTE' messages show theoretical trades"
echo "- No actual orders placed on Bitstamp"