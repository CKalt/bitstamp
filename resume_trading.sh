#!/bin/bash
# Script to resume auto trading with current LONG position

echo "========================================"
echo "Resume Auto Trading Setup"
echo "========================================"
echo ""
echo "Current Position:"
echo "  - Position: LONG"
echo "  - BTC Amount: 1.36 BTC"
echo "  - Entry Price: $117,454"
echo "  - Position Value: $159,737.44"
echo ""

# Calculate position details
BTC_AMOUNT=1.36
ENTRY_PRICE=117454
POSITION_VALUE=$(echo "$BTC_AMOUNT * $ENTRY_PRICE" | bc)

echo "Resume Command Options:"
echo ""
echo "1. Direct resume command (in TDR client):"
echo "   resume_auto_trade 1.36btc long 117454"
echo ""
echo "2. Via JSON command file:"
cat > resume_command.json << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%S)Z",
  "command": "resume_auto_trade 1.36btc long 117454",
  "source": "manual_resume",
  "args": ""
}
EOF
echo "   Created resume_command.json"
echo ""

# Create resume-auto-trade.json for server
cat > resume-auto-trade-server.json << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%S).000000",
  "position": "LONG",
  "amount": $BTC_AMOUNT,
  "unit": "btc",
  "entry_price": $ENTRY_PRICE,
  "current_price": $ENTRY_PRICE,
  "unrealized_pnl": 0.0,
  "command": "resume_auto_trade ${BTC_AMOUNT}btc long $ENTRY_PRICE",
  "strategy": {
    "type": "AdaptiveMultiStrategy",
    "short_window": 6,
    "long_window": 34,
    "current_regime": "unknown",
    "active_strategy": "trending"
  },
  "balances": {
    "btc": $BTC_AMOUNT,
    "usd": 0.0
  },
  "trades_executed": 0,
  "last_trade_time": null,
  "trade_references": [],
  "pivot_protection": {
    "enabled": true,
    "tracker": {}
  }
}
EOF

echo "3. Server-side resume file created: resume-auto-trade-server.json"
echo ""
echo "========================================"
echo "Steps to Resume:"
echo "========================================"
echo ""
echo "1. FIRST - Update server configuration:"
echo "   scp best_strategy.json server:/home/chris/projects/bitstamp/"
echo "   scp resume-auto-trade-server.json server:/home/chris/projects/bitstamp/resume-auto-trade.json"
echo ""
echo "2. THEN - Restart server (it will auto-resume):"
echo "   ssh server"
echo "   cd /home/chris/projects/bitstamp"
echo "   git pull origin stable-added-adaptive-trad-n-chart-more"
echo "   python src/tdr_server.py"
echo ""
echo "3. OR - Resume manually from client:"
echo "   python src/tdr_client.py"
echo "   tdr> enable_commands"
echo "   tdr> resume_auto_trade 1.36btc long 117454"
echo ""
echo "4. VERIFY with:"
echo "   tdr> status long"
echo "   tdr> signal_monitor"
echo ""
echo "========================================"