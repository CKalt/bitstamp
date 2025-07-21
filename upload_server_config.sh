#!/bin/bash
# Script to upload correct best_strategy.json to server

echo "==================================="
echo "Server Configuration Upload Script"
echo "==================================="

# First, create the correct best_strategy.json with live trading enabled
cat > best_strategy_server.json << 'EOF'
{
    "Frequency": "1H",
    "Strategy": "MA",
    "Short_Window": 6,
    "Long_Window": 34,
    "Final_Balance": 10877.413931682839,
    "Total_Return": 8.774139316828386,
    "Total_Trades": 47.0,
    "Average_Trades_Per_Day": 1.5161290322580645,
    "Profit_Factor": 1.1236390721081366,
    "Sharpe_Ratio": 0.6001439204449514,
    "Bar_Size": "1H",
    "Last_Signal_Timestamp": 1753099200,
    "Last_Signal_Action": "GO LONG",
    "Last_Trade_Timestamp": 1753102567,
    "Last_Trade_Price": 118202.0,
    "do_live_trades": true,
    "start_window_days_back": 30,
    "end_window_days_back": 0,
    "auto_resume": true,
    "max_trades_per_day": 10,
    "min_trade_gap_minutes": 15,
    "signal_confirmation_bars": 2,
    "enable_pivot_protection": true,
    "pivot_buffer": 100,
    "pivot_lookback_hours": 2,
    "enable_trailing_pivots": true,
    "pivot_profit_tiers": [
        {"threshold": 0.05, "protection_ratio": 0.70},
        {"threshold": 0.10, "protection_ratio": 0.80},
        {"threshold": 0.15, "protection_ratio": 0.85},
        {"threshold": 0.20, "protection_ratio": 0.90}
    ],
    "pivot_respect_technical_levels": true,
    "regime_switch_threshold": 0.4,
    "emergency_loss_threshold": -5000,
    "emergency_override_enabled": true
}
EOF

echo "Created corrected best_strategy.json with:"
echo "  - do_live_trades: true (was false)"
echo "  - All necessary parameters for live trading"
echo ""

# Check if we can connect to server
SERVER_HOST=${1:-localhost}
SERVER_PORT=${2:-4000}

echo "Checking server connection at $SERVER_HOST:$SERVER_PORT..."
if curl -s -f "http://$SERVER_HOST:$SERVER_PORT/api/ping" > /dev/null 2>&1; then
    echo "✅ Server is reachable"
else
    echo "❌ Cannot reach server at http://$SERVER_HOST:$SERVER_PORT"
    echo "Make sure:"
    echo "  1. Server is running"
    echo "  2. SSH tunnel is active (if remote)"
    exit 1
fi

# For remote server, use SCP to upload file
if [ "$SERVER_HOST" != "localhost" ]; then
    echo ""
    echo "Uploading to remote server..."
    # You'll need to modify this with your actual server details
    scp best_strategy_server.json your-server:/home/chris/projects/bitstamp/best_strategy.json
    
    echo ""
    echo "To apply changes on server:"
    echo "  1. SSH to server"
    echo "  2. Restart TDR server"
    echo "  3. Server will auto-load new configuration"
else
    echo ""
    echo "For local server, copy file manually:"
    echo "  cp best_strategy_server.json /path/to/server/best_strategy.json"
fi

echo ""
echo "==================================="
echo "Next Steps:"
echo "1. Restart the server to load new configuration"
echo "2. Check status to verify configuration"
echo "3. Use 'signal_monitor' to track signals"
echo "===================================="