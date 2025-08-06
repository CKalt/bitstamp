#!/bin/bash
# Check P&L with fees shown separately

echo "💰 COMPLETE P&L ANALYSIS"
echo "======================="
echo ""

# Get current status
status=$(ssh ck 'curl -s http://localhost:4000/api/status')

# Parse values
current_price=$(echo "$status" | python3 -c "import json,sys; print(json.load(sys.stdin)['last_price'])")
entry_price=$(echo "$status" | python3 -c "import json,sys; print(json.load(sys.stdin)['position']['entry_price'])")
position_size=$(echo "$status" | python3 -c "import json,sys; print(json.load(sys.stdin)['position']['position_size'])")
position=$(echo "$status" | python3 -c "import json,sys; print(json.load(sys.stdin)['position']['position'])")

# Get fees from strategy
fees_info=$(ssh ck 'grep "total_fees_paid" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | tail -1' 2>/dev/null || echo "")

# Extract fees paid
total_fees=$(ssh ck 'grep -o "Fees: \$[0-9.]*" /home/chris/projects/bitstamp-testing/logs/tdr_server.log | grep -o "[0-9.]*" | awk "{sum+=\$1} END {print sum}"')

echo "📊 POSITION DETAILS:"
echo "-------------------"
echo "  Direction: $([ "$position" == "1" ] && echo "LONG" || echo "SHORT")"
echo "  Entry Price: \$$entry_price"
echo "  Current Price: \$$current_price"
echo "  Position Size: $position_size BTC"
echo ""

# Calculate P&L
python3 << EOF
entry = $entry_price
current = $current_price
size = $position_size
fees = ${total_fees:-0}

# Raw P&L (price movement only)
raw_pnl = (current - entry) * size
raw_pnl_pct = ((current - entry) / entry) * 100

# Position values
entry_value = entry * size
current_value = current * size

print("💵 P&L BREAKDOWN:")
print("-----------------")
print(f"  Entry Value: \${entry_value:,.2f}")
print(f"  Current Value: \${current_value:,.2f}")
print(f"  Price Movement P&L: \${raw_pnl:,.2f} ({raw_pnl_pct:+.2f}%)")
print(f"  Fees Paid: \${fees:,.2f}")
print(f"  ─────────────────────────")
print(f"  Net P&L: \${raw_pnl - fees:,.2f}")
print()
print("📝 NOTE: Net P&L = Price Movement P&L - Fees Paid")
EOF

echo ""
echo "✅ This correctly shows:"
echo "   - Entry price remains clean (actual trade price)"
echo "   - Fees tracked separately"  
echo "   - Net P&L accounts for both price movement and fees"