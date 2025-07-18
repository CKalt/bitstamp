#!/bin/bash
# Run backtest with the correct script

echo "Running backtest with correct parameters..."
echo "Testing last 120 days with various MA combinations"
echo "============================================================"

# Array of MA combinations to test
declare -a configs=(
    "5,15,very_fast"
    "8,21,fibonacci"
    "10,46,current"
    "12,26,macd"
    "15,30,medium"
    "20,50,classic"
    "50,200,golden_cross"
)

best_return=-999
best_config=""

# Test each configuration
for config in "${configs[@]}"; do
    IFS=',' read -r short long name <<< "$config"
    
    echo ""
    echo "Testing $name: MA($short, $long)"
    echo "----------------------------------------"
    
    # Run the backtest
    output=$(python src/backtest.py \
        --start-window-days-back 120 \
        --end-window-days-back 0 \
        --high-frequency 1H \
        --low-frequency 15T \
        --short-window $short \
        --long-window $long 2>&1)
    
    # Extract key metrics
    return_pct=$(echo "$output" | grep -i "total return" | grep -oE '[+-]?[0-9]+\.?[0-9]*' | head -1)
    num_trades=$(echo "$output" | grep -i "total trades" | grep -oE '[0-9]+' | head -1)
    win_rate=$(echo "$output" | grep -i "win rate" | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    
    echo "  Return: ${return_pct:-0}%"
    echo "  Trades: ${num_trades:-0}"
    echo "  Win Rate: ${win_rate:-0}%"
    
    # Check if this is the best so far
    if [[ $(echo "${return_pct:-0} > $best_return" | bc -l) -eq 1 ]]; then
        best_return=$return_pct
        best_config="MA($short, $long) - $name"
    fi
done

echo ""
echo "============================================================"
echo "BEST CONFIGURATION: $best_config"
echo "Return: $best_return%"
echo "============================================================"

# Now run the best one with full output
if [ ! -z "$best_config" ]; then
    echo ""
    echo "Running detailed backtest for best configuration..."
    python src/backtest.py \
        --start-window-days-back 120 \
        --end-window-days-back 0 \
        --high-frequency 1H \
        --low-frequency 15T
fi