#!/bin/bash
# Run adaptive strategy backtest with 1-minute bars
# This matches the configuration on ck gg tst for paper trading

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Adaptive Strategy Backtest (1-minute bars)${NC}"
echo "==========================================="

# Default values
DAYS=7
DATA_FILE="btcusd.log"
OUTPUT_DIR="backtest_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --days)
            DAYS="$2"
            shift 2
            ;;
        --data)
            DATA_FILE="$2"
            shift 2
            ;;
        --start)
            START_DATE="$2"
            shift 2
            ;;
        --end)
            END_DATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--days N] [--data file] [--start YYYY-MM-DD] [--end YYYY-MM-DD]"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate output filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/adaptive_1min_${TIMESTAMP}.json"

# Build command (use indexed version for fast date access)
CMD="python3 src/backtesting/backtest_adaptive_indexed.py"
CMD="$CMD --data $DATA_FILE"
CMD="$CMD --days $DAYS"
CMD="$CMD --output $OUTPUT_FILE"

if [ ! -z "$START_DATE" ]; then
    CMD="$CMD --start $START_DATE"
fi

if [ ! -z "$END_DATE" ]; then
    CMD="$CMD --end $END_DATE"
fi

# Show what we're doing
echo -e "${YELLOW}Configuration:${NC}"
echo "  Data file: $DATA_FILE"
if [ ! -z "$START_DATE" ] && [ ! -z "$END_DATE" ]; then
    echo "  Date range: $START_DATE to $END_DATE"
else
    echo "  Days to backtest: $DAYS"
fi
echo "  Output: $OUTPUT_FILE"
echo ""

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}Error: Data file $DATA_FILE not found${NC}"
    exit 1
fi

# Activate virtual environment if it exists
if [ -f "env/bin/activate" ]; then
    source env/bin/activate
elif [ -f "source-venv.sh" ]; then
    source source-venv.sh
fi

# Run the backtest
echo -e "${GREEN}Running backtest...${NC}"
$CMD

# Show results location
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo -e "${GREEN}Results saved to: $OUTPUT_FILE${NC}"
    
    # Extract key metrics from results
    echo ""
    echo -e "${YELLOW}Key Metrics:${NC}"
    python3 -c "
import json
with open('$OUTPUT_FILE', 'r') as f:
    data = json.load(f)
    print(f'  Initial Balance: \${data[\"initial_balance\"]:,.2f}')
    print(f'  Final Balance: \${data[\"final_balance\"]:,.2f}')
    return_pct = ((data['final_balance'] - data['initial_balance']) / data['initial_balance']) * 100
    print(f'  Total Return: {return_pct:.2f}%')
    print(f'  Total Trades: {data[\"total_trades\"]}')
"
fi

echo ""
echo "Done!"