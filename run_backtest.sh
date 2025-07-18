#!/bin/bash
# Helper script to run backtests with common configurations

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
DATA_FILE="btcusd.log"
CONFIG_FILE="best_strategy.json"
INITIAL_BALANCE=10000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            # Quick test with last 7 days
            START_DATE=$(date -v-7d +%Y-%m-%d 2>/dev/null || date -d '7 days ago' +%Y-%m-%d)
            echo -e "${YELLOW}Running quick backtest (last 7 days)${NC}"
            ;;
        --month)
            # Test with last 30 days
            START_DATE=$(date -v-30d +%Y-%m-%d 2>/dev/null || date -d '30 days ago' +%Y-%m-%d)
            echo -e "${YELLOW}Running monthly backtest (last 30 days)${NC}"
            ;;
        --year)
            # Test with last year
            START_DATE=$(date -v-365d +%Y-%m-%d 2>/dev/null || date -d '365 days ago' +%Y-%m-%d)
            echo -e "${YELLOW}Running yearly backtest${NC}"
            ;;
        --start)
            shift
            START_DATE=$1
            ;;
        --end)
            shift
            END_DATE=$1
            ;;
        --balance)
            shift
            INITIAL_BALANCE=$1
            ;;
        --config)
            shift
            CONFIG_FILE=$1
            ;;
        --save)
            shift
            SAVE_FILE=$1
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick|--month|--year] [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--balance amount] [--config file] [--save output.json]"
            exit 1
            ;;
    esac
    shift
done

# Activate virtual environment
if [ -f "env/bin/activate" ]; then
    source env/bin/activate
else
    echo -e "${RED}Error: Virtual environment not found${NC}"
    exit 1
fi

# Build command
CMD="python src/bktst.py --data $DATA_FILE --config $CONFIG_FILE --initial-balance $INITIAL_BALANCE"

if [ ! -z "$START_DATE" ]; then
    CMD="$CMD --start-date $START_DATE"
fi

if [ ! -z "$END_DATE" ]; then
    CMD="$CMD --end-date $END_DATE"
fi

if [ ! -z "$SAVE_FILE" ]; then
    CMD="$CMD --save-results $SAVE_FILE"
fi

# Show command
echo -e "${GREEN}Running: $CMD${NC}"
echo ""

# Run backtest
$CMD

# If results were saved, show summary
if [ ! -z "$SAVE_FILE" ] && [ -f "$SAVE_FILE" ]; then
    echo ""
    echo -e "${GREEN}Results saved to: $SAVE_FILE${NC}"
    echo "Key metrics:"
    cat $SAVE_FILE | jq '{
        total_return_pct,
        sharpe_ratio,
        max_drawdown_pct,
        win_rate,
        total_trades,
        pivot_trades
    }'
fi