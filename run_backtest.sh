#!/bin/bash
# Backtest runner with safety features and common presets

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CONFIG="config/strategies/adaptive_default.yaml"
PYTHON="python3"

# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --quick           Run quick 7-day backtest"
    echo "  --month           Run 30-day backtest"
    echo "  --year            Run 365-day backtest"
    echo "  --full            Run on all available data"
    echo "  --config FILE     Use specific config file (default: $CONFIG)"
    echo "  --start DATE      Start date (YYYY-MM-DD)"
    echo "  --end DATE        End date (YYYY-MM-DD)"
    echo "  --output FILE     Output file (default: timestamped)"
    echo "  --trades          Show all trades in output"
    echo "  --quiet           Minimal output"
    echo "  --verbose         Verbose output"
    echo "  --low-frequency   Primary timeframe (default: 15T)"
    echo "  --high-frequency  Secondary timeframe (default: 1H)"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --quick                    # Last 7 days with default config"
    echo "  $0 --month --trades           # Last 30 days, show trades"
    echo "  $0 --config my_config.yaml    # Use custom config"
    echo "  $0 --start 2024-01-01 --end 2024-12-31  # Specific date range"
}

# Parse command line arguments
ARGS=""
# Default timeframes matching main branch
LOW_FREQ="15T"
HIGH_FREQ="1H"

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --quick)
            ARGS="$ARGS --quick"
            shift
            ;;
        --month)
            ARGS="$ARGS --month"
            shift
            ;;
        --year)
            ARGS="$ARGS --year"
            shift
            ;;
        --full)
            # No date args means use all data
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --start)
            ARGS="$ARGS --start-date $2"
            shift 2
            ;;
        --end)
            ARGS="$ARGS --end-date $2"
            shift 2
            ;;
        --output)
            ARGS="$ARGS --output-file $2"
            shift 2
            ;;
        --trades)
            ARGS="$ARGS --show-trades"
            shift
            ;;
        --quiet)
            ARGS="$ARGS --quiet"
            shift
            ;;
        --verbose)
            ARGS="$ARGS --verbose"
            shift
            ;;
        --low-frequency)
            LOW_FREQ="$2"
            shift 2
            ;;
        --high-frequency)
            HIGH_FREQ="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}Error: Configuration file not found: $CONFIG${NC}"
    exit 1
fi

# Check if btcusd.log exists
if [ ! -f "btcusd.log" ]; then
    echo -e "${RED}Error: btcusd.log not found in current directory${NC}"
    echo "Please ensure you're running from the project root directory"
    exit 1
fi

# Display what we're about to do
echo -e "${GREEN}Running backtest with configuration: $CONFIG${NC}"

# Create results directory if it doesn't exist
mkdir -p backtest_results

# Run the backtest
echo -e "${YELLOW}Starting backtest...${NC}"
echo -e "${GREEN}Using timeframes: ${LOW_FREQ} (primary) and ${HIGH_FREQ} (secondary)${NC}"
$PYTHON src/backtesting/run_backtest.py --config "$CONFIG" --low-frequency "$LOW_FREQ" --high-frequency "$HIGH_FREQ" $ARGS

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Backtest completed successfully!${NC}"
else
    echo -e "${RED}Backtest failed!${NC}"
    exit 1
fi