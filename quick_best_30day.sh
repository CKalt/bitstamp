#!/bin/bash
# Quick test of key MA combinations for last 30 days

echo "Testing best MA combinations for last 30 days..."
echo "This will take a few minutes..."

# Test current strategy
echo -e "\n1. Testing current MA 6/34..."
source source-venv.sh
python src/bktst.py \
  --start-window-days-back 30 \
  --end-window-days-back 0 \
  --high-frequency 1H \
  --low-frequency 15T \
  --initial 10000 \
  --ma-short 6 \
  --ma-long 34 \
  > test_6_34.log 2>&1

echo "Results for MA 6/34:"
grep -E "Total Return:|Total Trades:|Sharpe Ratio:" test_6_34.log || echo "Failed"

# Test slower combination
echo -e "\n2. Testing MA 10/20..."
python src/bktst.py \
  --start-window-days-back 30 \
  --end-window-days-back 0 \
  --high-frequency 1H \
  --low-frequency 15T \
  --initial 10000 \
  --ma-short 10 \
  --ma-long 20 \
  > test_10_20.log 2>&1

echo "Results for MA 10/20:"
grep -E "Total Return:|Total Trades:|Sharpe Ratio:" test_10_20.log || echo "Failed"

# Test medium combination
echo -e "\n3. Testing MA 20/50..."
python src/bktst.py \
  --start-window-days-back 30 \
  --end-window-days-back 0 \
  --high-frequency 1H \
  --low-frequency 15T \
  --initial 10000 \
  --ma-short 20 \
  --ma-long 50 \
  > test_20_50.log 2>&1

echo "Results for MA 20/50:"
grep -E "Total Return:|Total Trades:|Sharpe Ratio:" test_20_50.log || echo "Failed"

echo -e "\nDone! Check the log files for detailed results."