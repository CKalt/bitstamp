#!/bin/bash
. ./source-venv.sh
DAYSET="120 90 60 30"
#DAYSET="30"
for days in ${DAYSET}; do
    echo "python src/btc_log_analyzer.py --start-window-days-back $days --trading-window-days 30 --high-frequency '45T' --low-frequency '10T'"
    python src/btc_log_analyzer.py --start-window-days-back $days --trading-window-days 30 --high-frequency '60T' --low-frequency '10T'
done
