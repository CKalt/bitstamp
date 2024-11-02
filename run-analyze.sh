#!/bin/bash
. ./source-venv.sh
#DAYSET="120 90 60 30"
DAYSET="30"
for days in ${DAYSET}; do
    echo "python src/bktst.py --start-window-days-back $days --trading-window-days 30"
    python src/bktst.py --start-window-days-back $days --trading-window-days 30
done
