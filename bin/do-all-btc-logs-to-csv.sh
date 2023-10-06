#!/bin/bash
for i in bchbtc	bchusd	btcusd; do
    echo $i
    src/btc-logs-to-csv.py ${i}.log ${i}.csv
done
