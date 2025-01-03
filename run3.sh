python src/bktst.py \
  --start-window-days-back 90 \
  --end-window-days-back 60 \
  --high-frequency 1H \
  --low-frequency 15T

mv best_strategy.json best_strategy-run3.json
