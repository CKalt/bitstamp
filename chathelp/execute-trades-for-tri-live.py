Below I list a python script `src/tri-live.py` that I wish you to enhance.

It monitors live ticker data using bitstamp's websocket v2 api, and waits and looks for triangle arbitrage opportunities. It searchings for these opportuntities by analyzing theoretical trades among a set of three preset crypto currency pairs. It then generates the three "hypothetical" arbitrage trades for each such opportunity found and sends the detail to a log file called trades/live-trades.txt.

My goal now, is to add code so that once the hypothetical opportunity has been discovered, the trades are executed, in additional the of simply logged. By this I mean a set of 3 parallel (non-blocking) shell calls out to a python script designed to execute trades is made.  The python script is called `src/place-order.py` and it is described below details of how it will accept command line arguments that will place market orders to buy or sell bitstamp currency pairs. It is very important that the calls to execute these obtain their own unix process so that they process parallel. In this way we seek to increase the chances that market orders will quickly exploit the arbitrage trading opportunity before it disapears. We do not want to execute them sequentially because there is no reason to have any trade wait for another to complete.

Here is the output of one such opportunity that becomes logged to the file named:  trades/live-trades.txt

Trade 2023-10-26 14:38:29.825731: Timestamp (Epoch): 1698331109.825731
Trade 2023-10-26 14:38:29.825731: Timestamp (Human): 2023-10-26 14:38:29
--------------------------------------------------
1. Trading 1 BTC for BCH using 0.00716184 BCH/BTC
Timestamp (Epoch) for BCH/BTC price: 1698331109.0
Initial: 1 BTC
After Trade (minus fee): 139.34966433 BCH
--------------------------------------------------
2. Trading 139.34966433 BCH for USD using 244.70 BCH/USD
Timestamp (Epoch) for BCH/USD price: 1698331087.0
After Trade (minus fee): 34030.67 USD
--------------------------------------------------
3. Trading 34030.67 USD for BTC using 33938.00 USD/BTC
Timestamp (Epoch) for BTC/USD price: 1698331107.0
After Trade (minus fee): 1.00072496 BTC
--------------------------------------------------
Profit for Trade 2023-10-26 14:38:29.825731: 0.00072496 BTC
==================================================

Please note that the first trade is for 1 BTC. Please also note that each of the two subsequent trades, the amount traded is comoputed based on the last price observed from the ticker feed for each currency pair.

Please also note the line above: "Trade 2023-10-26 14:38:29.825731: Timestamp (Epoch): 1698331109.825731"

We will use the Timestamp (Epoch) '1698331109.825731' as the --log_dir parameter passed to each of the 3 trades to be executed by the src/place-order.py script.

I wish a new command line option --amount to be added to the src/tri-live.py script and logic to provide the user with the means to control this value and thus over the 3 trades to be executed for each arbitrage opportunity discovered.

I wish you to find in the code where this log is generated, and enhance the code so that when the user launches the script with the option --trade-count <qty> --btc_amount 0.001000, where <qty> is some integer value greater than 0, and the floating point value next to --btc_amount  indicates how BTC should be traded with the first pair.

Then for that number of opportunities found that would result in the above such output being generated I need you to launch a shell process so that it runs in parallel and performs the following as though it were from the bash command line:

python src/place-order.py --order_type "market-sell" --currency_pair "btcusd" --btc_amount 0.001000 --price 34564.0 --log_dir 1698331109.825731

It is important that these run without blocking. The linux system should cause a parallel process to launch so that each of these trades may be run as quickly as possible.

All we want our src/tri-live.py to do is to launch these.

I also wish you to record into the trades/live-trades.txt these very lines that were used to execute these trades.

The possible --order_type values are "market-buy" and "market-sell"

See if you can figure out whether to use a market-buy or market-sell based on the trade pair, that of course is very important. 

For --price simply put the last price that at which the pair was traded.

Finally please include an option called --dry-run which will display the src/place-order.py command lines but not execute them.

Now here's the list of src/tri-live.py

--------------------------------------------------------