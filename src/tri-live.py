# src/tri-live.py
import json
import os
import asyncio
import websockets
import pandas as pd
import time
import csv
import subprocess

print(f"Current working directory: {os.getcwd()}")


TRADE_COUNT = 0  # default, you might set this from command line args
BTC_AMOUNT = 0.001000  # default, you might set this from command line args
DRY_RUN = True  # default, you might set this from command line args
ALWAYS_PROFITABLE = True  # this can be changed manually if needed
if not DRY_RUN:
    ALWAYS_PROFITABLE = False


# Directory to save the historical data
HISTORICAL_DATA_DIR = "historical_data"

# Ensure the directory exists
if not os.path.exists(HISTORICAL_DATA_DIR):
    os.makedirs(HISTORICAL_DATA_DIR)

# Global data buffers for real-time data
data_buffers = {
    'bchbtc': None,
    'bchusd': None,
    'btcusd': None
}


def execute_trade(bchbtc_price, bchusd_price, btcusd_price, current_time):
    global TRADE_COUNT
    global BTC_AMOUNT
    global DRY_RUN

    # Extract necessary data
    trade_timestamp = current_time.timestamp()
    trade_bch_amount = (BTC_AMOUNT / bchbtc_price) * (1 - TRANSACTION_FEE)
    trade_usd_amount = trade_bch_amount * bchusd_price * (1 - TRANSACTION_FEE)

    # Decide the script to run based on DRY_RUN mode
    script_name = 'src/place-order-dry-run.py' if DRY_RUN else 'src/place-order.py'

    # Adjusted the amount used in commands for each trade to reflect the correct amount for the particular trade.
    trade1_cmd = f"python {script_name} --order_type 'market-sell' --currency_pair 'bchbtc' --amount {BTC_AMOUNT} --price {bchbtc_price} --log_dir {trade_timestamp}"
    trade2_cmd = f"python {script_name} --order_type 'market-sell' --currency_pair 'bchusd' --amount {trade_bch_amount} --price {bchusd_price} --log_dir {trade_timestamp}"
    trade3_cmd = f"python {script_name} --order_type 'market-buy' --currency_pair 'btcusd' --amount {trade_usd_amount} --price {btcusd_price} --log_dir {trade_timestamp}"

    if DRY_RUN:
        print("Dry Run: ", trade1_cmd)
        print("Dry Run: ", trade2_cmd)
        print("Dry Run: ", trade3_cmd)
    else:
        # These subprocess calls are blocking, consider subprocess.Popen to run them concurrently.
        subprocess.run(trade1_cmd, shell=True)
        subprocess.run(trade2_cmd, shell=True)
        subprocess.run(trade3_cmd, shell=True)

    TRADE_COUNT += 1

async def subscribe(url: str, symbol: str):
    channel = f"live_trades_{symbol}"

    while True:  # Keep trying to reconnect
        try:
            async with websockets.connect(url) as websocket:
                await websocket.send(json.dumps({
                    "event": "bts:subscribe",
                    "data": {
                        "channel": channel
                    }
                }))
                async for message in websocket:
                    data = json.loads(message)
                    process_real_time_data(symbol, data)
        except websockets.ConnectionClosed:
            print(f"{symbol}: Connection closed, trying to reconnect in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"{symbol}: An error occurred: {e}")

# Constants
PROFIT_THRESHOLD = 0.0001
TRANSACTION_FEE = 0.002
# Freshness thresholds for different currency pairs
DATA_FRESHNESS_THRESHOLDS = {
    'btcusd': 120,  # 2 minutes for BTC/USD
    'bchbtc': 1800, # 30 minutes for BCH/BTC
    'bchusd': 1800  # 30 minutes for BCH/USD
}
SKIP_TRADE_DELAY = 60  # seconds

# Global variable for last arbitrage timestamp
last_arbitrage_timestamp = None


def check_arbitrage_opportunity():
    global last_arbitrage_timestamp
    print("Checking for arbitrage opportunity...")
    # Print the value of ALWAYS_PROFITABLE
    print(f"Value of ALWAYS_PROFITABLE: {ALWAYS_PROFITABLE}")
    current_time = pd.Timestamp.now()

    # Check if any data buffer is missing or None
    missing_or_none_data = [symbol for symbol, data in data_buffers.items() if data is None]
    if missing_or_none_data:
        print(f"Missing or None data for symbols: {', '.join(missing_or_none_data)}. Waiting for data.")
        return
    
    if not ALWAYS_PROFITABLE:
        for symbol, data in data_buffers.items():
            # Use the specific freshness threshold for the current symbol
            freshness_threshold = DATA_FRESHNESS_THRESHOLDS.get(symbol, 60)
            age_of_data = (current_time - data['timestamp']).seconds
            if age_of_data > freshness_threshold:
                recommended_threshold = age_of_data + 10  # Adding 10 seconds for buffer
                print(f"Data for {symbol} is {age_of_data} seconds old, which exceeds the freshness threshold of {freshness_threshold} seconds. "
                      f"If you want this to be considered fresh, raise the threshold to at least {recommended_threshold} seconds.")
                return

        if last_arbitrage_timestamp and (current_time - last_arbitrage_timestamp).seconds < SKIP_TRADE_DELAY:
            print(f"Waiting due to trade delay. Exiting check.")
            return


    bchbtc_price = data_buffers['bchbtc']['price']
    bchusd_price = data_buffers['bchusd']['price']
    btcusd_price = data_buffers['btcusd']['price']

    final_btc = ((1 / bchbtc_price) * (1 - TRANSACTION_FEE) * bchusd_price *
                 (1 - TRANSACTION_FEE)) / btcusd_price * (1 - TRANSACTION_FEE)
    profit_or_loss = final_btc - 1.0

    if ALWAYS_PROFITABLE or profit_or_loss > PROFIT_THRESHOLD:
        # Execute trades immediately upon detecting opportunity
        print("about to call execute_trade");
        execute_trade(bchbtc_price, bchusd_price, btcusd_price, current_time)
        print("back from execute_trade");

        last_arbitrage_timestamp = current_time
        print("Arbitrage Opportunity Detected!")
        print(f"Timestamp: {current_time}")
        print(
            f"BCH/BTC Price: {bchbtc_price:.8f} | BCH Obtained: {(1.0 / bchbtc_price) * (1 - TRANSACTION_FEE):.8f}")
        print(
            f"BCH/USD Price: {bchusd_price:.2f} | USD Obtained: {((1.0 / bchbtc_price) * (1 - TRANSACTION_FEE)) * bchusd_price * (1 - TRANSACTION_FEE):.2f}")
        print(
            f"BTC/USD Price: {btcusd_price:.2f} | Final BTC: {final_btc:.8f}")
        print(f"Profit: {profit_or_loss:.8f} BTC")
        print("-" * 50)

        # Check and create trades directory if not exists
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        TRADES_DIR = os.path.join(SCRIPT_DIR, 'trades')
        if not os.path.exists(TRADES_DIR):
            print(f"Making TRADES_DIR ({TRADES_DIR})...")
            os.makedirs(TRADES_DIR)

        # Write trade details to the file
        with open('trades/live-trades.txt', 'a') as file:
            file.write(
                f"Mode: {'DRY_RUN' if DRY_RUN else 'LIVE'}, {'ALWAYS_PROFITABLE' if ALWAYS_PROFITABLE else 'REAL PROFIT'}\n")
            file.write(
                f"Trade {last_arbitrage_timestamp}: Timestamp (Epoch): {current_time.timestamp()}\n")
            file.write(
                f"Trade {last_arbitrage_timestamp}: Timestamp (Human): {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write("-" * 50 + "\n")
            file.write(
                f"1. Trading 1 BTC for BCH using {bchbtc_price:.8f} BCH/BTC\n")
            file.write(
                f"Timestamp (Epoch) for BCH/BTC price: {data_buffers['bchbtc']['timestamp'].timestamp()}\n")
            file.write(f"Initial: 1 BTC\n")
            file.write(
                f"After Trade (minus fee): {(1.0 / bchbtc_price) * (1 - TRANSACTION_FEE):.8f} BCH\n")
            file.write("-" * 50 + "\n")
            file.write(
                f"2. Trading {(1.0 / bchbtc_price) * (1 - TRANSACTION_FEE):.8f} BCH for USD using {bchusd_price:.2f} BCH/USD\n")
            file.write(
                f"Timestamp (Epoch) for BCH/USD price: {data_buffers['bchusd']['timestamp'].timestamp()}\n")
            file.write(
                f"After Trade (minus fee): {((1.0 / bchbtc_price) * (1 - TRANSACTION_FEE)) * bchusd_price * (1 - TRANSACTION_FEE):.2f} USD\n")
            file.write("-" * 50 + "\n")
            file.write(
                f"3. Trading {((1.0 / bchbtc_price) * (1 - TRANSACTION_FEE)) * bchusd_price * (1 - TRANSACTION_FEE):.2f} USD for BTC using {btcusd_price:.2f} USD/BTC\n")
            file.write(
                f"Timestamp (Epoch) for BTC/USD price: {data_buffers['btcusd']['timestamp'].timestamp()}\n")
            file.write(f"After Trade (minus fee): {final_btc:.8f} BTC\n")
            file.write("-" * 50 + "\n")
            file.write(
                f"Profit for Trade {last_arbitrage_timestamp}: {profit_or_loss:.8f} BTC\n")
            file.write("=" * 50 + "\n\n")


def log_trade_data(symbol, data):
    """Log the trade data to a CSV file."""
    file_path = os.path.join(HISTORICAL_DATA_DIR, f"{symbol}.csv")
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['id', 'timestamp', 'amount', 'price', 'type',
                      'microtimestamp', 'buy_order_id', 'sell_order_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        trade_data = {
            'id': data['data']['id'],
            'timestamp': data['data']['timestamp'],
            'amount': data['data']['amount'],
            'price': data['data']['price'],
            'type': data['data']['type'],
            'microtimestamp': data['data']['microtimestamp'],
            'buy_order_id': data['data']['buy_order_id'],
            'sell_order_id': data['data']['sell_order_id'],
        }
        writer.writerow(trade_data)


def log_data_to_file(symbol, data):
    """Log the received data to a .log file inside the historical_data directory."""
    log_filename = os.path.join(HISTORICAL_DATA_DIR, f"{symbol}.log")
    with open(log_filename, "a") as file:
        file.write(json.dumps(data) + "\n")


def process_real_time_data(symbol, data):
    try:
        # Check if the received message is a trade message
        if 'data' in data and 'timestamp' in data['data']:
            timestamp_value = int(data['data']['timestamp'])
            timestamp = pd.to_datetime(timestamp_value, unit='s')
            price = data['data']['price']
            data_buffers[symbol] = {'timestamp': timestamp, 'price': price}
            print(f"Updated {symbol}: {timestamp} with price {price:.8f}")
            log_trade_data(symbol, data)
            check_arbitrage_opportunity()
        elif 'event' in data and data['event'] == 'bts:subscription_succeeded':
            print(f"Subscription succeeded for {symbol}.")
        else:
            print(f"Unexpected data format for {symbol}: {data}")
    except Exception as e:
        print(f"{symbol}: An error occurred: {str(e)}: {data}")


async def main():
    url = 'wss://ws.bitstamp.net'
    with open("websock-ticker-config.json", "r") as file:
        symbols = json.load(file)
    await asyncio.gather(*(subscribe(url, symbol) for symbol in symbols))

asyncio.get_event_loop().run_until_complete(main())
