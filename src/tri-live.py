# src/tri-live.py
import argparse
import json
import os
import asyncio
import websockets
import pandas as pd
import time
import csv
import subprocess
import heapq
import random
import itertools

TRADE_COUNT = 0  # default, you might set this from command line args
BTC_AMOUNT = 0.001000  # default, you might set this from command line args
DRY_RUN = True  # default, you might set this from command line args
ALWAYS_PROFITABLE = True  # this can be changed manually if needed

# Directories
HISTORICAL_DATA_DIR = "historical_data"
TEST_DATA_DIR = "test_data"

# Ensure directories exists
if not os.path.exists(HISTORICAL_DATA_DIR):
    os.makedirs(HISTORICAL_DATA_DIR)
if not os.path.exists(TEST_DATA_DIR):
    os.makedirs(TEST_DATA_DIR)

# Global data buffers for real-time data
data_buffers = {
    'bchbtc': None,
    'bchusd': None,
    'btcusd': None
}


def merge_sorted(*iterables, key=None):
    h = []
    for iterable in iterables:
        for item in iterable:
            heapq.heappush(h, (key(item), item))
    while h:
        yield heapq.heappop(h)[1]


def generate_test_data():
    num_initial_events = 10
    num_profitable_events = 3
    min_interval = 1
    max_interval = 5

    for symbol in symbols:
        base_timestamp = pd.Timestamp.now()
        test_data = []

        # Generate initial events with randomized timestamps
        for i in range(num_initial_events):
            interval = random.randint(min_interval, max_interval)
            timestamp = base_timestamp + pd.Timedelta(seconds=interval)
            base_timestamp = timestamp

            test_data.append({
                'id': str(i + 1),
                'timestamp': str(int(timestamp.timestamp())),
                'amount': f'{random.uniform(0.1, 1.0):.8f}',
                'price': f'{random.uniform(1000, 2000):.8f}',
                'type': str(random.randint(0, 1)),
                'microtimestamp': f'{int(timestamp.timestamp() * 1000000)}',
                'buy_order_id': str(random.randint(1000000000, 9999999999)),
                'sell_order_id': str(random.randint(1000000000, 9999999999)),
            })

        # Generate profitable events
        profitable_prices = {
            'bchbtc': 0.02,
            'bchusd': 2000.0,
            'btcusd': 30000.0,
        }

        for i in range(num_profitable_events):
            interval = random.randint(min_interval, max_interval)
            timestamp = base_timestamp + pd.Timedelta(seconds=interval)
            base_timestamp = timestamp

            test_data.append({
                'id': str(num_initial_events + i + 1),
                'timestamp': str(int(timestamp.timestamp())),
                'amount': f'{random.uniform(0.1, 1.0):.8f}',
                'price': f'{profitable_prices[symbol]:.8f}',
                'type': str(random.randint(0, 1)),
                'microtimestamp': f'{int(timestamp.timestamp() * 1000000)}',
                'buy_order_id': str(random.randint(1000000000, 9999999999)),
                'sell_order_id': str(random.randint(1000000000, 9999999999)),
            })

        # Write test data to CSV file
        file_path = os.path.join(TEST_DATA_DIR, f"{symbol}_test.csv")
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['id', 'timestamp', 'amount', 'price', 'type',
                          'microtimestamp', 'buy_order_id', 'sell_order_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(test_data)

# Generate expected results JSON
    expected_results = {
        "arbitrage_opportunity_detected": True,
        "profitable_timestamp": int(base_timestamp.timestamp()),
        "trades": [
            {
                "trade_type": "sell",
                "currency_pair": "bchbtc",
                "amount": BTC_AMOUNT,
                "price": profitable_prices['bchbtc']
            },
            {
                "trade_type": "sell",
                "currency_pair": "bchusd",
                "amount": (BTC_AMOUNT / profitable_prices['bchbtc']) * (1 - TRANSACTION_FEE),
                "price": profitable_prices['bchusd']
            },
            {
                "trade_type": "buy",
                "currency_pair": "btcusd",
                "amount": ((BTC_AMOUNT / profitable_prices['bchbtc']) * (1 - TRANSACTION_FEE) * profitable_prices['bchusd'] * (1 - TRANSACTION_FEE)),
                "price": profitable_prices['btcusd']
            }
        ],
        "profit": ((1 / profitable_prices['bchbtc']) * (1 - TRANSACTION_FEE) * profitable_prices['bchusd'] * (1 - TRANSACTION_FEE)) / profitable_prices['btcusd'] * (1 - TRANSACTION_FEE) - 1.0
    }

    expected_results_file = os.path.join(
        TEST_DATA_DIR, "expected_results.json")
    with open(expected_results_file, "w") as json_file:
            json.dump(expected_results, json_file, indent=4)

    print("Test data and expected results generated successfully.")

def read_test_data(symbol):
    file_path = os.path.join(TEST_DATA_DIR, f"{symbol}_test.csv")
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
    return data


def read_historical_data(symbol):
    file_path = os.path.join(HISTORICAL_DATA_DIR, f"{symbol}.csv")
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
    return data


def execute_trade(bchbtc_price, bchusd_price, btcusd_price, current_time):
    global TRADE_COUNT
    global BTC_AMOUNT
    global DRY_RUN
    global actual_results

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

        print("Trade details:")
        print(f"  Trade Type: sell")
        print(f"  Currency Pair: bchbtc")
        print(f"  Amount: {BTC_AMOUNT}")
        print(f"  Price: {bchbtc_price}")

        # Capture the trades in actual_results
        actual_results["trades"].append({
            "trade_type": "sell",
            "currency_pair": "bchbtc",
            "amount": BTC_AMOUNT,
            "price": bchbtc_price
        })
        actual_results["trades"].append({
            "trade_type": "sell",
            "currency_pair": "bchusd",
            "amount": trade_bch_amount,
            "price": bchusd_price
        })
        actual_results["trades"].append({
            "trade_type": "buy",
            "currency_pair": "btcusd",
            "amount": trade_usd_amount,
            "price": btcusd_price
        })
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
    'bchbtc': 1800,  # 30 minutes for BCH/BTC
    'bchusd': 1800  # 30 minutes for BCH/USD
}
SKIP_TRADE_DELAY = 60  # seconds

# Global variable for last arbitrage timestamp
last_arbitrage_timestamp = None


def check_arbitrage_opportunity():
    global last_arbitrage_timestamp
    global actual_results
    print("Checking for arbitrage opportunity...")
    # Print the value of ALWAYS_PROFITABLE
    print(f"Value of ALWAYS_PROFITABLE: {ALWAYS_PROFITABLE}")
    current_time = pd.Timestamp.now()

    # Check if any data buffer is missing or None
    missing_or_none_data = [symbol for symbol,
                            data in data_buffers.items() if data is None]
    if missing_or_none_data:
        print(
            f"Missing or None data for symbols: {', '.join(missing_or_none_data)}. Waiting for data.")
        return None

    if not ALWAYS_PROFITABLE:
        for symbol, data in data_buffers.items():
            # Use the specific freshness threshold for the current symbol
            freshness_threshold = DATA_FRESHNESS_THRESHOLDS.get(symbol, 60)
            age_of_data = (current_time - data['timestamp']).seconds
            if age_of_data > freshness_threshold:
                recommended_threshold = age_of_data + 10  # Adding 10 seconds for buffer
                print(f"Data for {symbol} is {age_of_data} seconds old, which exceeds the freshness threshold of {freshness_threshold} seconds. "
                      f"If you want this to be considered fresh, raise the threshold to at least {recommended_threshold} seconds.")
                return None

        if last_arbitrage_timestamp and (current_time - last_arbitrage_timestamp).seconds < SKIP_TRADE_DELAY:
            print(f"Waiting due to trade delay. Exiting check.")
            return None

    bchbtc_price = data_buffers['bchbtc']['price']
    bchusd_price = data_buffers['bchusd']['price']
    btcusd_price = data_buffers['btcusd']['price']

    print(f"bchbtc_price: {bchbtc_price}")
    print(f"bchusd_price: {bchusd_price}")
    print(f"btcusd_price: {btcusd_price}")

    final_btc = ((1 / bchbtc_price) * (1 - TRANSACTION_FEE) * bchusd_price *
                 (1 - TRANSACTION_FEE)) / btcusd_price * (1 - TRANSACTION_FEE)
    profit_or_loss = final_btc - 1.0

    print(f"profit_or_loss: {profit_or_loss}")

    if ALWAYS_PROFITABLE or profit_or_loss > PROFIT_THRESHOLD:
        print(
            f"Arbitrage opportunity detected with profit_or_loss: {profit_or_loss}")

        # Execute trades immediately upon detecting opportunity
        print("about to call execute_trade")
        execute_trade(bchbtc_price, bchusd_price, btcusd_price, current_time)
        print("back from execute_trade")

        # Set actual_results["arbitrage_opportunity_detected"] to true
        actual_results["arbitrage_opportunity_detected"] = True

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
        trades_file = os.path.join(TRADES_DIR, 'live-trades.txt')
        with open(trades_file, 'a') as file:
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

        return profit_or_loss

    return None


def log_ticker_data(symbol, data):
    """Log the ticker data to a CSV file."""
    file_path = os.path.join(HISTORICAL_DATA_DIR, f"{symbol}.csv")
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['id', 'timestamp', 'amount', 'price', 'type',
                      'microtimestamp', 'buy_order_id', 'sell_order_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        ticker_data = {
            'id': data['data']['id'],
            'timestamp': data['data']['timestamp'],
            'amount': data['data']['amount'],
            'price': data['data']['price'],
            'type': data['data']['type'],
            'microtimestamp': data['data']['microtimestamp'],
            'buy_order_id': data['data']['buy_order_id'],
            'sell_order_id': data['data']['sell_order_id'],
        }
        writer.writerow(ticker_data)


def process_real_time_data(symbol, data):
    try:
        # Check if the received message is a trade message
        if 'data' in data and 'timestamp' in data['data']:
            timestamp_value = int(data['data']['timestamp'])
            timestamp = pd.to_datetime(timestamp_value, unit='s')
            price = float(data['data']['price'])  # Convert price to float
            data_buffers[symbol] = {'timestamp': timestamp, 'price': price}
            print(f"Updated {symbol}: {timestamp} with price {price:.8f}")
            log_ticker_data(symbol, data)
            check_arbitrage_opportunity()
        elif 'event' in data and data['event'] == 'bts:subscription_succeeded':
            print(f"Subscription succeeded for {symbol}.")
        else:
            print(f"Unexpected data format for {symbol}: {data}")
    except Exception as e:
        print(f"{symbol}: An error occurred: {str(e)}: {data}")

def display_test_results(actual_results):
    print("Test Results:")
    print("Arbitrage opportunity detected!")
    print("Simulated Trades (Dry Run):")
    for trade in actual_results["trades"]:
        print(f"{trade['trade_type'].capitalize()} {trade['amount']} {trade['currency_pair'].upper()} at a price of {trade['price']}")
    print(f"Profit: {actual_results['profit']:.8f} BTC")

    # Load expected results from JSON file
    with open(os.path.join(TEST_DATA_DIR, "expected_results.json"), "r") as json_file:
        expected_results = json.load(json_file)

    print("Expected results:")
    print(json.dumps(expected_results, indent=4))
    print("Actual results:")
    print(json.dumps(actual_results, indent=4))

    # Compare actual results with expected results
    tolerance = 1e-8
    if (
        actual_results["arbitrage_opportunity_detected"] == expected_results["arbitrage_opportunity_detected"]
        and abs(actual_results["profitable_timestamp"] - expected_results["profitable_timestamp"]) < tolerance
        and len(actual_results["trades"]) == len(expected_results["trades"])
        and all(
            trade["trade_type"] == expected_trade["trade_type"]
            and trade["currency_pair"] == expected_trade["currency_pair"]
            and abs(trade["amount"] - expected_trade["amount"]) < tolerance
            and abs(trade["price"] - expected_trade["price"]) < tolerance
            for trade, expected_trade in zip(actual_results["trades"], expected_results["trades"])
        )
        and abs(actual_results["profit"] - expected_results["profit"]) < tolerance
    ):
        print("Test passed! Actual results match the expected results.")
    else:
        print("Test failed! Actual results do not match the expected results.")
        print("Expected results:")
        print(json.dumps(expected_results, indent=4))
        print("Actual results:")
        print(json.dumps(actual_results, indent=4))


def display_run_options():
    print("Run Options:")
    print(f"  Ticker Mode: {args.ticker_mode}")
    print(f"  Dry Run: {DRY_RUN}")
    print(f"  Always Profitable: {ALWAYS_PROFITABLE}")
    print(f"  Generate Test Data: {args.generate_test_data}")
    print()


async def async_data_stream(data):
    for item in data:
        yield item

symbols = ['bchbtc', 'bchusd', 'btcusd']


async def main():
    global actual_results
    global DRY_RUN
    global ALWAYS_PROFITABLE

    parser = argparse.ArgumentParser(description='Crypto Arbitrage Bot')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ticker-mode', choices=['historical', 'live', 'test'], default='live',
                       help='Ticker data mode: "historical" for historical data, "live" for real-time data, "test" for test data')
    group.add_argument('--generate-test-data', action='store_true',
                       help='Generate randomized test data')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run in dry-run mode (simulate trades)')
    parser.add_argument('--always-profitable', action='store_true',
                        help='Consider every arbitrage opportunity as profitable (default: False)')
    args = parser.parse_args()

    if args.generate_test_data:
        generate_test_data()
        print("Test data generated successfully. Exiting.")
        return

    DRY_RUN = args.dry_run or args.ticker_mode == 'test' or args.ticker_mode == 'historical'
    ALWAYS_PROFITABLE = args.always_profitable

    if args.generate_test_data:
        generate_test_data()
        print("Test data generated successfully. Exiting.")
        return

    if args.ticker_mode == 'test':
        actual_results = {
            "arbitrage_opportunity_detected": False,
            "profitable_timestamp": None,
            "trades": [],
            "profit": 0.0
        }

        test_data = [read_test_data(symbol) for symbol in symbols]
        # Ensure that 'timestamp' is converted to integer if it's a string
        for data_list in test_data:
            for data in data_list:
                data['timestamp'] = int(data['timestamp'])

        merged_data = sorted(itertools.chain(*test_data),
                             key=lambda x: pd.to_datetime(x['timestamp'], unit='s'))

        print("Initial actual_results:")
        print(json.dumps(actual_results, indent=4))

        data_stream = async_data_stream(merged_data)

        async for data in data_stream:
            print(f"Processing data: {data}")

            # Find the symbol safely, provide a default value if not found
            symbol = next(
                (s for s in symbols if f"{s}_test.csv" in data['id']), None)
            if symbol is None:
                continue  # Skip this iteration if the symbol was not found

            process_real_time_data(symbol, data)

            print("Current actual_results after processing data:")
            print(json.dumps(actual_results, indent=4))

            # Check if an arbitrage opportunity was detected
            if actual_results["arbitrage_opportunity_detected"]:
                print("Arbitrage opportunity detected in main!")
                print(f"Profitable Timestamp: {actual_results['profitable_timestamp']}")
                print(f"Profit: {actual_results['profit']}")

        print("Final actual_results before displaying test results:")
        print(json.dumps(actual_results, indent=4))

        display_test_results(actual_results)

    elif args.ticker_mode == 'historical':
        data_reader = read_test_data if args.ticker_mode == 'test' else read_historical_data
        data_streams = [data_reader(symbol) for symbol in symbols]
        merged_data = sorted(itertools.chain(
            *data_streams), key=lambda x: pd.to_datetime(x['timestamp'], unit='s'))

        for data in merged_data:
            symbol = next(
                symbol for symbol in symbols if f"{symbol}_test.csv" in data['id'] or f"{symbol}.csv" in data['id'])
            process_real_time_data(symbol, data)

    else:  # 'live' mode
        url = 'wss://ws.bitstamp.net'
        await asyncio.gather(*(subscribe(url, symbol) for symbol in symbols))


asyncio.get_event_loop().run_until_complete(main())