import json
import os
import asyncio
import websockets
import pandas as pd
import time
import csv


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
DATA_FRESHNESS_THRESHOLD = 60  # seconds
SKIP_TRADE_DELAY = 60  # seconds

# Global variable for last arbitrage timestamp
last_arbitrage_timestamp = None

def check_arbitrage_opportunity():
    global last_arbitrage_timestamp
    current_time = pd.Timestamp.now()

    for symbol, data in data_buffers.items():
        if data is None:
            return
        if (current_time - data['timestamp']).seconds > DATA_FRESHNESS_THRESHOLD:
            return
    if last_arbitrage_timestamp and (current_time - last_arbitrage_timestamp).seconds < SKIP_TRADE_DELAY:
        return

    bchbtc_price = data_buffers['bchbtc']['price']
    bchusd_price = data_buffers['bchusd']['price']
    btcusd_price = data_buffers['btcusd']['price']

    final_btc = ((1 / bchbtc_price) * (1 - TRANSACTION_FEE) * bchusd_price * (1 - TRANSACTION_FEE)) / btcusd_price * (1 - TRANSACTION_FEE)
    profit_or_loss = final_btc - 1.0

    if profit_or_loss > PROFIT_THRESHOLD:
        last_arbitrage_timestamp = current_time
        print(f"Arbitrage Opportunity Detected!")
        print(f"Timestamp: {current_time}")
        print(f"BCH/BTC Price: {bchbtc_price:.8f} | BCH Obtained: {(1.0 / bchbtc_price) * (1 - TRANSACTION_FEE):.8f}")
        print(f"BCH/USD Price: {bchusd_price:.2f} | USD Obtained: {((1.0 / bchbtc_price) * (1 - TRANSACTION_FEE)) * bchusd_price * (1 - TRANSACTION_FEE):.2f}")
        print(f"BTC/USD Price: {btcusd_price:.2f} | Final BTC: {final_btc:.8f}")
        print(f"Profit: {profit_or_loss:.8f} BTC")
        print("-" * 50)


def log_trade_data(symbol, data):
    """Log the trade data to a CSV file."""
    file_path = os.path.join(HISTORICAL_DATA_DIR, f"{symbol}.csv")
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['id', 'timestamp', 'amount', 'price', 'type', 'microtimestamp', 'buy_order_id', 'sell_order_id']
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

    """Log the received data to a .log file inside the historical_data directory."""
    log_filename = os.path.join(HISTORICAL_DATA_DIR, f"{symbol}.log")
    with open(log_filename, "a") as file:
        file.write(json.dumps(data) + "\n")

    """Log the received data to a .log file."""
    log_filename = f"{symbol}.log"
    with open(log_filename, "a") as file:
        file.write(json.dumps(data) + "\n")


def process_real_time_data(symbol, data):
    try:
        # Log the received data first
        log_data_to_file(symbol, data)

        # Then process the data as before
        if 'data' in data and 'timestamp' in data['data']:
            timestamp_value = int(data['data']['timestamp'])
            timestamp = pd.to_datetime(timestamp_value, unit='s')
            price = data['data']['price']
            data_buffers[symbol] = {'timestamp': timestamp, 'price': price}
            print(f"Updated {symbol}: {timestamp} with price {price:.8f}")
            check_arbitrage_opportunity()
        elif 'event' in data and data['event'] == 'bts:subscription_succeeded':
            print(f"Subscription succeeded for {symbol}.")
        else:
            print(f"Unexpected data format for {symbol}: {data}")
    except Exception as e:
        print(f"{symbol}: An error occurred: {str(e)}: {data}")

    try:
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

    try:
        # Check if the received message is a trade message
        if 'data' in data and 'timestamp' in data['data']:
            # Convert the timestamp to an integer and then to a datetime
            timestamp_value = int(data['data']['timestamp'])
            timestamp = pd.to_datetime(timestamp_value, unit='s')

            price = data['data']['price']
            data_buffers[symbol] = {'timestamp': timestamp, 'price': price}
            print(f"Updated {symbol}: {timestamp} with price {price:.8f}")
            check_arbitrage_opportunity()

            # Log the trade data
            log_trade_data(symbol, data)
        elif 'event' in data and data['event'] == 'bts:subscription_succeeded':
            # Handle subscription confirmation messages
            print(f"Subscription succeeded for {symbol}.")
        else:
            # Print unexpected data formats for diagnosis
            print(f"Unexpected data format for {symbol}: {data}")
    except Exception as e:
        print(f"{symbol}: An error occurred: {str(e)}: {data}")

async def main():
    url = 'wss://ws.bitstamp.net'
    with open("websock-ticker-config.json", "r") as file:
        symbols = json.load(file)
    await asyncio.gather(*(subscribe(url, symbol) for symbol in symbols))

asyncio.get_event_loop().run_until_complete(main())
