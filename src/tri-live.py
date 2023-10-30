
import json
import os
import asyncio
import websockets
import pandas as pd
import time
import csv
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
HISTORICAL_DATA_DIR = "historical_data"
PROFIT_THRESHOLD = 0.0001
TRANSACTION_FEE = 0.002
DATA_FRESHNESS_THRESHOLD = 60  # seconds
SKIP_TRADE_DELAY = 60  # seconds
WEB_SOCKET_URL = 'wss://ws.bitstamp.net'

# Global Variables
data_buffers = {
    'bchbtc': None,
    'bchusd': None,
    'btcusd': None
}
last_arbitrage_timestamp = None

# Ensure historical data directory exists
if not os.path.exists(HISTORICAL_DATA_DIR):
    os.makedirs(HISTORICAL_DATA_DIR)

def process_real_time_data(symbol, data):
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

async def subscribe(url: str, symbol: str):
    channel = f"live_trades_{symbol}"
    while True:
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

    # [Rest of the arbitrage checking logic remains unchanged...]

async def main():
    with open("websock-ticker-config.json", "r") as file:
        symbols = json.load(file)
    await asyncio.gather(*(subscribe(WEB_SOCKET_URL, symbol) for symbol in symbols))

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

# [Rest of the code remains unchanged...]

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
