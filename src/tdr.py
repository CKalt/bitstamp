#!/usr/bin/env python
# src/tdr.py

import asyncio
import websockets
import json
import time
import argparse
import hashlib
import hmac
import uuid
from urllib.parse import urlencode
import requests
from collections import deque
import cmd
import threading
from datetime import datetime, timedelta

class CandlestickData:
    def __init__(self):
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = 0
        self.trades = 0

class CryptoDataManager:
    def __init__(self, symbols):
        self.data = {symbol: deque(maxlen=3600) for symbol in symbols}  # Store last hour of data
        self.candlesticks = {symbol: {} for symbol in symbols}  # Store current candlestick data
        self.observers = []  # List of functions to call when a candlestick is completed

    def add_trade(self, symbol, price, timestamp):
        price = float(price)
        self.data[symbol].append((timestamp, price))
        
        # Update candlestick data
        minute = timestamp - (timestamp % 60)  # Round down to the nearest minute
        if minute not in self.candlesticks[symbol]:
            self.candlesticks[symbol][minute] = CandlestickData()
        
        candle = self.candlesticks[symbol][minute]
        if candle.open is None:
            candle.open = price
        candle.high = max(candle.high or price, price)
        candle.low = min(candle.low or price, price)
        candle.close = price
        candle.volume += 1  # Assuming 1 unit per trade, adjust if volume data is available
        candle.trades += 1

        # Check if the candlestick is complete
        current_time = int(time.time())
        if current_time >= minute + 60:
            self._complete_candlestick(symbol, minute)

    def _complete_candlestick(self, symbol, minute):
        candle = self.candlesticks[symbol].pop(minute, None)
        if candle:
            for observer in self.observers:
                observer(symbol, minute, candle)

    def add_observer(self, callback):
        self.observers.append(callback)

    def get_current_price(self, symbol):
        if self.data[symbol]:
            return self.data[symbol][-1][1]
        return None

    def get_price_range(self, symbol, minutes):
        now = time.time()
        relevant_data = [price for ts, price in self.data[symbol] if now - ts <= minutes * 60]
        if relevant_data:
            return min(relevant_data), max(relevant_data)
        return None, None

async def subscribe_to_websocket(url: str, symbol: str, data_manager):
    channel = f"live_trades_{symbol}"

    while True:  # Keep trying to reconnect
        try:
            async with websockets.connect(url) as websocket:
                print(f"Connected to WebSocket for {symbol}")
                
                # Subscribing to the channel.
                await websocket.send(json.dumps({
                    "event": "bts:subscribe",
                    "data": {
                        "channel": channel
                    }
                }))

                # Receiving messages.
                async for message in websocket:
                    print(f"{symbol}: {message}")
                    data = json.loads(message)
                    if data['event'] == 'trade':
                        price = data['data']['price']
                        timestamp = data['data']['timestamp']
                        data_manager.add_trade(symbol, price, timestamp)

        except websockets.ConnectionClosed:
            print(f"{symbol}: Connection closed, trying to reconnect in 5 seconds...")
            time.sleep(5)  # Wait for 5 seconds before trying to reconnect
        except Exception as e:
            print(f"{symbol}: An error occurred: {e}")
            time.sleep(5)  # Wait for 5 seconds before trying to reconnect

class OrderPlacer:
    def __init__(self, config_file='.bitstamp'):
        self.config = self.read_config(config_file)
        self.api_key = self.config['api_key']
        self.api_secret = bytes(self.config['api_secret'], 'utf-8')

    @staticmethod
    def read_config(file_name):
        with open(file_name, 'r') as f:
            return json.load(f)

    def place_order(self, order_type, currency_pair, amount, price=None):
        timestamp = str(int(round(time.time() * 1000)))
        nonce = str(uuid.uuid4())
        content_type = 'application/x-www-form-urlencoded'
        
        payload = {'amount': str(amount)}
        if price:
            payload['price'] = str(price)
        
        endpoint = f"/api/v2/{'buy' if 'buy' in order_type else 'sell'}/{'market/' if 'market' in order_type else ''}{currency_pair}/"
        
        message = f"BITSTAMP {self.api_key}POSTwww.bitstamp.net{endpoint}{content_type}{nonce}{timestamp}v2{urlencode(payload)}"
        signature = hmac.new(self.api_secret, msg=message.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()
        
        headers = {
            'X-Auth': f'BITSTAMP {self.api_key}',
            'X-Auth-Signature': signature,
            'X-Auth-Nonce': nonce,
            'X-Auth-Timestamp': timestamp,
            'X-Auth-Version': 'v2',
            'Content-Type': content_type
        }
        
        url = f"https://www.bitstamp.net{endpoint}"
        r = requests.post(url, headers=headers, data=urlencode(payload))
        
        if r.status_code == 200:
            return json.loads(r.content.decode('utf-8'))
        else:
            return f"Error: {r.status_code} - {r.text}"

class CryptoShell(cmd.Cmd):
    intro = 'Welcome to the Crypto Shell. Type help or ? to list commands.\n'
    prompt = '(crypto) '

    def __init__(self, data_manager, order_placer):
        super().__init__()
        self.data_manager = data_manager
        self.order_placer = order_placer
        self.candlestick_output = {}

    def do_price(self, arg):
        """Show current price for a symbol: price <symbol>"""
        symbol = arg.strip().lower()
        price = self.data_manager.get_current_price(symbol)
        if price:
            print(f"Current price of {symbol}: ${price:.2f}")
        else:
            print(f"No data available for {symbol}")

    def do_range(self, arg):
        """Show price range for last n minutes: range <symbol> <minutes>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: range <symbol> <minutes>")
            return
        symbol, minutes = args[0].lower(), int(args[1])
        min_price, max_price = self.data_manager.get_price_range(symbol, minutes)
        if min_price and max_price:
            print(f"Price range for {symbol} in last {minutes} minutes:")
            print(f"Min: ${min_price:.2f}, Max: ${max_price:.2f}")
        else:
            print(f"No data available for {symbol} in the last {minutes} minutes")

    def do_buy(self, arg):
        """Place a market buy order: buy <symbol> <amount>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: buy <symbol> <amount>")
            return
        symbol, amount = args[0].lower(), float(args[1])
        result = self.order_placer.place_order("market-buy", symbol, amount)
        print(json.dumps(result, indent=2))

    def do_sell(self, arg):
        """Place a market sell order: sell <symbol> <amount>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: sell <symbol> <amount>")
            return
        symbol, amount = args[0].lower(), float(args[1])
        result = self.order_placer.place_order("market-sell", symbol, amount)
        print(json.dumps(result, indent=2))

    def do_candles(self, arg):
        """Toggle 1-minute candlestick output for a symbol: candles <symbol>"""
        symbol = arg.strip().lower()
        if symbol in self.candlestick_output:
            del self.candlestick_output[symbol]
            print(f"Stopped 1-minute candlestick output for {symbol}")
        else:
            self.candlestick_output[symbol] = True
            print(f"Started 1-minute candlestick output for {symbol}")

    def candlestick_callback(self, symbol, minute, candle):
        if symbol in self.candlestick_output:
            timestamp = datetime.fromtimestamp(minute).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{symbol} - {timestamp}: Open: ${candle.open:.2f}, High: ${candle.high:.2f}, "
                  f"Low: ${candle.low:.2f}, Close: ${candle.close:.2f}, "
                  f"Volume: {candle.volume}, Trades: {candle.trades}")

    def do_quit(self, arg):
        """Quit the program"""
        print("Quitting...")
        return True

def run_websocket(url, symbols, data_manager):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [subscribe_to_websocket(url, symbol, data_manager) for symbol in symbols]
    loop.run_until_complete(asyncio.gather(*tasks))

def main():
    symbols = ["btcusd", "ethusd"]  # Add more symbols as needed
    data_manager = CryptoDataManager(symbols)
    order_placer = OrderPlacer()

    shell = CryptoShell(data_manager, order_placer)
    data_manager.add_observer(shell.candlestick_callback)

    # Start WebSocket connections in a separate thread
    url = 'wss://ws.bitstamp.net'
    websocket_thread = threading.Thread(target=run_websocket, args=(url, symbols, data_manager), daemon=True)
    websocket_thread.start()

    # Start the shell
    shell.cmdloop()

if __name__ == '__main__':
    main()