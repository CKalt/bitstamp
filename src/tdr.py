#!env/bin/python
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
    def __init__(self, symbols, verbose=False):
        self.data = {symbol: deque(maxlen=3600)
                     for symbol in symbols}  # Store last hour of data
        # Store current candlestick data
        self.candlesticks = {symbol: {} for symbol in symbols}
        self.observers = []  # List of functions to call when a candlestick is completed
        self.verbose = verbose
        self.last_candle_time = {symbol: None for symbol in symbols}
        self.last_price = {symbol: None for symbol in symbols}

    def add_trade(self, symbol, price, timestamp):
        price = float(price)
        self.data[symbol].append((timestamp, price))
        self.last_price[symbol] = price

        # Update candlestick data
        # Round down to the nearest minute
        minute = timestamp - (timestamp % 60)
        if minute not in self.candlesticks[symbol]:
            self.candlesticks[symbol][minute] = CandlestickData()

        candle = self.candlesticks[symbol][minute]
        if candle.open is None:
            candle.open = price
        candle.high = max(candle.high or price, price)
        candle.low = min(candle.low or price, price)
        candle.close = price
        candle.volume += 1
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
            self.last_candle_time[symbol] = minute

    def force_update_candlesticks(self):
        current_time = int(time.time())
        current_minute = current_time - (current_time % 60)
        for symbol in self.candlesticks:
            if self.last_candle_time[symbol] is None or current_minute > self.last_candle_time[symbol] + 60:
                last_complete_minute = self.last_candle_time[symbol] if self.last_candle_time[
                    symbol] is not None else current_minute - 120
                for minute in range(last_complete_minute + 60, current_minute, 60):
                    if minute not in self.candlesticks[symbol]:
                        # Create a zero-trade candlestick
                        zero_candle = CandlestickData()
                        if self.last_price[symbol] is not None:
                            zero_candle.open = zero_candle.high = zero_candle.low = zero_candle.close = self.last_price[
                                symbol]
                        self._complete_candlestick(symbol, minute)
                    else:
                        self._complete_candlestick(symbol, minute)

    def add_observer(self, callback):
        self.observers.append(callback)

    def get_current_price(self, symbol):
        if self.data[symbol]:
            return self.data[symbol][-1][1]
        return None

    def get_price_range(self, symbol, minutes):
        now = time.time()
        relevant_data = [price for ts,
                         price in self.data[symbol] if now - ts <= minutes * 60]
        if relevant_data:
            return min(relevant_data), max(relevant_data)
        return None, None

    def force_update_candlesticks(self):
        current_time = int(time.time())
        current_minute = current_time - (current_time % 60)
        for symbol in self.candlesticks:
            if self.last_candle_time[symbol] is None or current_minute > self.last_candle_time[symbol] + 60:
                if current_minute - 60 in self.candlesticks[symbol]:
                    self._complete_candlestick(symbol, current_minute - 60)
                elif self.candlesticks[symbol]:
                    last_candle_minute = max(self.candlesticks[symbol].keys())
                    self._complete_candlestick(symbol, last_candle_minute)


async def subscribe_to_websocket(url: str, symbol: str, data_manager):
    channel = f"live_trades_{symbol}"

    while True:  # Keep trying to reconnect
        try:
            async with websockets.connect(url) as websocket:
                if data_manager.verbose:
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
                    if data_manager.verbose:
                        print(f"{symbol}: {message}")
                    data = json.loads(message)
                    if data['event'] == 'trade':
                        price = data['data']['price']
                        timestamp = int(data['data']['timestamp'])
                        data_manager.add_trade(symbol, price, timestamp)

        except websockets.ConnectionClosed:
            if data_manager.verbose:
                print(
                    f"{symbol}: Connection closed, trying to reconnect in 5 seconds...")
            # Wait for 5 seconds before trying to reconnect
            await asyncio.sleep(5)
        except Exception as e:
            if data_manager.verbose:
                print(f"{symbol}: An error occurred: {str(e)}")
            # Wait for 5 seconds before trying to reconnect
            await asyncio.sleep(5)


class OrderPlacer:
    def __init__(self, config_file='.bitstamp'):
        self.config = self.read_config(config_file)
        self.api_key = self.config['api_key']
        self.api_secret = bytes(self.config['api_secret'], 'utf-8')

    @staticmethod
    def read_config(file_name):
        with open(file_name, 'r') as f:
            return json.load(f)

    def place_order(self, order_type, currency_pair, amount, price=None, **kwargs):
        timestamp = str(int(round(time.time() * 1000)))
        nonce = str(uuid.uuid4())
        content_type = 'application/x-www-form-urlencoded'

        payload = {'amount': str(amount)}
        if price:
            payload['price'] = str(price)
        
        # Add additional parameters for limit orders
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = str(value).lower() if isinstance(value, bool) else str(value)

        if 'market' in order_type:
            endpoint = f"/api/v2/{'buy' if 'buy' in order_type else 'sell'}/market/{currency_pair}/"
        else:
            endpoint = f"/api/v2/{'buy' if 'buy' in order_type else 'sell'}/{currency_pair}/"

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

    def place_limit_buy_order(self, currency_pair, amount, price, **kwargs):
        return self.place_order('buy', currency_pair, amount, price, **kwargs)

    def place_limit_sell_order(self, currency_pair, amount, price, **kwargs):
        return self.place_order('sell', currency_pair, amount, price, **kwargs)


class CryptoShell(cmd.Cmd):
    intro = 'Welcome to the Crypto Shell. Type help or ? to list commands.\n'
    prompt = '(crypto) '

    def __init__(self, data_manager, order_placer, verbose=False):
        super().__init__()
        self.data_manager = data_manager
        self.order_placer = order_placer
        self.candlestick_output = {}
        self.verbose = verbose
        self.examples = {
            'price': 'price btcusd',
            'range': 'range btcusd 30',
            'buy': 'buy btcusd 0.001',
            'sell': 'sell btcusd 0.001',
            'candles': 'candles btcusd',
            'verbose': 'verbose on',
            'example': 'example price',
            'limit_buy': 'limit_buy btcusd 0.001 50000 daily_order=true',
            'limit_sell': 'limit_sell btcusd 0.001 60000 ioc_order=true'
        }

    def do_example(self, arg):
        """Show an example of how to use a command: example <command>"""
        command = arg.strip().lower()
        if command in self.examples:
            print(f"Example usage of '{command}':")
            print(f"  {self.examples[command]}")
        else:
            print(
                f"No example available for '{command}'. Available commands are:")
            print(", ".join(self.examples.keys()))

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
        min_price, max_price = self.data_manager.get_price_range(
            symbol, minutes)
        if min_price and max_price:
            print(f"Price range for {symbol} in last {minutes} minutes:")
            print(f"Min: ${min_price:.2f}, Max: ${max_price:.2f}")
        else:
            print(
                f"No data available for {symbol} in the last {minutes} minutes")

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
            timestamp = datetime.fromtimestamp(
                minute).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{symbol} - {timestamp}: Open: ${candle.open:.2f}, High: ${candle.high:.2f}, "
                  f"Low: ${candle.low:.2f}, Close: ${candle.close:.2f}, "
                  f"Volume: {candle.volume}, Trades: {candle.trades}")

    def do_verbose(self, arg):
        """Toggle verbose mode: verbose [on|off]"""
        if arg.lower() in ('on', 'true', '1'):
            self.verbose = True
            self.data_manager.verbose = True
            print("Verbose mode turned on")
        elif arg.lower() in ('off', 'false', '0'):
            self.verbose = False
            self.data_manager.verbose = False
            print("Verbose mode turned off")
        else:
            print(
                f"Current verbose setting: {'on' if self.verbose else 'off'}")

    def do_limit_buy(self, arg):
        """Place a limit buy order: limit_buy <symbol> <amount> <price> [options]"""
        args = arg.split()
        if len(args) < 3:
            print("Usage: limit_buy <symbol> <amount> <price> [options]")
            return
        symbol, amount, price = args[0].lower(), float(args[1]), float(args[2])
        options = self.parse_order_options(args[3:])
        result = self.order_placer.place_limit_buy_order(
            symbol, amount, price, **options)
        print(json.dumps(result, indent=2))

    def do_limit_sell(self, arg):
        """Place a limit sell order: limit_sell <symbol> <amount> <price> [options]"""
        args = arg.split()
        if len(args) < 3:
            print("Usage: limit_sell <symbol> <amount> <price> [options]")
            return
        symbol, amount, price = args[0].lower(), float(args[1]), float(args[2])
        options = self.parse_order_options(args[3:])
        result = self.order_placer.place_limit_sell_order(
            symbol, amount, price, **options)
        print(json.dumps(result, indent=2))

    def parse_order_options(self, args):
        options = {}
        for arg in args:
            if '=' in arg:
                key, value = arg.split('=')
                if key in ['daily_order', 'ioc_order', 'fok_order', 'moc_order', 'gtd_order']:
                    options[key] = value.lower() == 'true'
                elif key == 'expire_time':
                    options[key] = int(value)
                elif key == 'client_order_id':
                    options[key] = value
                elif key == 'limit_price':
                    options[key] = float(value)
        return options

    def do_quit(self, arg):
        """Quit the program"""
        print("Quitting...")
        return True


def run_websocket(url, symbols, data_manager):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [subscribe_to_websocket(url, symbol, data_manager)
             for symbol in symbols]
    loop.run_until_complete(asyncio.gather(*tasks))


def candlestick_timer(data_manager):
    while True:
        time.sleep(60)  # Wait for 60 seconds
        data_manager.force_update_candlesticks()


def main():
    parser = argparse.ArgumentParser(description="Crypto trading shell")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Enable verbose output")
    args = parser.parse_args()

    symbols = ["btcusd", "ethusd"]  # Add more symbols as needed
    data_manager = CryptoDataManager(symbols, verbose=args.verbose)
    order_placer = OrderPlacer()

    shell = CryptoShell(data_manager, order_placer, verbose=args.verbose)
    data_manager.add_observer(shell.candlestick_callback)

    # Start WebSocket connections in a separate thread
    url = 'wss://ws.bitstamp.net'
    websocket_thread = threading.Thread(
        target=run_websocket, args=(url, symbols, data_manager), daemon=True)
    websocket_thread.start()

    # Start candlestick timer in a separate thread
    timer_thread = threading.Thread(
        target=candlestick_timer, args=(data_manager,), daemon=True)
    timer_thread.start()

    # Start the shell
    shell.cmdloop()


if __name__ == '__main__':
    main()