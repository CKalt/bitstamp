#!/env/bin/python
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
import logging
import sys
import os

class CandlestickData:
    def __init__(self):
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = 0
        self.trades = 0

class CryptoDataManager:
    def __init__(self, symbols, logger, verbose=False):
        self.data = {symbol: deque(maxlen=3600) for symbol in symbols}  # Store last hour of data
        self.candlesticks = {symbol: {} for symbol in symbols}
        self.candlestick_observers = []  # Observers for candlestick completion
        self.trade_observers = []        # Observers for each trade
        self.logger = logger
        self.verbose = verbose
        self.last_candle_time = {symbol: None for symbol in symbols}
        self.last_price = {symbol: None for symbol in symbols}
        self.order_placer = None  # Will be set later

    def add_candlestick_observer(self, callback):
        self.candlestick_observers.append(callback)

    def add_trade_observer(self, callback):
        self.trade_observers.append(callback)

    def set_verbose(self, verbose):
        self.verbose = verbose

    def add_trade(self, symbol, price, timestamp):
        price = float(price)
        self.data[symbol].append((timestamp, price))
        self.last_price[symbol] = price

        # Update candlestick data
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

        # Notify trade observers
        for observer in self.trade_observers:
            observer(symbol, price, timestamp)

        # Check if the candlestick is complete
        current_time = int(time.time())
        if current_time >= minute + 60:
            self._complete_candlestick(symbol, minute)

    def _complete_candlestick(self, symbol, minute):
        candle = self.candlesticks[symbol].pop(minute, None)
        if candle:
            readable_time = datetime.fromtimestamp(minute).strftime('%Y-%m-%d %H:%M:%S')
            if self.verbose:
                self.logger.debug(f"Completing candlestick for {symbol} at {readable_time}")
            for observer in self.candlestick_observers:
                observer(symbol, minute, candle)
            self.last_candle_time[symbol] = minute

    def force_update_candlesticks(self):
        current_time = int(time.time())
        current_minute = current_time - (current_time % 60)
        for symbol in self.candlesticks:
            if self.last_candle_time[symbol] is None or current_minute > self.last_candle_time[symbol] + 60:
                last_complete_minute = self.last_candle_time[symbol] if self.last_candle_time[symbol] is not None else current_minute - 120
                for minute in range(last_complete_minute + 60, current_minute, 60):
                    if minute not in self.candlesticks[symbol]:
                        # Create a zero-trade candlestick
                        zero_candle = CandlestickData()
                        if self.last_price[symbol] is not None:
                            zero_candle.open = zero_candle.high = zero_candle.low = zero_candle.close = self.last_price[symbol]
                        self._complete_candlestick(symbol, minute)
                    else:
                        self._complete_candlestick(symbol, minute)

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

    def get_price_series(self, symbol):
        return [price for timestamp, price in self.data[symbol]]

async def subscribe_to_websocket(url: str, symbol: str, data_manager):
    channel = f"live_trades_{symbol}"

    while True:  # Keep trying to reconnect
        try:
            data_manager.logger.debug(f"Attempting to connect to WebSocket for {symbol}...")
            async with websockets.connect(url) as websocket:
                data_manager.logger.debug(f"Connected to WebSocket for {symbol}")

                # Subscribing to the channel.
                subscribe_message = {
                    "event": "bts:subscribe",
                    "data": {
                        "channel": channel
                    }
                }
                await websocket.send(json.dumps(subscribe_message))
                data_manager.logger.debug(f"Subscribed to channel: {channel}")

                # Receiving messages.
                async for message in websocket:
                    data_manager.logger.debug(f"{symbol}: {message}")
                    data = json.loads(message)
                    if data.get('event') == 'trade':
                        price = data['data']['price']
                        timestamp = int(float(data['data']['timestamp']))
                        # Adjust timestamp if necessary (e.g., if in milliseconds)
                        # timestamp = timestamp // 1000  # Uncomment if timestamp is in milliseconds
                        data_manager.add_trade(symbol, price, timestamp)

        except websockets.ConnectionClosed:
            data_manager.logger.error(f"{symbol}: Connection closed, trying to reconnect in 5 seconds...")
            # Wait for 5 seconds before trying to reconnect
            await asyncio.sleep(5)
        except Exception as e:
            data_manager.logger.error(f"{symbol}: An error occurred: {str(e)}")
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

class MACrossoverStrategy:
    def __init__(self, data_manager, short_window, long_window, amount, symbol, logger, live_trading=False):
        self.data_manager = data_manager
        self.order_placer = data_manager.order_placer
        self.short_window = short_window
        self.long_window = long_window
        self.amount = amount
        self.symbol = symbol
        self.logger = logger
        self.position = 0  # 1 for long, -1 for short, 0 for neutral
        self.prices = []
        self.running = False
        self.live_trading = live_trading
        self.trade_log = []  # For dry run logging
        self.trade_log_file = 'trades.json'  # Log file for all trades

    def start(self):
        self.running = True
        threading.Thread(target=self.run_strategy_loop, daemon=True).start()

    def stop(self):
        self.running = False
        # Save trades to trades.json if dry run
        if self.trade_log:
            try:
                with open(self.trade_log_file, 'w') as f:
                    json.dump(self.trade_log, f, indent=2)
                print(f"Trades logged to '{self.trade_log_file}'")
            except Exception as e:
                print(f"Failed to write trades to '{self.trade_log_file}': {e}")

    def run_strategy_loop(self):
        while self.running:
            price_series = self.data_manager.get_price_series(self.symbol)
            if price_series and len(price_series) >= self.long_window:
                self.prices = price_series
                self.check_for_signals()
            time.sleep(5)  # Adjust as needed

    def check_for_signals(self):
        short_ma = sum(self.prices[-self.short_window:]) / self.short_window
        long_ma = sum(self.prices[-self.long_window:]) / self.long_window
        current_price = self.prices[-1]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if short_ma > long_ma and self.position <= 0:
            # Buy signal
            self.logger.info(f"Buy signal: short MA ({short_ma}) > long MA ({long_ma}) at price {current_price}")
            self.position = 1
            self.execute_trade("buy", current_price, timestamp)
        elif short_ma < long_ma and self.position >= 0:
            # Sell signal
            self.logger.info(f"Sell signal: short MA ({short_ma}) < long MA ({long_ma}) at price {current_price}")
            self.position = -1
            self.execute_trade("sell", current_price, timestamp)

    def execute_trade(self, trade_type, price, timestamp):
        trade_info = {
            'type': trade_type,
            'symbol': self.symbol,
            'amount': self.amount,
            'price': price,
            'timestamp': timestamp,
            'live_trading': self.live_trading
        }
        if self.live_trading:
            result = self.order_placer.place_order(
                f"market-{trade_type}", self.symbol, self.amount)
            self.logger.info(f"Executed live {trade_type} order: {result}")
            trade_info['order_result'] = result
        else:
            self.logger.info(f"Executed dry run {trade_type} order: {trade_info}")
            self.trade_log.append(trade_info)
        # Write the trade_info to the trade log file
        try:
            with open(self.trade_log_file, 'a') as f:
                f.write(json.dumps(trade_info) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write trade to log file: {e}")

class CryptoShell(cmd.Cmd):
    intro = 'Welcome to the Crypto Shell. Type help or ? to list commands.\n'
    prompt = '(crypto) '

    def __init__(self, data_manager, order_placer, logger, verbose=False, live_trading=False):
        super().__init__()
        self.data_manager = data_manager
        self.order_placer = order_placer
        self.data_manager.order_placer = order_placer  # Set in data manager
        self.logger = logger
        self.candlestick_output = {}
        self.ticker_output = {}
        self.verbose = verbose
        self.live_trading = live_trading
        self.auto_trader = None
        self.examples = {
            'price': 'price btcusd',
            'range': 'range btcusd 30',
            'buy': 'buy btcusd 0.001',
            'sell': 'sell btcusd 0.001',
            'candles': 'candles btcusd',
            'ticker': 'ticker btcusd',
            'example': 'example price',
            'limit_buy': 'limit_buy btcusd 0.001 50000 daily_order=true',
            'limit_sell': 'limit_sell btcusd 0.001 60000 ioc_order=true',
            'auto_trade': 'auto_trade 2',
            'stop_auto_trade': 'stop_auto_trade'
        }

        # Register callbacks
        self.data_manager.add_candlestick_observer(self.candlestick_callback)
        self.data_manager.add_trade_observer(self.trade_callback)

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
        if not symbol:
            print("Usage: price <symbol>")
            return
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
        if min_price is not None and max_price is not None:
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
        if not symbol:
            print("Usage: candles <symbol>")
            return
        if symbol in self.candlestick_output:
            del self.candlestick_output[symbol]
            print(f"Stopped 1-minute candlestick output for {symbol}")
        else:
            self.candlestick_output[symbol] = True
            print(f"Started 1-minute candlestick output for {symbol}")

    def do_ticker(self, arg):
        """Toggle real-time trade output for a symbol: ticker <symbol>"""
        symbol = arg.strip().lower()
        if not symbol:
            print("Usage: ticker <symbol>")
            return
        if symbol in self.ticker_output:
            del self.ticker_output[symbol]
            print(f"Stopped real-time trade output for {symbol}")
        else:
            self.ticker_output[symbol] = True
            print(f"Started real-time trade output for {symbol}")

    def candlestick_callback(self, symbol, minute, candle):
        if symbol in self.candlestick_output:
            timestamp = datetime.fromtimestamp(minute).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{symbol} - {timestamp}: Open: ${candle.open:.2f}, High: ${candle.high:.2f}, "
                  f"Low: ${candle.low:.2f}, Close: ${candle.close:.2f}, "
                  f"Volume: {candle.volume}, Trades: {candle.trades}")

    def trade_callback(self, symbol, price, timestamp):
        if symbol in self.ticker_output:
            time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{symbol} - {time_str}: Price: ${price:.2f}")

    def do_verbose(self, arg):
        """Enable verbose mode and optionally specify a log file: verbose [logfile]"""
        arg = arg.strip()
        if not arg:
            # If no logfile is provided, default to stderr and set DEBUG level
            if not self.verbose:
                self.logger.setLevel(logging.DEBUG)
                # Check if a DEBUG StreamHandler is already present to avoid duplicates
                debug_handlers = [
                    h for h in self.logger.handlers
                    if isinstance(h, logging.StreamHandler) and h.level == logging.DEBUG
                ]
                if not debug_handlers:
                    debug_stream_handler = logging.StreamHandler(sys.stderr)
                    debug_stream_handler.setLevel(logging.DEBUG)
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    debug_stream_handler.setFormatter(formatter)
                    self.logger.addHandler(debug_stream_handler)
                self.data_manager.set_verbose(True)
                self.verbose = True
                print("Verbose mode enabled. Logs are being printed to stderr.")
            else:
                print("Verbose mode is already enabled.")
        else:
            log_file = arg
            # Remove existing FileHandlers to prevent duplicates
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    self.logger.removeHandler(handler)
            # Add FileHandler for verbose logs
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                self.data_manager.set_verbose(True)
                self.verbose = True
                print(f"Verbose mode enabled. Logs are being written to {log_file}.")
            except Exception as e:
                print(f"Failed to open log file {log_file}: {e}")

    def do_limit_buy(self, arg):
        """Place a limit buy order: limit_buy <symbol> <amount> <price> [options]"""
        args = arg.split()
        if len(args) < 3:
            print("Usage: limit_buy <symbol> <amount> <price> [options]")
            return
        symbol, amount, price = args[0].lower(), float(args[1]), float(args[2])
        options = self.parse_order_options(args[3:])
        result = self.order_placer.place_limit_buy_order(symbol, amount, price, **options)
        print(json.dumps(result, indent=2))

    def do_limit_sell(self, arg):
        """Place a limit sell order: limit_sell <symbol> <amount> <price> [options]"""
        args = arg.split()
        if len(args) < 3:
            print("Usage: limit_sell <symbol> <amount> <price> [options]")
            return
        symbol, amount, price = args[0].lower(), float(args[1]), float(args[2])
        options = self.parse_order_options(args[3:])
        result = self.order_placer.place_limit_sell_order(symbol, amount, price, **options)
        print(json.dumps(result, indent=2))

    def parse_order_options(self, args):
        options = {}
        for arg in args:
            if '=' in arg:
                key, value = arg.split('=', 1)
                if key in ['daily_order', 'ioc_order', 'fok_order', 'moc_order', 'gtd_order']:
                    options[key] = value.lower() == 'true'
                elif key == 'expire_time':
                    try:
                        options[key] = int(value)
                    except ValueError:
                        print(f"Invalid value for {key}: {value}. It should be an integer.")
                elif key == 'client_order_id':
                    options[key] = value
                elif key == 'limit_price':
                    try:
                        options[key] = float(value)
                    except ValueError:
                        print(f"Invalid value for {key}: {value}. It should be a float.")
        return options

    def do_auto_trade(self, arg):
        """Start auto-trading using the best strategy: auto_trade <amount>"""
        args = arg.split()
        if len(args) != 1:
            print("Usage: auto_trade <amount>")
            return
        amount = float(args[0])

        # Determine the file path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'best_strategy.json')

        try:
            with open(file_path, 'r') as f:
                best_strategy_params = json.load(f)
        except FileNotFoundError:
            print(f"Best strategy parameters file '{file_path}' not found.")
            return

        strategy_name = best_strategy_params.get('Strategy')
        if strategy_name != 'MA':
            print(f"The best strategy is not MA Crossover. It's {strategy_name}.")
            return

        short_window = int(best_strategy_params['Short_Window'])
        long_window = int(best_strategy_params['Long_Window'])
        symbol = 'btcusd'  # Adjust as needed

        self.auto_trader = MACrossoverStrategy(
            self.data_manager, short_window, long_window, amount, symbol, self.logger, live_trading=self.live_trading)
        self.auto_trader.start()
        print(f"Auto-trading started with amount {amount} using MA Crossover strategy.")
        print(f"Trades will be logged to '{self.auto_trader.trade_log_file}'.")
        if not self.live_trading:
            print("Running in dry run mode.")

    def do_stop_auto_trade(self, arg):
        """Stop auto-trading"""
        if hasattr(self, 'auto_trader') and self.auto_trader.running:
            self.auto_trader.stop()
            print("Auto-trading stopped.")
        else:
            print("Auto-trading is not running.")

    def do_help(self, arg):
        """List available commands with "help" or detailed help with "help cmd"."""
        super().do_help(arg)
        if arg == '':
            print("\nCustom Commands:")
            print("  auto_trade       Start auto-trading using the best strategy")
            print("  stop_auto_trade  Stop auto-trading")

    def do_quit(self, arg):
        """Quit the program"""
        print("Quitting...")
        if hasattr(self, 'auto_trader') and self.auto_trader.running:
            self.auto_trader.stop()
        return True

def run_websocket(url, symbols, data_manager):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [subscribe_to_websocket(url, symbol, data_manager) for symbol in symbols]
    try:
        loop.run_until_complete(asyncio.gather(*tasks))
    except Exception as e:
        data_manager.logger.error(f"WebSocket thread encountered an error: {e}")
    finally:
        loop.close()

def candlestick_timer(data_manager):
    while True:
        time.sleep(60)  # Wait for 60 seconds
        data_manager.logger.debug("[Timer] Triggering force_update_candlesticks()")
        data_manager.force_update_candlesticks()

def setup_logging(verbose, log_file=None):
    logger = logging.getLogger("CryptoShellLogger")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # StreamHandler for console output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler for logging to a file
    file_handler = logging.FileHandler('crypto_shell.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def main():
    parser = argparse.ArgumentParser(description="Crypto trading shell")
    parser.add_argument('-v', '--verbose', nargs='?', const=True, default=False,
                        help="Enable verbose output and optionally specify a log file (e.g., --verbose logfile.log)")
    parser.add_argument('--do-live-trades', action='store_true', help="Enable live trading (default is dry run)")
    args = parser.parse_args()

    # Setup logging based on verbose argument
    if args.verbose is True:
        # Verbose enabled without specifying a log file; log to stderr
        logger = setup_logging(verbose=True)
        verbose_flag = True
    elif isinstance(args.verbose, str):
        # Verbose enabled with a log file; log to the specified file
        logger = setup_logging(verbose=True, log_file=args.verbose)
        verbose_flag = True
    else:
        # Verbose disabled
        logger = setup_logging(verbose=False)
        verbose_flag = False

    live_trading = args.do_live_trades

    if live_trading:
        print("Live trading is ENABLED.")
    else:
        print("Live trading is DISABLED. Running in dry run mode.")

    symbols = ["btcusd", "ethusd"]  # Add more symbols as needed
    data_manager = CryptoDataManager(symbols, logger=logger, verbose=verbose_flag)
    order_placer = OrderPlacer()
    data_manager.order_placer = order_placer  # Set order placer in data manager

    shell = CryptoShell(data_manager, order_placer, logger=logger, verbose=verbose_flag, live_trading=live_trading)

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
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting...")
        if hasattr(shell, 'auto_trader') and shell.auto_trader.running:
            shell.auto_trader.stop()

if __name__ == '__main__':
    main()
