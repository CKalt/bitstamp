#!/usr/bin/env python
# src/tdr.py

import sys
import os
import pandas as pd
import numpy as np
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
import cmd
import threading
from datetime import datetime, timedelta
import logging
import tempfile
import csv

# Adjust sys.path to import modules from 'src' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

# Import necessary functions and modules from technical_indicators.py and other dependencies
from indicators.technical_indicators import (
    ensure_datetime_index,
    add_moving_averages,
    generate_ma_signals,
    calculate_rsi,
    generate_rsi_signals,
    calculate_bollinger_bands,
    generate_bollinger_band_signals,
    calculate_macd,
    generate_macd_signals,
)
# Assuming there's a helper function print_strategy_results in utils.helpers
# from utils.helpers import print_strategy_results  # Adjust if the path differs

# Import the same settings used in bktst.py
HIGH_FREQUENCY = '1H'  # High-frequency resampling used in backtesting

def parse_log_file(file_path, start_date=None, end_date=None):
    """
    Parses a JSON Lines log file and returns a DataFrame.
    Each line in the log file is a JSON object containing trade data.
    Writes valid records to a temporary CSV file to handle large files efficiently.

    Parameters:
        file_path (str): Path to the log file.
        start_date (datetime, optional): Start date to filter data.
        end_date (datetime, optional): End date to filter data.

    Returns:
        pd.DataFrame: DataFrame with 'timestamp', 'price', 'amount' columns.
    """

    total_lines = 0
    valid_lines = 0

    # First, count total lines for progress feedback
    with open(file_path, 'r') as f:
        for _ in f:
            total_lines += 1

    print(f"Total lines to parse: {total_lines}")

    # Create a temporary file to store valid records
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='', suffix='.csv')
    temp_file_name = temp_file.name
    temp_file.close()  # We'll reopen it for writing

    with open(file_path, 'r') as f_in, open(temp_file_name, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['timestamp', 'price', 'amount'])  # Write header

        for idx, line in enumerate(f_in, 1):
            try:
                obj = json.loads(line.strip())
                if not isinstance(obj, dict):
                    continue

                # Ensure 'data' field exists and is a dict
                data = obj.get('data')
                if not isinstance(data, dict):
                    continue

                # Extract 'timestamp', 'price', and 'amount' from 'data'
                if 'timestamp' in data and 'price' in data and 'amount' in data:
                    timestamp = int(data['timestamp'])
                    price = float(data['price'])
                    amount = float(data['amount'])

                    # Filter by date range if specified
                    if start_date and timestamp < int(start_date.timestamp()):
                        continue
                    if end_date and timestamp > int(end_date.timestamp()):
                        continue

                    writer.writerow([timestamp, price, amount])
                    valid_lines += 1
                else:
                    continue
            except json.JSONDecodeError:
                continue
            except (ValueError, TypeError) as e:
                continue

            # Progress feedback every 10%
            if total_lines >= 10:
                if idx % (total_lines // 10) == 0:
                    progress = (idx / total_lines) * 100
                    print(f"Parsing log file: {progress:.0f}% completed.")

    print(f"Finished parsing log file. Total lines: {total_lines}, Valid trades: {valid_lines}")

    # Read the temporary CSV file into a DataFrame
    df = pd.read_csv(temp_file_name)

    # Delete the temporary file
    os.remove(temp_file_name)

    return df


class CryptoDataManager:
    def __init__(self, symbols, logger, verbose=False):
        self.data = {symbol: pd.DataFrame(columns=['timestamp', 'price', 'amount']) for symbol in symbols}
        self.candlesticks = {symbol: {} for symbol in symbols}
        self.candlestick_observers = []
        self.trade_observers = []
        self.logger = logger
        self.verbose = verbose
        self.last_price = {symbol: None for symbol in symbols}
        self.order_placer = None

    def load_historical_data(self, data_dict):
        total_symbols = len(data_dict)
        for idx, (symbol, df) in enumerate(data_dict.items(), 1):
            self.data[symbol] = df.reset_index(drop=True)
            if not df.empty:
                self.last_price[symbol] = df.iloc[-1]['price']
                self.logger.debug(f"Loaded historical data for {symbol}, last price: {self.last_price[symbol]}")
            print(f"Loaded historical data for {symbol} ({idx}/{total_symbols})")

    def add_candlestick_observer(self, callback):
        self.candlestick_observers.append(callback)

    def add_trade_observer(self, callback):
        self.trade_observers.append(callback)

    def set_verbose(self, verbose):
        self.verbose = verbose

    def add_trade(self, symbol, price, timestamp):
        price = float(price)
        # Ensure 'timestamp' is in UNIX epoch format (integer seconds)
        # Do NOT convert to datetime here; let technical_indicators.py handle it
        new_row = {'timestamp': timestamp, 'price': price, 'amount': 0}
        self.data[symbol] = pd.concat([self.data[symbol], pd.DataFrame([new_row])], ignore_index=True)
        self.last_price[symbol] = price

        # Notify trade observers
        for observer in self.trade_observers:
            observer(symbol, price, timestamp)

    def get_current_price(self, symbol):
        if not self.data[symbol].empty:
            return self.data[symbol].iloc[-1]['price']
        return None

    def get_price_range(self, symbol, minutes):
        now = pd.Timestamp.now()
        start_time = now - pd.Timedelta(minutes=minutes)
        df = self.data[symbol]
        mask = df['timestamp'] >= int(start_time.timestamp())
        relevant_data = df.loc[mask, 'price']
        if not relevant_data.empty:
            return relevant_data.min(), relevant_data.max()
        return None, None

    def get_price_dataframe(self, symbol):
        return self.data[symbol]

    def get_data_point_count(self, symbol):
        return len(self.data[symbol])

async def subscribe_to_websocket(url: str, symbol: str, data_manager):
    channel = f"live_trades_{symbol}"

    while True:  # Keep trying to reconnect
        try:
            data_manager.logger.info(f"{symbol}: Attempting to connect to WebSocket...")
            async with websockets.connect(url) as websocket:
                data_manager.logger.info(f"{symbol}: Connected to WebSocket.")

                # Subscribing to the channel.
                subscribe_message = {
                    "event": "bts:subscribe",
                    "data": {
                        "channel": channel
                    }
                }
                await websocket.send(json.dumps(subscribe_message))
                data_manager.logger.info(f"{symbol}: Subscribed to channel: {channel}")

                # Receiving messages.
                async for message in websocket:
                    data_manager.logger.debug(f"{symbol}: {message}")
                    data = json.loads(message)
                    if data.get('event') == 'trade':
                        price = data['data']['price']
                        timestamp = int(float(data['data']['timestamp']))
                        data_manager.add_trade(symbol, price, timestamp)

        except websockets.ConnectionClosed:
            data_manager.logger.error(f"{symbol}: Connection closed, trying to reconnect in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            data_manager.logger.error(f"{symbol}: An error occurred: {str(e)}")
            await asyncio.sleep(5)


class OrderPlacer:
    def __init__(self, config_file='.bitstamp'):
        self.config_file = config_file
        self.config = self.read_config(self.config_file)
        self.api_key = self.config['api_key']
        self.api_secret = bytes(self.config['api_secret'], 'utf-8')

    def read_config(self, file_name):
        # Use current working directory
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, file_name)
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to read config file '{file_name}': {e}")

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
        self.running = False
        self.live_trading = live_trading
        self.trade_log = []  # For dry run logging
        self.trade_log_file = 'trades.json'
        self.last_signal_time = None  # To prevent duplicate signals

    def start(self):
        self.running = True
        threading.Thread(target=self.run_strategy_loop, daemon=True).start()
        self.logger.info("Strategy loop started.")

    def stop(self):
        self.running = False
        self.logger.info("Strategy loop stopped.")
        # Save trades to trades.json if dry run
        if self.trade_log and not self.live_trading:
            try:
                # Use current working directory
                current_dir = os.getcwd()
                file_path = os.path.join(current_dir, self.trade_log_file)
                with open(file_path, 'w') as f:
                    json.dump(self.trade_log, f, indent=2)
                self.logger.info(f"Trades logged to '{file_path}'")
            except Exception as e:
                self.logger.error(f"Failed to write trades to '{file_path}': {e}")

    def run_strategy_loop(self):
        while self.running:
            df = self.data_manager.get_price_dataframe(self.symbol)
            if not df.empty:
                try:
                    # Ensure datetime index
                    df = ensure_datetime_index(df)
                    # Resample to high frequency used in backtesting
                    df_resampled = df.resample(HIGH_FREQUENCY).agg({
                        'price': 'last',
                        'amount': 'sum',
                        'timestamp': 'last'  # Retain the latest timestamp in the resampled period
                    }).dropna()
                    if len(df_resampled) >= self.long_window:
                        # Calculate moving averages
                        df_ma = add_moving_averages(df_resampled.copy(), self.short_window, self.long_window)
                        # Generate MA signals
                        df_ma = generate_ma_signals(df_ma)
                        # Get the latest signal
                        latest_signal = df_ma.iloc[-1]['MA_Signal']
                        signal_time = df_ma.index[-1]
                        current_price = df_ma.iloc[-1]['price']
                        self.check_for_signals(latest_signal, current_price, signal_time)
                except KeyError as e:
                    self.logger.error(f"Missing column during strategy loop: {e}")
                except Exception as e:
                    self.logger.error(f"Error during strategy loop: {e}")
            else:
                self.logger.debug(f"No data available for {self.symbol} to run strategy.")
            time.sleep(60)  # Check every minute

    def check_for_signals(self, latest_signal, current_price, signal_time):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if self.last_signal_time == signal_time:
            # Signal already processed
            return

        if latest_signal == 1 and self.position <= 0:
            # Buy signal
            self.logger.info(f"Buy signal at price {current_price}")
            self.position = 1
            self.execute_trade("buy", current_price, timestamp)
            self.last_signal_time = signal_time
        elif latest_signal == -1 and self.position >= 0:
            # Sell signal
            self.logger.info(f"Sell signal at price {current_price}")
            self.position = -1
            self.execute_trade("sell", current_price, timestamp)
            self.last_signal_time = signal_time

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
            # Use current working directory
            current_dir = os.getcwd()
            file_path = os.path.join(current_dir, self.trade_log_file)
            with open(file_path, 'a') as f:
                f.write(json.dumps(trade_info) + '\n')
            self.logger.debug(f"Trade info written to '{file_path}'")
        except Exception as e:
            self.logger.error(f"Failed to write trade to log file: {e}")

    def get_status(self):
        """Return the current status of the strategy."""
        status = {
            'running': self.running,
            'position': self.position,
        }
        return status


class CryptoShell(cmd.Cmd):
    intro = 'Welcome to the Crypto Shell. Type help or ? to list commands.\n'
    prompt = '(crypto) '

    def __init__(self, data_manager, order_placer, logger, verbose=False, live_trading=False):
        super().__init__()
        self.data_manager = data_manager
        self.order_placer = order_placer
        self.data_manager.order_placer = order_placer  # Set order placer in data manager
        self.logger = logger
        self.candlestick_output = {}
        self.ticker_output = {}
        self.verbose = verbose
        self.live_trading = live_trading
        self.auto_trader = None  # Initialize to None
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
            'stop_auto_trade': 'stop_auto_trade',
            'status': 'status'
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
            timestamp = minute.strftime('%Y-%m-%d %H:%M:%S')
            print(f"{symbol} - {timestamp}: Open: ${candle['open']:.2f}, High: ${candle['high']:.2f}, "
                  f"Low: ${candle['low']:.2f}, Close: ${candle['close']:.2f}, "
                  f"Volume: {candle['volume']}, Trades: {candle['trades']}")

    def trade_callback(self, symbol, price, timestamp):
        if symbol in self.ticker_output:
            # Convert UNIX timestamp to readable format
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
                # Use current working directory
                current_dir = os.getcwd()
                log_file_path = os.path.join(current_dir, log_file)
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                self.data_manager.set_verbose(True)
                self.verbose = True
                print(f"Verbose mode enabled. Logs are being written to {log_file_path}.")
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
        try:
            amount = float(args[0])
        except ValueError:
            print("Amount must be a number.")
            return

        # Use current working directory
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, 'best_strategy.json')

        try:
            with open(file_path, 'r') as f:
                best_strategy_params = json.load(f)
        except FileNotFoundError:
            print(f"Best strategy parameters file '{file_path}' not found.")
            return
        except json.JSONDecodeError:
            print(f"Best strategy parameters file '{file_path}' is not a valid JSON.")
            return

        strategy_name = best_strategy_params.get('Strategy')
        if strategy_name != 'MA':
            print(f"The best strategy is not MA Crossover. It's {strategy_name}.")
            return

        try:
            short_window = int(best_strategy_params['Short_Window'])
            long_window = int(best_strategy_params['Long_Window'])
        except (KeyError, ValueError) as e:
            print(f"Invalid strategy parameters: {e}")
            return

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
        if self.auto_trader is not None and self.auto_trader.running:
            self.auto_trader.stop()
            print("Auto-trading stopped.")
        else:
            print("Auto-trading is not running.")

    def do_status(self, arg):
        """Show the status of auto-trading."""
        if self.auto_trader is not None and self.auto_trader.running:
            status = self.auto_trader.get_status()
            position = {1: 'Long', -1: 'Short', 0: 'Neutral'}.get(status['position'], 'Unknown')
            print("Auto-Trading Status:")
            print(f"  Running: {status['running']}")
            print(f"  Position: {position}")
        else:
            print("Auto-trading is not running.")

    def do_help(self, arg):
        """List available commands with "help" or detailed help with "help cmd"."""
        super().do_help(arg)
        if arg == '':
            print("\nCustom Commands:")
            print("  auto_trade       Start auto-trading using the best strategy")
            print("  stop_auto_trade  Stop auto-trading")
            print("  status           Show the status of auto-trading")

    def do_quit(self, arg):
        """Quit the program"""
        print("Quitting...")
        if self.auto_trader is not None and self.auto_trader.running:
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
    # Use current working directory
    current_dir = os.getcwd()
    if log_file:
        log_file_path = os.path.join(current_dir, log_file)
    else:
        log_file_path = os.path.join(current_dir, 'crypto_shell.log')
    file_handler = logging.FileHandler(log_file_path)
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
        logger = setup_logging(verbose=True)
        verbose_flag = True
    elif isinstance(args.verbose, str):
        logger = setup_logging(verbose=True, log_file=args.verbose)
        verbose_flag = True
    else:
        logger = setup_logging(verbose=False)
        verbose_flag = False

    live_trading = args.do_live_trades

    if live_trading:
        print("Live trading is ENABLED.")
    else:
        print("Live trading is DISABLED. Running in dry run mode.")

    symbols = ["btcusd"]  # Adjust as needed
    data_manager = CryptoDataManager(symbols, logger=logger, verbose=verbose_flag)

    # Read historical data
    file_path = 'btcusd.log'  # Ensure this is the correct path to your log file
    start_date = None  # Adjust as needed
    end_date = datetime.now()

    # Parse historical data using the updated parse_log_file function
    try:
        df = parse_log_file(file_path, start_date, end_date)
    except Exception as e:
        print(f"Failed to parse log file '{file_path}': {e}")
        sys.exit(1)

    if df.empty:
        print(f"No historical data found in '{file_path}'. Exiting.")
        sys.exit(1)

    try:
        # Retain 'timestamp' as UNIX epoch seconds
        if 'timestamp' not in df.columns:
            print(f"Missing 'timestamp' column in log file '{file_path}'. Exiting.")
            sys.exit(1)
        # Ensure 'timestamp' is integer seconds
        df['timestamp'] = df['timestamp'].astype(int)
        # Do NOT convert 'timestamp' to datetime here
        # Let technical_indicators.py handle datetime indexing
        # If 'datetime' column exists, drop it to avoid confusion
        if 'datetime' in df.columns:
            df = df.drop(columns=['datetime'])
    except KeyError as e:
        print(f"Missing expected column in log file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing log file: {e}")
        sys.exit(1)

    # Load historical data into the data manager
    data_manager.load_historical_data({'btcusd': df})

    order_placer = OrderPlacer()
    data_manager.order_placer = order_placer  # Set order placer in data manager

    shell = CryptoShell(data_manager, order_placer, logger=logger, verbose=verbose_flag, live_trading=live_trading)

    # Start WebSocket connections in a separate thread
    url = 'wss://ws.bitstamp.net'
    websocket_thread = threading.Thread(
        target=run_websocket, args=(url, symbols, data_manager), daemon=True)
    websocket_thread.start()
    logger.debug("WebSocket thread started.")

    # Start the shell
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting...")
        if shell.auto_trader is not None and shell.auto_trader.running:
            shell.auto_trader.stop()


if __name__ == '__main__':
    main()
