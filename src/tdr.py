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
import threading

# For the Plotly and Dash implementation
try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Output, Input
    import plotly.graph_objs as go
except ImportError:
    pass  # We will handle the ImportError in the do_chart method

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

# Import the same settings used in bktst.py
HIGH_FREQUENCY = '1H'  # High-frequency resampling used in backtesting


def parse_log_file(file_path, start_date=None, end_date=None):
    """
    Parses a JSON Lines log file and returns a DataFrame.
    Each line in the log file is a JSON object containing trade data.
    Aggregates trade data into minute-level candlesticks during parsing.

    Parameters:
        file_path (str): Path to the log file.
        start_date (datetime, optional): Start date to filter data.
        end_date (datetime, optional): End date to filter data.

    Returns:
        pd.DataFrame: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']
    """
    import json
    import pandas as pd
    import time
    from datetime import datetime

    # First, count total lines for progress feedback
    total_lines = 0
    with open(file_path, 'r') as f:
        for _ in f:
            total_lines += 1

    print("Total lines to parse: {}".format(total_lines))

    valid_lines = 0

    # Prepare progress thresholds
    progress_percentages = list(range(1, 101))  # [1, 2, ..., 100]
    progress_thresholds = [int(total_lines * p / 100)
                           for p in progress_percentages]
    next_progress_idx = 0

    # Time-based progress feedback
    last_feedback_time = time.time()
    feedback_interval = 5  # seconds

    # Create a dictionary to hold aggregated data
    minute_bars = {}

    with open(file_path, 'r') as f:
        for idx, line in enumerate(f, 1):
            try:
                obj = json.loads(line.strip())
                if not isinstance(obj, dict):
                    continue

                data = obj.get('data')
                if not isinstance(data, dict):
                    continue

                if 'timestamp' in data and 'price' in data and 'amount' in data:
                    timestamp = int(data['timestamp'])
                    price = float(data['price'])
                    amount = float(data['amount'])

                    # Filter by date range
                    if start_date and timestamp < int(start_date.timestamp()):
                        continue
                    if end_date and timestamp > int(end_date.timestamp()):
                        continue

                    dt = datetime.fromtimestamp(timestamp)
                    minute = dt.replace(second=0, microsecond=0)

                    if minute not in minute_bars:
                        minute_bars[minute] = {
                            'timestamp': int(minute.timestamp()),
                            'open': price,
                            'high': price,
                            'low': price,
                            'close': price,
                            'volume': amount,
                            'trades': 1
                        }
                    else:
                        bar = minute_bars[minute]
                        bar['high'] = max(bar['high'], price)
                        bar['low'] = min(bar['low'], price)
                        bar['close'] = price
                        bar['volume'] += amount
                        bar['trades'] += 1

                    valid_lines += 1
                else:
                    continue
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

            # Progress feedback based on line count
            if next_progress_idx < len(progress_thresholds) and idx >= progress_thresholds[next_progress_idx]:
                progress = progress_percentages[next_progress_idx]
                print("Parsing log file: {}% completed.".format(
                    progress), flush=True)
                next_progress_idx += 1

            # Time-based progress feedback every 'feedback_interval' seconds
            current_time = time.time()
            if current_time - last_feedback_time >= feedback_interval:
                percent_complete = (idx / total_lines) * 100
                print("Parsing log file: {:.2f}% completed.".format(
                    percent_complete), flush=True)
                last_feedback_time = current_time

    # Ensure 100% is printed
    if next_progress_idx < len(progress_thresholds):
        print("Parsing log file: 100% completed.", flush=True)

    print("Finished parsing log file. Total lines: {}, Valid trades: {}".format(
        total_lines, valid_lines))

    # Convert minute_bars to DataFrame
    df = pd.DataFrame.from_dict(minute_bars, orient='index')
    df.sort_index(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


class CryptoDataManager:
    def __init__(self, symbols, logger, verbose=False):
        self.data = {symbol: pd.DataFrame(columns=[
                                          'timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']) for symbol in symbols}
        self.candlesticks = {symbol: {} for symbol in symbols}
        self.candlestick_observers = []
        self.trade_observers = []
        self.logger = logger
        self.verbose = verbose
        self.last_price = {symbol: None for symbol in symbols}
        self.order_placer = None

        # For enhanced status tracking
        self.last_trade = {symbol: None for symbol in symbols}
        self.next_trigger = {symbol: None for symbol in symbols}
        self.current_trends = {symbol: {} for symbol in symbols}

    def load_historical_data(self, data_dict):
        total_symbols = len(data_dict)
        for idx, (symbol, df) in enumerate(data_dict.items(), 1):
            self.data[symbol] = df.reset_index(drop=True)
            if not df.empty:
                self.last_price[symbol] = df.iloc[-1]['close']
                self.logger.debug("Loaded historical data for {}, last price: {}".format(
                    symbol, self.last_price[symbol]))
            print(
                "Loaded historical data for {} ({}/{})".format(symbol, idx, total_symbols))

    def add_candlestick_observer(self, callback):
        self.candlestick_observers.append(callback)

    def add_trade_observer(self, callback):
        self.trade_observers.append(callback)

    def set_verbose(self, verbose):
        self.verbose = verbose

    def add_trade(self, symbol, price, timestamp, trade_reason="Live Trade"):
        price = float(price)
        # Aggregate trade into current minute candlestick
        dt = datetime.fromtimestamp(timestamp)
        minute = dt.replace(second=0, microsecond=0)

        if symbol not in self.candlesticks:
            self.candlesticks[symbol] = {}

        if minute not in self.candlesticks[symbol]:
            self.candlesticks[symbol][minute] = {
                'timestamp': int(minute.timestamp()),
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 0.0,
                'trades': 0
            }

        candle = self.candlesticks[symbol][minute]
        candle['high'] = max(candle['high'], price)
        candle['low'] = min(candle['low'], price)
        candle['close'] = price
        candle['volume'] += 0.0  # We don't have 'amount' for live data here
        candle['trades'] += 1

        self.last_price[symbol] = price

        # Notify trade observers
        for observer in self.trade_observers:
            observer(symbol, price, timestamp, trade_reason)

    def get_current_price(self, symbol):
        if not self.data[symbol].empty:
            return self.data[symbol].iloc[-1]['close']
        return None

    def get_price_range(self, symbol, minutes):
        now = pd.Timestamp.now()
        start_time = now - pd.Timedelta(minutes=minutes)
        df = self.data[symbol]
        mask = df['timestamp'] >= int(start_time.timestamp())
        relevant_data = df.loc[mask, 'close']
        if not relevant_data.empty:
            return relevant_data.min(), relevant_data.max()
        return None, None

    def get_price_dataframe(self, symbol):
        # Combine historical data with live data
        df = self.data[symbol].copy()
        if not df.empty:
            df['source'] = 'historical'
        else:
            df['source'] = pd.Series(dtype=str)
        # Add live candlesticks
        if symbol in self.candlesticks:
            live_df = pd.DataFrame.from_dict(
                self.candlesticks[symbol], orient='index')
            live_df.sort_index(inplace=True)
            live_df['source'] = 'live'
            df = pd.concat([df, live_df], ignore_index=True)
            df.drop_duplicates(subset='timestamp', keep='last', inplace=True)
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def get_data_point_count(self, symbol):
        return len(self.data[symbol])


class Trade:
    def __init__(self, trade_type, symbol, amount, price, timestamp, reason, data_source, signal_timestamp, live_trading=False, order_result=None):
        self.type = trade_type
        self.symbol = symbol
        self.amount = amount
        self.price = price
        self.timestamp = timestamp  # Time when the trade was executed
        self.reason = reason
        self.data_source = data_source  # 'historical' or 'live'
        self.signal_timestamp = signal_timestamp
        self.live_trading = live_trading
        self.order_result = order_result

    def to_dict(self):
        trade_info = {
            'type': self.type,
            'symbol': self.symbol,
            'amount': self.amount,
            'price': self.price,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'signal_timestamp': self.signal_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': self.data_source,
            'live_trading': self.live_trading,
            'reason': self.reason
        }
        if self.order_result:
            trade_info['order_result'] = self.order_result
        return trade_info


async def subscribe_to_websocket(url: str, symbol: str, data_manager):
    channel = f"live_trades_{symbol}"

    while True:  # Keep trying to reconnect
        try:
            data_manager.logger.info(
                "{}: Attempting to connect to WebSocket...".format(symbol))
            async with websockets.connect(url) as websocket:
                data_manager.logger.info(
                    "{}: Connected to WebSocket.".format(symbol))

                # Subscribing to the channel.
                subscribe_message = {
                    "event": "bts:subscribe",
                    "data": {
                        "channel": channel
                    }
                }
                await websocket.send(json.dumps(subscribe_message))
                data_manager.logger.info(
                    "{}: Subscribed to channel: {}".format(symbol, channel))

                # Receiving messages.
                async for message in websocket:
                    data_manager.logger.debug("{}: {}".format(symbol, message))
                    data = json.loads(message)
                    if data.get('event') == 'trade':
                        price = data['data']['price']
                        timestamp = int(float(data['data']['timestamp']))
                        data_manager.add_trade(
                            symbol, price, timestamp, trade_reason="Live Trade")

        except websockets.ConnectionClosed:
            data_manager.logger.error(
                "{}: Connection closed, trying to reconnect in 5 seconds...".format(symbol))
            await asyncio.sleep(5)
        except Exception as e:
            data_manager.logger.error(
                "{}: An error occurred: {}".format(symbol, str(e)))
            await asyncio.sleep(5)


class OrderPlacer:
    def __init__(self, config_file='.bitstamp'):
        self.config_file = config_file
        self.config = self.read_config(self.config_file)
        self.api_key = self.config['api_key']
        self.api_secret = bytes(self.config['api_secret'], 'utf-8')

    def read_config(self, file_name):
        # Use current working directory
        file_path = os.path.abspath(file_name)
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(
                "Failed to read config file '{}': {}".format(file_name, e))

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
                payload[key] = str(value).lower() if isinstance(
                    value, bool) else str(value)

        if 'market' in order_type:
            endpoint = "/api/v2/{}/market/{}/".format(
                'buy' if 'buy' in order_type else 'sell', currency_pair)
        else:
            endpoint = "/api/v2/{}/{}".format(
                'buy' if 'buy' in order_type else 'sell', currency_pair)

        message = 'BITSTAMP ' + self.api_key + 'POST' + 'www.bitstamp.net' + \
            endpoint + content_type + nonce + timestamp + urlencode(payload)
        signature = hmac.new(self.api_secret, msg=message.encode(
            'utf-8'), digestmod=hashlib.sha256).hexdigest()

        headers = {
            'X-Auth': "BITSTAMP {}".format(self.api_key),
            'X-Auth-Signature': signature,
            'X-Auth-Nonce': nonce,
            'X-Auth-Timestamp': timestamp,
            'X-Auth-Version': 'v2',
            'Content-Type': content_type
        }

        url = "https://www.bitstamp.net{}".format(endpoint)
        r = requests.post(url, headers=headers, data=urlencode(payload))

        if r.status_code == 200:
            return json.loads(r.content.decode('utf-8'))
        else:
            return "Error: {} - {}".format(r.status_code, r.text)

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
        self.last_trade_reason = None
        self.last_trade_data_source = None
        self.last_trade_signal_timestamp = None
        self.next_trigger = None
        self.current_trends = {}
        self.df_ma = pd.DataFrame()
        # Record the time when the strategy starts
        self.strategy_start_time = datetime.now()

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
                file_path = os.path.abspath(self.trade_log_file)
                with open(file_path, 'w') as f:
                    json.dump([trade.to_dict()
                              for trade in self.trade_log], f, indent=2)
                self.logger.info("Trades logged to '{}'".format(file_path))
            except Exception as e:
                self.logger.error(
                    "Failed to write trades to '{}': {}".format(self.trade_log_file, e))

    def run_strategy_loop(self):
        while self.running:
            df = self.data_manager.get_price_dataframe(self.symbol)
            if not df.empty:
                try:
                    # Ensure datetime index
                    df = ensure_datetime_index(df)
                    # Resample to high frequency used in backtesting
                    df_resampled = df.resample(HIGH_FREQUENCY).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum',
                        'trades': 'sum',
                        'timestamp': 'last',  # Retain the latest timestamp in the resampled period
                        'source': 'last'  # Retain the source of the last data point in the resampled period
                    }).dropna()

                    if len(df_resampled) >= self.long_window:
                        # Calculate moving averages with 'close' as the price column
                        df_ma = add_moving_averages(
                            df_resampled.copy(), self.short_window, self.long_window, price_col='close')
                        # Generate MA signals
                        df_ma = generate_ma_signals(df_ma)
                        # Get the latest signal
                        latest_signal = df_ma.iloc[-1]['MA_Signal']
                        signal_time = df_ma.index[-1]
                        current_price = df_ma.iloc[-1]['close']
                        signal_source = df_ma.iloc[-1]['source']
                        self.next_trigger = self.determine_next_trigger(df_ma)
                        self.current_trends = self.get_current_trends(df_ma)
                        self.df_ma = df_ma  # Store df_ma for status reporting
                        self.check_for_signals(
                            latest_signal, current_price, signal_time)
                    else:
                        self.logger.debug(
                            "Not enough data to compute moving averages.")
                except KeyError as e:
                    self.logger.error(
                        "Missing column during strategy loop: {}".format(e))
                except Exception as e:
                    self.logger.error(
                        "Error during strategy loop: {}".format(e))
            else:
                self.logger.debug(
                    "No data available for {} to run strategy.".format(self.symbol))
            time.sleep(60)  # Check every minute

    def determine_next_trigger(self, df_ma):
        """
        Determines the next trigger based on the moving averages.
        """
        # Placeholder for actual logic to determine the next trigger
        # This can be based on upcoming crossovers or other indicators
        # For simplicity, we'll return the last signal time and its direction
        if len(df_ma) < 2:
            return None
        last_signal = df_ma.iloc[-1]['MA_Signal']
        previous_signal = df_ma.iloc[-2]['MA_Signal']
        if last_signal != previous_signal:
            if last_signal == 1:
                return "Next trigger: Potential sell signal when short MA crosses below long MA."
            elif last_signal == -1:
                return "Next trigger: Potential buy signal when short MA crosses above long MA."
        return "Next trigger: Awaiting next crossover signal."

    def get_current_trends(self, df_ma):
        """
        Analyzes current trends towards the next trigger.
        """
        # Placeholder for trend analysis
        # This can involve analyzing recent price movements, moving average slopes, etc.
        # For simplicity, we'll return a basic trend description
        if len(df_ma) < 2:
            return {}
        short_ma_current = df_ma.iloc[-1]['Short_MA']
        short_ma_prev = df_ma.iloc[-2]['Short_MA']
        long_ma_current = df_ma.iloc[-1]['Long_MA']
        long_ma_prev = df_ma.iloc[-2]['Long_MA']

        short_ma_slope = short_ma_current - short_ma_prev
        long_ma_slope = long_ma_current - long_ma_prev

        trend = {
            'Short_MA_Slope': 'Upwards' if short_ma_slope > 0 else 'Downwards',
            'Long_MA_Slope': 'Upwards' if long_ma_slope > 0 else 'Downwards'
        }
        return trend

    def check_for_signals(self, latest_signal, current_price, signal_time):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if self.last_signal_time == signal_time:
            # Signal already processed
            return

        if latest_signal == 1 and self.position <= 0:
            # Buy signal
            self.logger.info("Buy signal at price {}".format(current_price))
            self.position = 1
            self.last_trade_reason = "MA Crossover: Short MA crossed above Long MA."
            self.execute_trade("buy", current_price, timestamp, signal_time)
            self.last_signal_time = signal_time
        elif latest_signal == -1 and self.position >= 0:
            # Sell signal
            self.logger.info("Sell signal at price {}".format(current_price))
            self.position = -1
            self.last_trade_reason = "MA Crossover: Short MA crossed below Long MA."
            self.execute_trade("sell", current_price, timestamp, signal_time)
            self.last_signal_time = signal_time

    def execute_trade(self, trade_type, price, timestamp, signal_timestamp):
        # Determine if the signal is based on historical or live data
        if signal_timestamp < self.strategy_start_time:
            data_source = 'historical'
        else:
            data_source = 'live'

        trade_info = Trade(
            trade_type,
            self.symbol,
            self.amount,
            price,
            datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'),
            self.last_trade_reason,
            data_source,
            signal_timestamp,
            live_trading=self.live_trading
        )

        # Store the last trade data source and signal timestamp
        self.last_trade_data_source = data_source
        self.last_trade_signal_timestamp = signal_timestamp

        if self.live_trading:
            result = self.order_placer.place_order(
                "market-{}".format(trade_type), self.symbol, self.amount
            )
            self.logger.info(
                "Executed live {} order: {}".format(trade_type, result))
            trade_info.order_result = result
        else:
            self.logger.info("Executed dry run {} order: {}".format(
                trade_type, trade_info.to_dict()))
            self.trade_log.append(trade_info)

        # Write the trade_info to the trade log file
        try:
            # Use current working directory
            file_path = os.path.abspath(self.trade_log_file)
            with open(file_path, 'a') as f:
                f.write(json.dumps(trade_info.to_dict()) + '\n')
            self.logger.debug("Trade info written to '{}'".format(file_path))
        except Exception as e:
            self.logger.error(
                "Failed to write trade to log file: {}".format(e))

    def get_status(self):
        """Return the current status of the strategy."""
        status = {
            'running': self.running,
            'position': self.position,
            'last_trade': None,
            'last_trade_data_source': None,
            'last_trade_signal_timestamp': None,
            'next_trigger': self.next_trigger,
            'current_trends': self.current_trends,
            'ma_difference': None,
            'ma_slope_difference': None
        }

        if self.last_trade_reason:
            status['last_trade'] = self.last_trade_reason
            status['last_trade_data_source'] = self.last_trade_data_source
            status['last_trade_signal_timestamp'] = self.last_trade_signal_timestamp.strftime(
                '%Y-%m-%d %H:%M:%S') if self.last_trade_signal_timestamp else None

        # Calculate current MA difference
        if hasattr(self, 'df_ma') and not self.df_ma.empty:
            status['ma_difference'] = self.df_ma.iloc[-1]['Short_MA'] - \
                self.df_ma.iloc[-1]['Long_MA']
            # Calculate MA slopes
            if len(self.df_ma) >= 2:
                short_ma_slope = self.df_ma.iloc[-1]['Short_MA'] - \
                    self.df_ma.iloc[-2]['Short_MA']
                long_ma_slope = self.df_ma.iloc[-1]['Long_MA'] - \
                    self.df_ma.iloc[-2]['Long_MA']
                status['ma_slope_difference'] = short_ma_slope - long_ma_slope
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
            'status': 'status',
            'chart': 'chart btcusd 1H'
        }

        # Register callbacks
        self.data_manager.add_candlestick_observer(self.candlestick_callback)
        self.data_manager.add_trade_observer(self.trade_callback)

    def emptyline(self):
        """Do nothing on empty input line."""
        pass

    def do_example(self, arg):
        """Show an example of how to use a command: example <command>"""
        command = arg.strip().lower()
        if command in self.examples:
            print("Example usage of '{}':".format(command))
            print("  {}".format(self.examples[command]))
        else:
            print(
                "No example available for '{}'. Available commands are:".format(command))
            print(", ".join(self.examples.keys()))

    def do_price(self, arg):
        """Show current price for a symbol: price <symbol>"""
        symbol = arg.strip().lower()
        if not symbol:
            print("Usage: price <symbol>")
            return
        price = self.data_manager.get_current_price(symbol)
        if price:
            print("Current price of {}: ${:.2f}".format(symbol, price))
        else:
            print("No data available for {}".format(symbol))

    def do_range(self, arg):
        """Show price range for last n minutes: range <symbol> <minutes>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: range <symbol> <minutes>")
            return
        symbol, minutes = args[0].lower(), int(args[1])
        min_price, max_price = self.data_manager.get_price_range(
            symbol, minutes)
        if min_price is not None and max_price is not None:
            print("Price range for {} in last {} minutes:".format(symbol, minutes))
            print("Min: ${:.2f}, Max: ${:.2f}".format(min_price, max_price))
        else:
            print("No data available for {} in the last {} minutes".format(
                symbol, minutes))

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
            print("Stopped 1-minute candlestick output for {}".format(symbol))
        else:
            self.candlestick_output[symbol] = True
            print("Started 1-minute candlestick output for {}".format(symbol))

    def do_ticker(self, arg):
        """Toggle real-time trade output for a symbol: ticker <symbol>"""
        symbol = arg.strip().lower()
        if not symbol:
            print("Usage: ticker <symbol>")
            return
        if symbol in self.ticker_output:
            del self.ticker_output[symbol]
            print("Stopped real-time trade output for {}".format(symbol))
        else:
            self.ticker_output[symbol] = True
            print("Started real-time trade output for {}".format(symbol))

    def candlestick_callback(self, symbol, minute, candle):
        if symbol in self.candlestick_output:
            timestamp = datetime.fromtimestamp(
                candle['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            print("{} - {}: Open: ${:.2f}, High: ${:.2f}, Low: ${:.2f}, Close: ${:.2f}, Volume: {}, Trades: {}".format(
                symbol, timestamp, candle['open'], candle['high'], candle[
                    'low'], candle['close'], candle['volume'], candle['trades']
            ))

    def trade_callback(self, symbol, price, timestamp, trade_reason):
        if symbol in self.ticker_output:
            # Convert UNIX timestamp to readable format
            time_str = datetime.fromtimestamp(
                timestamp).strftime('%Y-%m-%d %H:%M:%S')
            print("{} - {}: Price: ${:.2f}".format(symbol, time_str, price))

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
                    formatter = logging.Formatter(
                        '%(asctime)s - %(levelname)s - %(message)s')
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
                log_file_path = os.path.abspath(log_file)
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                self.data_manager.set_verbose(True)
                self.verbose = True
                print("Verbose mode enabled. Logs are being written to {}.".format(
                    log_file_path))
            except Exception as e:
                print("Failed to open log file {}: {}".format(log_file, e))

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
                key, value = arg.split('=', 1)
                if key in ['daily_order', 'ioc_order', 'fok_order', 'moc_order', 'gtd_order']:
                    options[key] = value.lower() == 'true'
                elif key == 'expire_time':
                    try:
                        options[key] = int(value)
                    except ValueError:
                        print(
                            "Invalid value for {}: {}. It should be an integer.".format(key, value))
                elif key == 'client_order_id':
                    options[key] = value
                elif key == 'limit_price':
                    try:
                        options[key] = float(value)
                    except ValueError:
                        print(
                            "Invalid value for {}: {}. It should be a float.".format(key, value))
        return options

    def do_auto_trade(self, arg):
        """Start auto-trading using the best strategy: auto_trade <amount>"""
        if self.auto_trader is not None and self.auto_trader.running:
            print(
                "Auto-trading is already running. Please stop it before starting a new one.")
            return

        args = arg.split()
        if len(args) != 1:
            print("Usage: auto_trade <amount>")
            return
        try:
            amount = float(args[0])
        except ValueError:
            print("Amount must be a number.")
            return

        # Use the directory from which the command is issued
        file_path = os.path.abspath('best_strategy.json')

        print("Current working directory: {}".format(os.getcwd()))
        print("Looking for best_strategy.json at: {}".format(file_path))

        try:
            with open(file_path, 'r') as f:
                best_strategy_params = json.load(f)
        except FileNotFoundError:
            print("Best strategy parameters file '{}' not found.".format(file_path))
            return
        except json.JSONDecodeError:
            print(
                "Best strategy parameters file '{}' is not a valid JSON.".format(file_path))
            return

        strategy_name = best_strategy_params.get('Strategy')
        if strategy_name != 'MA':
            print("The best strategy is not MA Crossover. It's {}.".format(
                strategy_name))
            return

        try:
            short_window = int(best_strategy_params['Short_Window'])
            long_window = int(best_strategy_params['Long_Window'])
        except (KeyError, ValueError) as e:
            print("Invalid strategy parameters: {}".format(e))
            return

        symbol = 'btcusd'  # Adjust as needed

        self.auto_trader = MACrossoverStrategy(
            self.data_manager, short_window, long_window, amount, symbol, self.logger, live_trading=self.live_trading)
        self.auto_trader.start()
        print("Auto-trading started with amount {} using MA Crossover strategy.".format(amount))
        print("Trades will be logged to '{}'.".format(
            self.auto_trader.trade_log_file))
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
            position = {1: 'Long', -1: 'Short',
                        0: 'Neutral'}.get(status['position'], 'Unknown')
            print("Auto-Trading Status:")
            print("  Running: {}".format(status['running']))
            print("  Position: {}".format(position))
            if status['last_trade']:
                print("  Last Trade Reason: {}".format(status['last_trade']))
                print("  Last Trade Based On: {} Data".format(
                    status['last_trade_data_source'].capitalize()))
                print("  Signal Timestamp: {}".format(
                    status['last_trade_signal_timestamp']))
            if status['next_trigger']:
                print("  {}".format(status['next_trigger']))
            if status['current_trends']:
                print("  Current Trends:")
                for key, value in status['current_trends'].items():
                    print("    {}: {}".format(key, value))
            if status['ma_difference'] is not None:
                print(
                    "  Current MA Difference (Short MA - Long MA): {:.6f}".format(status['ma_difference']))
            if status['ma_slope_difference'] is not None:
                print("  Current MA Slope Difference (Short MA Slope - Long MA Slope): {:.6f}".format(
                    status['ma_slope_difference']))
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
            print("  chart            Show a live updating chart: chart [symbol] [bar_size]")

    def do_quit(self, arg):
        """Quit the program"""
        print("Quitting...")
        if self.auto_trader is not None and self.auto_trader.running:
            self.auto_trader.stop()
        return True

    def do_chart(self, arg):
        """Show a dynamically updating candlestick chart with moving averages and trade signals: chart [symbol] [bar_size]"""
        args = arg.split()
        symbol = 'btcusd'
        bar_size = '1H'  # Default bar size is one hour
        if len(args) >= 1:
            symbol = args[0].strip().lower()
        if len(args) >= 2:
            bar_size = args[1].strip()
        if symbol not in self.data_manager.data:
            print("No data available for symbol '{}'.".format(symbol))
            return

        try:
            import dash
            from dash import dcc, html
            from dash.dependencies import Output, Input
            import plotly.graph_objs as go
            import threading
        except ImportError:
            print("Required libraries are not installed. Please install them using 'pip install dash plotly'.")
            return

        # Determine moving average windows
        if self.auto_trader and isinstance(self.auto_trader, MACrossoverStrategy):
            short_window = self.auto_trader.short_window
            long_window = self.auto_trader.long_window
        else:
            # Try to read from best_strategy.json
            try:
                with open('best_strategy.json', 'r') as f:
                    best_strategy_params = json.load(f)
                    if best_strategy_params.get('Strategy') == 'MA':
                        short_window = int(best_strategy_params['Short_Window'])
                        long_window = int(best_strategy_params['Long_Window'])
                    else:
                        print("Best strategy is not MA Crossover.")
                        return
            except Exception as e:
                print("Could not determine moving average windows. Start auto_trading or provide best_strategy.json.")
                return

        # Define the Dash app
        app = dash.Dash(__name__)
        app.layout = html.Div(children=[
            html.H1(children='{} Real-time Candlestick Chart'.format(symbol.upper())),
            dcc.Graph(id='live-graph'),
            dcc.Interval(
                id='graph-update',
                interval=60*1000,  # Update every minute
                n_intervals=0
            )
        ])

        @app.callback(
            Output('live-graph', 'figure'),
            Input('graph-update', 'n_intervals')
        )
        def update_graph_live(n):
            df = self.data_manager.get_price_dataframe(symbol)
            if df.empty:
                return {}

            # Ensure datetime index
            df = ensure_datetime_index(df)

            try:
                # Resample to the specified bar size
                df_resampled = df.resample(bar_size).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'trades': 'sum',
                    'timestamp': 'last',
                    'source': 'last'
                }).dropna()
            except ValueError as e:
                print(f"Invalid bar size '{bar_size}'. Please use a valid pandas resampling string like '1H', '15min', etc.")
                return {}

            if len(df_resampled) < long_window:
                return {}

            # Compute moving averages
            df_ma = add_moving_averages(df_resampled.copy(), short_window, long_window, price_col='close')
            # Generate MA signals
            df_ma = generate_ma_signals(df_ma)

            # Create Buy/Sell signal price columns
            df_ma['Buy_Signal_Price'] = np.where(df_ma['MA_Signal'] == 1, df_ma['close'], np.nan)
            df_ma['Sell_Signal_Price'] = np.where(df_ma['MA_Signal'] == -1, df_ma['close'], np.nan)

            # Create the candlestick chart
            candlestick = go.Candlestick(
                x=df_ma.index,
                open=df_ma['open'],
                high=df_ma['high'],
                low=df_ma['low'],
                close=df_ma['close'],
                name='Candlestick'
            )

            # Add moving averages
            short_ma = go.Scatter(
                x=df_ma.index,
                y=df_ma['Short_MA'],
                line=dict(color='blue', width=1),
                name='Short MA ({})'.format(short_window)
            )

            long_ma = go.Scatter(
                x=df_ma.index,
                y=df_ma['Long_MA'],
                line=dict(color='red', width=1),
                name='Long MA ({})'.format(long_window)
            )

            # Add buy/sell signals
            buy_signals = go.Scatter(
                x=df_ma.index,
                y=df_ma['Buy_Signal_Price'],
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=12),
                name='Buy Signal'
            )

            sell_signals = go.Scatter(
                x=df_ma.index,
                y=df_ma['Sell_Signal_Price'],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=12),
                name='Sell Signal'
            )

            data = [candlestick, short_ma, long_ma, buy_signals, sell_signals]

            layout = go.Layout(
                xaxis=dict(title='Time'),
                yaxis=dict(title='Price ($)'),
                title='{} Candlestick Chart with Moving Averages and Trade Signals'.format(symbol.upper()),
                height=600
            )

            return {'data': data, 'layout': layout}

        # Run the Dash app in a separate thread
        def run_app():
            app.run_server(debug=False, use_reloader=False)

        threading.Thread(target=run_app).start()
        print("Dash app is running at http://127.0.0.1:8050/")

    def do_exit(self, arg):
        """Exit the program"""
        return self.do_quit(arg)


def run_websocket(url, symbols, data_manager):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [subscribe_to_websocket(url, symbol, data_manager)
             for symbol in symbols]
    try:
        loop.run_until_complete(asyncio.gather(*tasks))
    except Exception as e:
        data_manager.logger.error(
            "WebSocket thread encountered an error: {}".format(e))
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
    if log_file:
        log_file_path = os.path.abspath(log_file)
    else:
        log_file_path = os.path.abspath('crypto_shell.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def main():
    parser = argparse.ArgumentParser(description="Crypto trading shell")
    parser.add_argument('-v', '--verbose', nargs='?', const=True, default=False,
                        help="Enable verbose output and optionally specify a log file (e.g., --verbose logfile.log)")
    parser.add_argument('--do-live-trades', action='store_true',
                        help="Enable live trading (default is dry run)")
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
    data_manager = CryptoDataManager(
        symbols, logger=logger, verbose=verbose_flag)

    # Read historical data
    # Use the directory from which the command is issued
    # Ensure this is the correct path to your log file
    file_path = os.path.abspath('btcusd.log')

    print("Current working directory: {}".format(os.getcwd()))

    # Limit the data to prevent memory issues (e.g., last 90 days)
    start_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now()

    # Parse historical data using the updated parse_log_file function
    try:
        df = parse_log_file(file_path, start_date, end_date)
    except Exception as e:
        print("Failed to parse log file '{}': {}".format(file_path, e))
        sys.exit(1)

    if df.empty:
        print("No historical data found in '{}' for the specified date range. Exiting.".format(
            file_path))
        sys.exit(1)

    try:
        # Retain 'timestamp' as UNIX epoch seconds
        if 'timestamp' not in df.columns:
            print(
                "Missing 'timestamp' column in log file '{}'. Exiting.".format(file_path))
            sys.exit(1)
        # Ensure 'timestamp' is integer seconds
        df['timestamp'] = df['timestamp'].astype(int)
        # Do NOT convert 'timestamp' to datetime here
        # Let technical_indicators.py handle datetime indexing
        # If 'datetime' column exists, drop it to avoid confusion
        if 'datetime' in df.columns:
            df = df.drop(columns=['datetime'])
    except KeyError as e:
        print("Missing expected column in log file: {}".format(e))
        sys.exit(1)
    except Exception as e:
        print("Error processing log file: {}".format(e))
        sys.exit(1)

    # Load aggregated data into the data manager
    data_manager.load_historical_data({'btcusd': df})

    order_placer = OrderPlacer()
    data_manager.order_placer = order_placer  # Set order placer in data manager

    shell = CryptoShell(data_manager, order_placer, logger=logger,
                        verbose=verbose_flag, live_trading=live_trading)

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
