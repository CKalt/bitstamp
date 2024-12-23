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
import logging
import threading
from datetime import datetime, timedelta
from multiprocessing import Process, Manager, set_start_method

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

# Import parse_log_file from data.loader
from data.loader import parse_log_file

# Import indicators/technical_indicators
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

# We remove CLI argument parsing. We'll read config from best_strategy.json if present.
HIGH_FREQUENCY = '1H'  # Default; can be overridden if desired from best_strategy.json

###############################################################################
# Classes: CryptoDataManager, Trade, MACrossoverStrategy, etc.
###############################################################################

class CryptoDataManager:
    def __init__(self, symbols, logger, verbose=False):
        self.data = {symbol: pd.DataFrame(columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades'
        ]) for symbol in symbols}
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
        """
        Load historical data for each symbol from a dictionary of DataFrames.
        """
        total_symbols = len(data_dict)
        for idx, (symbol, df) in enumerate(data_dict.items(), 1):
            self.data[symbol] = df.reset_index(drop=True)
            if not df.empty:
                self.last_price[symbol] = df.iloc[-1]['close']
                self.logger.debug("Loaded historical data for {}, last price: {}".format(
                    symbol, self.last_price[symbol]))
            print("Loaded historical data for {} ({}/{})".format(symbol, idx, total_symbols))

    def add_candlestick_observer(self, callback):
        """
        Register a callback for candlestick updates.
        """
        self.candlestick_observers.append(callback)

    def add_trade_observer(self, callback):
        """
        Register a callback for trade updates.
        """
        self.trade_observers.append(callback)

    def set_verbose(self, verbose):
        self.verbose = verbose

    def add_trade(self, symbol, price, timestamp, trade_reason="Live Trade"):
        """
        Add a new trade to the candlestick data. 
        Aggregates trades into current-minute candles.
        """
        price = float(price)
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
                'volume': 0.0,  # For real-time data, we don't have 'amount', so 0.0
                'trades': 1
            }
        else:
            candle = self.candlesticks[symbol][minute]
            candle['high'] = max(candle['high'], price)
            candle['low'] = min(candle['low'], price)
            candle['close'] = price
            candle['trades'] += 1

        self.last_price[symbol] = price

        # Notify any observers (including the real-time strategy, if attached)
        for observer in self.trade_observers:
            observer(symbol, price, timestamp, trade_reason)

    def get_current_price(self, symbol):
        """
        Return the last known 'close' price for the given symbol from historical data.
        """
        if not self.data[symbol].empty:
            return self.data[symbol].iloc[-1]['close']
        return None

    def get_price_range(self, symbol, minutes):
        """
        Return the min and max price in the last 'minutes' of data.
        """
        now = pd.Timestamp.now()
        start_time = now - pd.Timedelta(minutes=minutes)
        df = self.data[symbol]
        mask = df['timestamp'] >= int(start_time.timestamp())
        relevant_data = df.loc[mask, 'close']
        if not relevant_data.empty:
            return relevant_data.min(), relevant_data.max()
        return None, None

    def get_price_dataframe(self, symbol):
        """
        Combine historical data with live candlesticks for a given symbol.
        """
        df = self.data[symbol].copy()
        if not df.empty:
            df['source'] = 'historical'
        else:
            df['source'] = pd.Series(dtype=str)

        # Add live candlesticks to the DataFrame
        if symbol in self.candlesticks:
            live_df = pd.DataFrame.from_dict(self.candlesticks[symbol], orient='index')
            live_df.sort_index(inplace=True)
            live_df['source'] = 'live'
            df = pd.concat([df, live_df], ignore_index=True)
            df.drop_duplicates(subset='timestamp', keep='last', inplace=True)
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    def get_data_point_count(self, symbol):
        """
        Return the total count of data points stored for a given symbol.
        """
        return len(self.data[symbol])


class Trade:
    """
    Represents a single trade, whether historical or live.
    """
    def __init__(self, trade_type, symbol, amount, price, timestamp, reason,
                 data_source, signal_timestamp, live_trading=False, order_result=None):
        self.type = trade_type
        self.symbol = symbol
        self.amount = amount
        self.price = price
        self.timestamp = timestamp
        self.reason = reason
        self.data_source = data_source  # 'historical' or 'live'
        self.signal_timestamp = signal_timestamp
        self.live_trading = live_trading
        self.order_result = order_result

    def to_dict(self):
        """
        Convert the Trade object to a dictionary for logging/JSON.
        """
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


async def subscribe_to_websocket(url: str, symbol: str, data_manager, stop_event):
    """
    Async function to subscribe to the Bitstamp (or other) WebSocket channel
    for a given symbol, and store new trades into the data_manager.
    """
    channel = f"live_trades_{symbol}"

    while not stop_event.is_set():
        try:
            data_manager.logger.info(f"{symbol}: Attempting to connect to WebSocket...")
            async with websockets.connect(url) as websocket:
                data_manager.logger.info(f"{symbol}: Connected to WebSocket.")

                # Subscribe
                subscribe_message = {
                    "event": "bts:subscribe",
                    "data": {
                        "channel": channel
                    }
                }
                await websocket.send(json.dumps(subscribe_message))
                data_manager.logger.info(f"{symbol}: Subscribed to channel: {channel}")

                # Handle messages
                while not stop_event.is_set():
                    message = await websocket.recv()
                    data_manager.logger.debug(f"{symbol}: {message}")
                    data = json.loads(message)
                    if data.get('event') == 'trade':
                        price = data['data']['price']
                        timestamp = int(float(data['data']['timestamp']))
                        data_manager.add_trade(symbol, price, timestamp, "Live Trade")

        except websockets.ConnectionClosed:
            if stop_event.is_set():
                break
            data_manager.logger.error(f"{symbol}: Connection closed, retrying in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            if stop_event.is_set():
                break
            data_manager.logger.error(f"{symbol}: An error occurred: {str(e)}")
            await asyncio.sleep(5)


class OrderPlacer:
    """
    Handles order placement with the exchange (e.g. Bitstamp).
    """
    def __init__(self, config_file='.bitstamp'):
        self.config_file = config_file
        self.config = self.read_config(self.config_file)
        self.api_key = self.config['api_key']
        self.api_secret = bytes(self.config['api_secret'], 'utf-8')

    def read_config(self, file_name):
        file_path = os.path.abspath(file_name)
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to read config file '{file_name}': {e}")

    def place_order(self, order_type, currency_pair, amount, price=None, **kwargs):
        """
        Place a buy/sell or market/limit order using the exchange's REST API.
        """
        import time
        import uuid
        import hmac
        import hashlib
        from urllib.parse import urlencode
        import requests

        timestamp = str(int(round(time.time() * 1000)))
        nonce = str(uuid.uuid4())
        content_type = 'application/x-www-form-urlencoded'

        payload = {'amount': str(amount)}
        if price:
            payload['price'] = str(price)

        for key, value in kwargs.items():
            if value is not None:
                payload[key] = str(value).lower() if isinstance(value, bool) else str(value)

        if 'market' in order_type:
            endpoint = f"/api/v2/{'buy' if 'buy' in order_type else 'sell'}/market/{currency_pair}/"
        else:
            endpoint = f"/api/v2/{'buy' if 'buy' in order_type else 'sell'}/{currency_pair}/"

        payload_string = urlencode(payload)
        message = f"BITSTAMP {self.api_key}POSTwww.bitstamp.net{endpoint}{content_type}{nonce}{timestamp}v2{payload_string}"
        signature = hmac.new(self.api_secret, msg=message.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()

        headers = {
            'X-Auth': f'BITSTAMP {self.api_key}',
            'X-Auth-Signature': signature,
            'X-Auth-Nonce': nonce,
            'X-Auth-Timestamp': timestamp,
            'X-Auth-Version': 'v2',
            'Content-Type': content_type
        }

        logging.info(f"Request Method: POST")
        logging.info(f"Request URL: https://www.bitstamp.net{endpoint}")
        logging.info(f"Request Headers: {headers}")
        logging.info(f"Request Payload: {payload_string}")

        url = f"https://www.bitstamp.net{endpoint}"
        r = requests.post(url, headers=headers, data=payload_string)

        if r.status_code == 200:
            return json.loads(r.content.decode('utf-8'))
        else:
            logging.error(f"Error placing order: {r.status_code} - {r.text}")
            return {"status": "error", "reason": r.text, "code": "API_FAILURE"}

    def place_limit_buy_order(self, currency_pair, amount, price, **kwargs):
        return self.place_order('buy', currency_pair, amount, price, **kwargs)

    def place_limit_sell_order(self, currency_pair, amount, price, **kwargs):
        return self.place_order('sell', currency_pair, amount, price, **kwargs)


class MACrossoverStrategy:
    """
    Implements a basic Moving Average Crossover strategy with position tracking
    and optional daily trade limits.
    """
    def __init__(
        self,
        data_manager,
        short_window,
        long_window,
        amount,
        symbol,
        logger,
        live_trading=False,
        max_trades_per_day=5
    ):
        self.data_manager = data_manager
        self.order_placer = data_manager.order_placer
        self.short_window = short_window
        self.long_window = long_window
        self.initial_amount = amount
        self.current_amount = amount
        self.symbol = symbol
        self.logger = logger
        self.position = 0  # 1 for long, -1 for short, 0 for neutral
        self.running = False
        self.live_trading = live_trading
        self.trade_log = []
        self.trade_log_file = 'trades.json'
        self.last_signal_time = None
        self.last_trade_reason = None
        self.last_trade_data_source = None
        self.last_trade_signal_timestamp = None
        self.next_trigger = None
        self.current_trends = {}
        self.df_ma = pd.DataFrame()
        self.strategy_start_time = datetime.now()

        # Balance tracking
        self.initial_balance = amount
        self.current_balance = amount
        self.fee_percentage = 0.0012
        self.last_trade_price = None
        self.total_fees_paid = 0
        self.trades_executed = 0
        self.profitable_trades = 0
        self.total_profit_loss = 0

        # Daily trade limits
        self.max_trades_per_day = max_trades_per_day
        self.trade_count_today = 0
        self.current_day = datetime.utcnow().date()
        self.logger.debug(f"Trade limit set to {self.max_trades_per_day} trades/day.")

        # CHANGED / ADDED: track trades this hour for the new hourly limit
        self.trades_this_hour = []

        # CHANGED / ADDED: We attach this strategy as a trade observer so that 
        # we can update signals in real time as soon as a new trade arrives.
        # This will allow near-instant detection of MA crossovers.
        data_manager.add_trade_observer(self.check_instant_signal)

    def _clean_up_hourly_trades(self):
        """Remove trades older than 1 hour from trades_this_hour."""
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        self.trades_this_hour = [
            t for t in self.trades_this_hour if t > one_hour_ago
        ]

    def start(self):
        """
        Start the strategy loop in a background thread.
        """
        self.running = True
        self.strategy_thread = threading.Thread(target=self.run_strategy_loop, daemon=True)
        self.strategy_thread.start()
        self.logger.info("Strategy loop started.")

    def stop(self):
        """
        Stop the strategy loop and (if in dry-run) save trades to file.
        """
        self.running = False
        self.logger.info("Strategy loop stopped.")
        if self.trade_log and not self.live_trading:
            try:
                file_path = os.path.abspath(self.trade_log_file)
                with open(file_path, 'w') as f:
                    json.dump([t.to_dict() for t in self.trade_log], f, indent=2)
                self.logger.info(f"Trades logged to '{file_path}'")
            except Exception as e:
                self.logger.error(f"Failed to write trades: {e}")

    def calculate_fee(self, trade_amount, price):
        trade_value = trade_amount * price
        return trade_value * self.fee_percentage

    def update_balance(self, trade_type, price, amount):
        """
        Update balance after a trade is executed (including fees and P&L).
        """
        fee = self.calculate_fee(amount, price)
        self.total_fees_paid += fee

        if self.last_trade_price is not None:
            # P&L if closing a position
            if trade_type == "sell" and self.position == 1:  # closing long
                profit = amount * (price - self.last_trade_price) - fee
                self.current_balance += profit
                self.total_profit_loss += profit
                if profit > 0:
                    self.profitable_trades += 1
            elif trade_type == "buy" and self.position == -1:  # closing short
                profit = amount * (self.last_trade_price - price) - fee
                self.current_balance += profit
                self.total_profit_loss += profit
                if profit > 0:
                    self.profitable_trades += 1

        self.last_trade_price = price
        self.trades_executed += 1

        # Adjust self.current_amount based on new balance
        balance_ratio = self.current_balance / self.initial_balance
        self.current_amount = self.initial_amount * balance_ratio
        
        self.logger.info(
            f"Trade completed - Balance: ${self.current_balance:.2f}, "
            f"Fees: ${fee:.2f}, Next trade amount: {self.current_amount:.8f}, "
            f"Total P&L: ${self.total_profit_loss:.2f}"
        )

    def run_strategy_loop(self):
        """
        Strategy loop that checks for signals every minute and acts accordingly.
        (You can disable or ignore this if you want purely instant trades.)
        """
        while self.running:
            df = self.data_manager.get_price_dataframe(self.symbol)
            if not df.empty:
                try:
                    df = ensure_datetime_index(df)
                    # Original 1-hour resampling approach:
                    df_resampled = df.resample(HIGH_FREQUENCY).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum',
                        'trades': 'sum',
                        'timestamp': 'last',
                        'source': 'last'
                    }).dropna()

                    if len(df_resampled) >= self.long_window:
                        df_ma = add_moving_averages(
                            df_resampled.copy(), self.short_window, self.long_window, price_col='close')
                        df_ma = generate_ma_signals(df_ma)

                        latest_signal = df_ma.iloc[-1]['MA_Signal']
                        signal_time = df_ma.index[-1]
                        current_price = df_ma.iloc[-1]['close']
                        signal_source = df_ma.iloc[-1]['source']

                        self.next_trigger = self.determine_next_trigger(df_ma)
                        self.current_trends = self.get_current_trends(df_ma)
                        self.df_ma = df_ma

                        self.check_for_signals(latest_signal, current_price, signal_time)
                    else:
                        self.logger.debug("Not enough data to compute MAs.")
                except Exception as e:
                    self.logger.error(f"Error in strategy loop for {self.symbol}: {e}")
            else:
                self.logger.debug(f"No data loaded for {self.symbol} yet.")
            time.sleep(60)

    def determine_next_trigger(self, df_ma):
        """
        Look at the latest signals, produce a text explanation of next potential trigger.
        """
        if len(df_ma) < 2:
            return None
        last_signal = df_ma.iloc[-1]['MA_Signal']
        prev_signal = df_ma.iloc[-2]['MA_Signal']
        if last_signal != prev_signal:
            if last_signal == 1:
                return "Next trigger: Potential SELL if short crosses below long."
            elif last_signal == -1:
                return "Next trigger: Potential BUY if short crosses above long."
        return "Next trigger: Awaiting next crossover signal."

    def get_current_trends(self, df_ma):
        """
        Analyze short/long MA slopes and price trend.
        """
        if len(df_ma) < 2:
            return {}
        short_ma_curr = df_ma.iloc[-1]['Short_MA']
        short_ma_prev = df_ma.iloc[-2]['Short_MA']
        long_ma_curr = df_ma.iloc[-1]['Long_MA']
        long_ma_prev = df_ma.iloc[-2]['Long_MA']

        short_ma_slope = short_ma_curr - short_ma_prev
        long_ma_slope = long_ma_curr - long_ma_prev

        return {
            'Short_MA_Slope': 'Upwards' if short_ma_slope > 0 else 'Downwards',
            'Long_MA_Slope': 'Upwards' if long_ma_slope > 0 else 'Downwards',
            'Price_Trend': 'Bullish' if short_ma_curr > long_ma_curr else 'Bearish',
            'Trend_Strength': abs(short_ma_curr - long_ma_curr) / long_ma_curr * 100 if long_ma_curr != 0 else 0
        }

    # CHANGED / ADDED: A new callback that triggers on every real-time trade.
    def check_instant_signal(self, symbol, price, timestamp, trade_reason):
        """
        Real-time callback whenever a new trade is received for `symbol`.
        We'll do a quick 'MA crossover' check on streaming data, 
        and if there's a new signal, execute the trade instantly.
        """
        if not self.running:
            # If the strategy is not started, do nothing.
            return
        if symbol != self.symbol:
            return  # We only trade on self.symbol

        # Build/Update a rolling list of 'close' prices to approximate MAs in real time
        # Example approach: you might want to store more data, or fetch from data_manager.
        df_live = self.data_manager.get_price_dataframe(symbol)
        if df_live.empty:
            return

        # We'll skip resampling. Instead, just ensure a DateTime index and compute MAs directly.
        df_live = ensure_datetime_index(df_live)

        # Only proceed if we have enough data for the long_window
        if len(df_live) < self.long_window:
            return

        # We'll compute the MAs directly on the last N rows:
        df_ma = df_live.copy()
        df_ma['Short_MA'] = df_ma['close'].rolling(self.short_window).mean()
        df_ma['Long_MA'] = df_ma['close'].rolling(self.long_window).mean()
        df_ma.dropna(inplace=True)  # Must drop until both MAs are valid

        if df_ma.empty:
            return

        latest = df_ma.iloc[-1]
        short_ma_now = latest['Short_MA']
        long_ma_now = latest['Long_MA']
        # Convert to a simple signal: 1 if short>long, -1 if short<long
        signal_now = 1 if short_ma_now > long_ma_now else -1

        # Compare to the previous row's signal (if any) to detect a 'crossover'
        if len(df_ma) < 2:
            return

        prev = df_ma.iloc[-2]
        prev_signal = 1 if prev['Short_MA'] > prev['Long_MA'] else -1

        if signal_now == prev_signal:
            return  # No change in signal, do nothing

        # If we do have a signal change, let's record signal_time as the row's index
        signal_time = df_ma.index[-1]

        # The same check_for_signals logic but called immediately
        self.check_for_signals(signal_now, price, signal_time)

    def check_for_signals(self, latest_signal, current_price, signal_time):
        """
        Decide whether to buy or sell based on the latest MA signal.
        (Now also used by check_instant_signal in real time.)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self.last_signal_time == signal_time:
            return

        # MA CROSS: 1 => means go long if not already
        if latest_signal == 1 and self.position <= 0:
            self.logger.info(f"Buy signal triggered at {current_price}")
            self.position = 1
            self.last_trade_reason = "MA Crossover: short above long."
            self.execute_trade("buy", current_price, timestamp, signal_time)
            self.last_signal_time = signal_time

        # MA CROSS: -1 => means go short if not already
        elif latest_signal == -1 and self.position >= 0:
            self.logger.info(f"Sell signal triggered at {current_price}")
            self.position = -1
            self.last_trade_reason = "MA Crossover: short below long."
            self.execute_trade("sell", current_price, timestamp, signal_time)
            self.last_signal_time = signal_time

    def execute_trade(self, trade_type, price, timestamp, signal_timestamp):
        """
        Execute a trade (buy or sell) and update balances if we haven't exceeded daily or hourly limits.
        """
        today = datetime.utcnow().date()
        if today != self.current_day:
            self.current_day = today
            self.trade_count_today = 0
            self.logger.debug("New day, resetting daily trade count.")

        # Check daily trade limit
        if self.trade_count_today >= self.max_trades_per_day:
            self.logger.info(f"Reached daily trade limit {self.max_trades_per_day}, skipping trade.")
            return

        # ADDED: Check hourly trade limit (3 trades per rolling hour)
        self._clean_up_hourly_trades()
        max_trades_per_hour = 3
        if len(self.trades_this_hour) >= max_trades_per_hour:
            self.logger.info(f"Reached hourly trade limit {max_trades_per_hour}, skipping trade.")
            return

        data_source = 'historical' if signal_timestamp < self.strategy_start_time else 'live'
        trade_info = Trade(
            trade_type,
            self.symbol,
            self.current_amount,
            price,
            datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'),
            self.last_trade_reason,
            data_source,
            signal_timestamp,
            live_trading=self.live_trading
        )

        self.last_trade_data_source = data_source
        self.last_trade_signal_timestamp = signal_timestamp

        if self.live_trading:
            result = self.order_placer.place_order(
                f"market-{trade_type}", self.symbol, self.current_amount)
            self.logger.info(f"Executed LIVE {trade_type} order: {result}")
            trade_info.order_result = result
            self.trade_count_today += 1
            self.logger.debug(f"Trade count for {self.current_day}: {self.trade_count_today}/{self.max_trades_per_day}")
            self.update_balance(trade_type, price, self.current_amount)
        else:
            self.logger.info(f"Executed DRY RUN {trade_type} order: {trade_info.to_dict()}")
            self.trade_log.append(trade_info)
            self.update_balance(trade_type, price, self.current_amount)

        # Increment the hourly trade list
        self.trades_this_hour.append(datetime.utcnow())

        # Write to log file (trades.json)
        try:
            file_path = os.path.abspath(self.trade_log_file)
            with open(file_path, 'a') as f:
                f.write(json.dumps(trade_info.to_dict()) + '\n')
            self.logger.debug(f"Trade info appended to '{file_path}'")
        except Exception as e:
            self.logger.error(f"Failed to write trade to log file: {e}")

    def get_status(self):
        """
        Return a dictionary summarizing the current status of this strategy.
        """
        status = {
            'running': self.running,
            'position': self.position,
            'last_trade': None,
            'last_trade_data_source': None,
            'last_trade_signal_timestamp': None,
            'next_trigger': self.next_trigger,
            'current_trends': self.current_trends,
            'ma_difference': None,
            'ma_slope_difference': None,
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'total_return_pct': ((self.current_balance / self.initial_balance) - 1) * 100,
            'total_fees_paid': self.total_fees_paid,
            'trades_executed': self.trades_executed,
            'profitable_trades': self.profitable_trades,
            'win_rate': (self.profitable_trades / self.trades_executed * 100) if self.trades_executed else 0,
            'current_trade_amount': self.current_amount,
            'total_profit_loss': self.total_profit_loss,
            'average_profit_per_trade': (self.total_profit_loss / self.trades_executed) if self.trades_executed else 0,
            'trade_count_today': self.trade_count_today,
            'remaining_trades_today': max(0, self.max_trades_per_day - self.trade_count_today)
        }

        if self.last_trade_reason:
            status['last_trade'] = self.last_trade_reason
            status['last_trade_data_source'] = self.last_trade_data_source
            if self.last_trade_signal_timestamp:
                status['last_trade_signal_timestamp'] = self.last_trade_signal_timestamp.strftime('%Y-%m-%d %H:%M:%S')

        # If we have a DataFrame of MAs, show difference
        if hasattr(self, 'df_ma') and not self.df_ma.empty:
            status['ma_difference'] = self.df_ma.iloc[-1]['Short_MA'] - self.df_ma.iloc[-1]['Long_MA']
            if len(self.df_ma) >= 2:
                short_ma_slope = self.df_ma.iloc[-1]['Short_MA'] - self.df_ma.iloc[-2]['Short_MA']
                long_ma_slope = self.df_ma.iloc[-1]['Long_MA'] - self.df_ma.iloc[-2]['Long_MA']
                status['ma_slope_difference'] = short_ma_slope - long_ma_slope
                status['short_ma_momentum'] = 'Increasing' if short_ma_slope > 0 else 'Decreasing'
                status['long_ma_momentum'] = 'Increasing' if long_ma_slope > 0 else 'Decreasing'
                status['momentum_alignment'] = (
                    'Aligned'
                    if (short_ma_slope > 0 and long_ma_slope > 0)
                    or (short_ma_slope < 0 and long_ma_slope < 0)
                    else 'Diverging'
                )

        if self.trades_executed > 0:
            status['average_fee_per_trade'] = self.total_fees_paid / self.trades_executed
            status['risk_reward_ratio'] = abs(self.total_profit_loss / self.total_fees_paid) if self.total_fees_paid > 0 else 0

        return status

###############################################################################
# The interactive cmd-based shell
###############################################################################
import cmd
from flask import Flask, request
import requests

def run_dash_app(data_manager_dict, symbol, bar_size, short_window, long_window):
    """
    Dash-based real-time candlestick chart with MA signals for the chosen symbol.
    """
    import dash
    from dash import dcc, html
    from dash.dependencies import Output, Input
    import plotly.graph_objs as go
    import threading
    import pandas as pd
    import numpy as np

    server = Flask(__name__)
    app = dash.Dash(__name__, server=server)

    @server.route('/shutdown')
    def shutdown():
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            return 'Not running with the Werkzeug Server'
        func()
        return 'Server shutting down...'

    app.layout = html.Div(children=[
        html.H1(children='{} Real-time Candlestick Chart'.format(symbol.upper())),
        dcc.Graph(id='live-graph', style={'width': '100%', 'height': '80vh'}),
        dcc.Interval(id='graph-update', interval=60*1000, n_intervals=0)
    ])

    @app.callback(
        Output('live-graph', 'figure'),
        [Input('graph-update', 'n_intervals'),
         Input('live-graph', 'relayoutData')]
    )
    def update_graph_live(n, relayout_data):
        df = pd.DataFrame.from_dict(data_manager_dict[symbol])
        if df.empty:
            return {}

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)

        try:
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
        except ValueError:
            return {}

        if len(df_resampled) < long_window:
            return {}

        df_ma = add_moving_averages(df_resampled.copy(), short_window, long_window, price_col='close')
        df_ma = generate_ma_signals(df_ma)
        df_ma['Signal_Change'] = df_ma['MA_Signal'].diff()
        df_ma['Buy_Signal_Price'] = np.where(df_ma['Signal_Change'] == 2, df_ma['close'], np.nan)
        df_ma['Sell_Signal_Price'] = np.where(df_ma['Signal_Change'] == -2, df_ma['close'], np.nan)

        # Determine chart range
        if relayout_data and 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            x_start = pd.to_datetime(relayout_data['xaxis.range[0]'])
            x_end = pd.to_datetime(relayout_data['xaxis.range[1]'])
        else:
            x_end = df_ma.index.max()
            x_start = x_end - pd.Timedelta(days=7)

        df_visible = df_ma[(df_ma.index >= x_start) & (df_ma.index <= x_end)]
        if df_visible.empty:
            df_visible = df_ma

        y_min = df_visible[['low','Short_MA','Long_MA','Buy_Signal_Price','Sell_Signal_Price']].min().min()
        y_max = df_visible[['high','Short_MA','Long_MA','Buy_Signal_Price','Sell_Signal_Price']].max().max()
        y_padding = (y_max - y_min) * 0.05
        y_min -= y_padding
        y_max += y_padding

        candlestick = go.Candlestick(
            x=df_visible.index,
            open=df_visible['open'],
            high=df_visible['high'],
            low=df_visible['low'],
            close=df_visible['close'],
            name='Candlestick'
        )
        short_ma_line = go.Scatter(
            x=df_visible.index,
            y=df_visible['Short_MA'],
            line=dict(color='blue', width=1),
            name=f'Short MA ({short_window})'
        )
        long_ma_line = go.Scatter(
            x=df_visible.index,
            y=df_visible['Long_MA'],
            line=dict(color='red', width=1),
            name=f'Long MA ({long_window})'
        )
        buy_signals = go.Scatter(
            x=df_visible.index,
            y=df_visible['Buy_Signal_Price'],
            mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=12),
            name='Buy Signal'
        )
        sell_signals = go.Scatter(
            x=df_visible.index,
            y=df_visible['Sell_Signal_Price'],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=12),
            name='Sell Signal'
        )

        data = [candlestick, short_ma_line, long_ma_line, buy_signals, sell_signals]
        layout = go.Layout(
            xaxis=dict(title='Time', range=[x_start, x_end]),
            yaxis=dict(title='Price ($)', range=[y_min, y_max]),
            title='{} Candlestick Chart with MAs'.format(symbol.upper()),
            height=800
        )

        return {'data': data, 'layout': layout}

    app.run_server(debug=False, use_reloader=False)


class CryptoShell(cmd.Cmd):
    """
    An interactive command-based shell for controlling the Crypto trading system.
    """
    intro = 'Welcome to the Crypto Shell (No CLI args). Type help or ? to list commands.\n'
    prompt = '(crypto) '

    def __init__(self, data_manager, order_placer, logger,
                 verbose=False, live_trading=False, stop_event=None,
                 max_trades_per_day=5):
        super().__init__()
        self.data_manager = data_manager
        self.order_placer = order_placer
        self.data_manager.order_placer = order_placer  # Make sure they match
        self.logger = logger
        self.candlestick_output = {}
        self.ticker_output = {}
        self.verbose = verbose
        self.live_trading = live_trading
        self.auto_trader = None
        self.chart_process = None
        self.stop_event = stop_event
        self.manager = Manager()
        self.data_manager_dict = self.manager.dict()
        self.max_trades_per_day = max_trades_per_day

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
        """
        Called when user presses Enter with no command input.
        We override to do nothing instead of repeating last command.
        """
        pass

    def do_example(self, arg):
        """
        Show an example usage of a command: example <command>
        """
        command = arg.strip().lower()
        if command in self.examples:
            print("Example usage of '{}':".format(command))
            print("  {}".format(self.examples[command]))
        else:
            print("No example for '{}'. Available commands:".format(command))
            print(", ".join(self.examples.keys()))

    def do_price(self, arg):
        """
        Show current price for a symbol: price <symbol>
        """
        symbol = arg.strip().lower()
        if not symbol:
            print("Usage: price <symbol>")
            return
        price = self.data_manager.get_current_price(symbol)
        if price:
            print("Current price of {}: ${:.2f}".format(symbol, price))
        else:
            print("No data for {}".format(symbol))

    def do_range(self, arg):
        """
        Show min and max price in last N minutes: range <symbol> <minutes>
        """
        args = arg.split()
        if len(args) != 2:
            print("Usage: range <symbol> <minutes>")
            return
        symbol, minutes = args[0].lower(), int(args[1])
        min_price, max_price = self.data_manager.get_price_range(symbol, minutes)
        if min_price is not None and max_price is not None:
            print(f"Price range for {symbol} over last {minutes} minutes:")
            print(f"Min: ${min_price:.2f}, Max: ${max_price:.2f}")
        else:
            print(f"No data for {symbol} in that timeframe")

    def do_buy(self, arg):
        """
        Place a market buy order: buy <symbol> <amount>
        """
        args = arg.split()
        if len(args) != 2:
            print("Usage: buy <symbol> <amount>")
            return
        symbol, amount = args[0].lower(), float(args[1])
        result = self.order_placer.place_order("market-buy", symbol, amount)
        print(json.dumps(result, indent=2))

    def do_sell(self, arg):
        """
        Place a market sell order: sell <symbol> <amount>
        """
        args = arg.split()
        if len(args) != 2:
            print("Usage: sell <symbol> <amount>")
            return
        symbol, amount = args[0].lower(), float(args[1])
        result = self.order_placer.place_order("market-sell", symbol, amount)
        print(json.dumps(result, indent=2))

    def do_candles(self, arg):
        """
        Toggle 1-minute candlestick printout: candles <symbol>
        """
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
        """
        Toggle real-time trade output: ticker <symbol>
        """
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
        """
        Callback that prints candlestick data if toggled on via candles <symbol>.
        """
        if symbol in self.candlestick_output:
            timestamp = datetime.fromtimestamp(candle['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{symbol} - {timestamp}: "
                  f"Open={candle['open']:.2f}, High={candle['high']:.2f}, "
                  f"Low={candle['low']:.2f}, Close={candle['close']:.2f}, "
                  f"Volume={candle['volume']}, Trades={candle['trades']}")

    def trade_callback(self, symbol, price, timestamp, trade_reason):
        """
        Callback that prints real-time trades if toggled on via ticker <symbol>.
        """
        if symbol in self.ticker_output:
            time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{symbol} - {time_str}: Price=${price:.2f}")

    def do_verbose(self, arg):
        """
        Enable verbose logging to console or to a specified log file: verbose [logfile]
        """
        arg = arg.strip()
        if not arg:
            if not self.verbose:
                self.logger.setLevel(logging.DEBUG)
                debug_handlers = [h for h in self.logger.handlers
                                  if isinstance(h, logging.StreamHandler) and h.level == logging.DEBUG]
                if not debug_handlers:
                    debug_stream_handler = logging.StreamHandler(sys.stderr)
                    debug_stream_handler.setLevel(logging.DEBUG)
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    debug_stream_handler.setFormatter(formatter)
                    self.logger.addHandler(debug_stream_handler)
                self.data_manager.set_verbose(True)
                self.verbose = True
                print("Verbose mode enabled.")
            else:
                print("Verbose mode is already enabled.")
        else:
            log_file = arg
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    self.logger.removeHandler(handler)
            try:
                log_file_path = os.path.abspath(log_file)
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                self.data_manager.set_verbose(True)
                self.verbose = True
                print(f"Verbose logs being written to {log_file_path}.")
            except Exception as e:
                print(f"Failed to open log file {log_file}: {e}")

    def parse_order_options(self, args):
        """
        Parse key=value options for limit orders (like daily_order, ioc_order, etc.).
        """
        options = {}
        for arg in args:
            if '=' in arg:
                key, value = arg.split('=', 1)
                if key in ['daily_order', 'ioc_order', 'fok_order', 'moc_order', 'gtd_order']:
                    options[key] = (value.lower() == 'true')
                elif key == 'expire_time':
                    try:
                        options[key] = int(value)
                    except ValueError:
                        print(f"Invalid value for {key}: {value} (should be int).")
                elif key == 'client_order_id':
                    options[key] = value
                elif key == 'limit_price':
                    try:
                        options[key] = float(value)
                    except ValueError:
                        print(f"Invalid value for {key}: {value} (should be float).")
        return options

    def do_limit_buy(self, arg):
        """
        Place a limit buy order: limit_buy <symbol> <amount> <price> [options]
        """
        args = arg.split()
        if len(args) < 3:
            print("Usage: limit_buy <symbol> <amount> <price> [options]")
            return
        symbol, amount, price = args[0].lower(), float(args[1]), float(args[2])
        options = self.parse_order_options(args[3:])
        result = self.order_placer.place_limit_buy_order(symbol, amount, price, **options)
        print(json.dumps(result, indent=2))

    def do_limit_sell(self, arg):
        """
        Place a limit sell order: limit_sell <symbol> <amount> <price> [options]
        """
        args = arg.split()
        if len(args) < 3:
            print("Usage: limit_sell <symbol> <amount> <price> [options]")
            return
        symbol, amount, price = args[0].lower(), float(args[1]), float(args[2])
        options = self.parse_order_options(args[3:])
        result = self.order_placer.place_limit_sell_order(symbol, amount, price, **options)
        print(json.dumps(result, indent=2))

    def do_auto_trade(self, arg):
        """
        Start auto-trading using the best strategy from best_strategy.json, e.g.: auto_trade 2
        """
        if self.auto_trader and self.auto_trader.running:
            print("Auto-trading is already running. Stop it first.")
            return
        args_list = arg.split()
        if len(args_list) != 1:
            print("Usage: auto_trade <amount>")
            return
        try:
            amount = float(args_list[0])
        except ValueError:
            print("Amount must be numeric.")
            return

        file_path = os.path.abspath('best_strategy.json')
        if not os.path.exists(file_path):
            print(f"Error: '{file_path}' not found.")
            return

        with open(file_path, 'r') as f:
            best_strategy_params = json.load(f)

        strategy_name = best_strategy_params.get('Strategy')
        if strategy_name != 'MA':
            print(f"Best strategy is not 'MA'; it's {strategy_name}.")
            return

        short_window = int(best_strategy_params.get('Short_Window', 12))
        long_window = int(best_strategy_params.get('Long_Window', 36))
        do_live = best_strategy_params.get('do_live_trades', False)
        max_trades_day = best_strategy_params.get('max_trades_per_day', 5)

        self.auto_trader = MACrossoverStrategy(
            self.data_manager,
            short_window,
            long_window,
            amount,
            'btcusd',
            self.logger,
            live_trading=do_live,
            max_trades_per_day=max_trades_day
        )
        self.auto_trader.start()
        print(f"Auto-trading started with amount {amount}, using MA strategy "
              f"(Short={short_window}, Long={long_window}). do_live_trades={do_live}")

    def do_stop_auto_trade(self, arg):
        """
        Stop auto-trading if running.
        """
        if self.auto_trader and self.auto_trader.running:
            self.auto_trader.stop()
            print("Auto-trading stopped.")
        else:
            print("No auto-trading is running.")

    def do_status(self, arg):
        """
        Show status of auto-trading, including balance, P&L, and open position.
        """
        if self.auto_trader and self.auto_trader.running:
            status = self.auto_trader.get_status()
            pos_str = {1:'Long', -1:'Short', 0:'Neutral'}.get(status['position'], 'Unknown')

            print("\nAuto-Trading Status:")
            print("━"*50)
            print(f"  • Running: {status['running']}")
            print(f"  • Position: {pos_str}")
            print(f"  • Daily Trades: {status['trade_count_today']}/{self.auto_trader.max_trades_per_day}")
            print(f"  • Remaining Trades Today: {status['remaining_trades_today']}")
            print("\nBalance and Performance:")
            print(f"  • Initial Balance: ${status['initial_balance']:.2f}")
            print(f"  • Current Balance: ${status['current_balance']:.2f}")
            print(f"  • Total Return: {status['total_return_pct']:.2f}%")
            print(f"  • Total P&L: ${status['total_profit_loss']:.2f}")
            print(f"  • Current Trade Amount: {status['current_trade_amount']:.8f}")
            print(f"  • Total Fees Paid: ${status['total_fees_paid']:.2f}")

            print("\nTrading Statistics:")
            print(f"  • Total Trades: {status['trades_executed']}")
            print(f"  • Profitable Trades: {status['profitable_trades']}")
            print(f"  • Win Rate: {status['win_rate']:.1f}%")
            if status['trades_executed']>0:
                print(f"  • Avg Profit/Trade: ${status['average_profit_per_trade']:.2f}")
                print(f"  • Avg Fee/Trade: ${status['average_fee_per_trade']:.2f}")
                print(f"  • Risk/Reward Ratio: {status['risk_reward_ratio']:.2f}")

            if status['last_trade']:
                print("\nLast Trade Info:")
                print(f"  • Reason: {status['last_trade']}")
                print(f"  • Data Source: {status['last_trade_data_source']}")
                print(f"  • Signal Time: {status['last_trade_signal_timestamp']}")

            print("\nTechnical Analysis:")
            if status['next_trigger']:
                print(f"  • {status['next_trigger']}")
            if status['current_trends']:
                print("  • Current Trends:")
                for k, v in status['current_trends'].items():
                    print(f"    ◦ {k}: {v}")
            if status['ma_difference'] is not None:
                print(f"  • MA Difference: {status['ma_difference']:.4f}")
            if status['ma_slope_difference'] is not None:
                print(f"  • MA Slope Difference: {status['ma_slope_difference']:.4f}")
            if 'short_ma_momentum' in status:
                print(f"  • Short MA Momentum: {status['short_ma_momentum']}")
            if 'long_ma_momentum' in status:
                print(f"  • Long MA Momentum: {status['long_ma_momentum']}")
            if 'momentum_alignment' in status:
                print(f"  • Momentum Alignment: {status['momentum_alignment']}")

            if status['trades_executed'] == 0:
                print("\nNo trades yet, stats are limited.")
            elif status['win_rate'] < 40:
                print("Warning: Win rate is below 40%. Consider reviewing parameters.")
            if status['current_balance'] < status['initial_balance']*0.9:
                print("Warning: Balance is over 10% below initial.")
            if status['remaining_trades_today'] <= 1:
                print("Warning: Approaching daily trade limit!")

            session_duration = datetime.now() - self.auto_trader.strategy_start_time
            hours = session_duration.total_seconds() / 3600
            print(f"\nSession Duration: {hours:.1f} hours\n")
            print("━"*50)
        else:
            print("Auto-trading is not running.")

    def do_quit(self, arg):
        """
        Quit the program, shutting down threads and processes gracefully.
        """
        print("Quitting...")
        if self.auto_trader and self.auto_trader.running:
            self.auto_trader.stop()
        if self.chart_process and self.chart_process.is_alive():
            self.stop_dash_app()
        if self.stop_event:
            self.stop_event.set()
        return True

    def stop_dash_app(self):
        """
        If a Dash app is running in a separate process, attempt to shut it down.
        """
        if self.chart_process and self.chart_process.is_alive():
            try:
                requests.get('http://127.0.0.1:8050/shutdown')
                self.chart_process.join()
                print("Dash app shut down.")
            except Exception as e:
                print("Failed to shut down Dash app:", e)

    def do_exit(self, arg):
        """
        Alias for 'quit'.
        """
        return self.do_quit(arg)

    def do_chart(self, arg):
        """
        Show a Dash-based chart: chart [symbol] [bar_size].
        E.g., chart btcusd 1H
        """
        args = arg.split()
        symbol = 'btcusd'
        bar_size = '1H'
        if len(args) >= 1:
            symbol = args[0].strip().lower()
        if len(args) >= 2:
            bar_size = args[1].strip()
        if symbol not in self.data_manager.data:
            print(f"No data for symbol '{symbol}'.")
            return

        try:
            import dash
            from dash import dcc, html
            from dash.dependencies import Output, Input
            import plotly.graph_objs as go
            from flask import Flask, request
            from multiprocessing import Process
        except ImportError:
            print("Install dash & plotly first (pip install dash plotly).")
            return

        short_window = 12
        long_window = 36
        if self.auto_trader and isinstance(self.auto_trader, MACrossoverStrategy):
            short_window = self.auto_trader.short_window
            long_window = self.auto_trader.long_window
        else:
            # Attempt to read best_strategy.json
            try:
                with open('best_strategy.json','r') as f:
                    best_params = json.load(f)
                if best_params.get('Strategy') == 'MA':
                    short_window = int(best_params['Short_Window'])
                    long_window = int(best_params['Long_Window'])
            except:
                print("Could not read 'best_strategy.json' for windows. Using defaults.")

        self.data_manager_dict[symbol] = self.data_manager.get_price_dataframe(symbol).to_dict('list')

        def update_shared_data():
            while not self.stop_event.is_set():
                self.data_manager_dict[symbol] = self.data_manager.get_price_dataframe(symbol).to_dict('list')
                time.sleep(60)

        threading.Thread(target=update_shared_data, daemon=True).start()

        self.chart_process = Process(
            target=run_dash_app,
            args=(self.data_manager_dict, symbol, bar_size, short_window, long_window)
        )
        self.chart_process.start()
        print("Dash app is running at http://127.0.0.1:8050/")
        time.sleep(1)

###############################################################################
# The main() function: no CLI args, all config read from best_strategy.json
###############################################################################
def run_websocket(url, symbols, data_manager, stop_event):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [subscribe_to_websocket(url, symbol, data_manager, stop_event) for symbol in symbols]

    async def main():
        await asyncio.gather(*tasks)

    try:
        loop.run_until_complete(main())
    except Exception as e:
        data_manager.logger.error(f"WebSocket encountered error: {e}")
    finally:
        loop.close()

def setup_logging(verbose):
    logger = logging.getLogger("CryptoShellLogger")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler('crypto_shell.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def main():
    """
    Main entry point: read best_strategy.json for config, parse historical log if present,
    then launch the CryptoShell with or without live trades.
    """
    # Attempt to load config from best_strategy.json
    config_file = os.path.abspath("best_strategy.json")
    if not os.path.exists(config_file):
        print(f"No '{config_file}' found. Using default settings.")
        config = {}
    else:
        with open(config_file, 'r') as f:
            config = json.load(f)

    start_back = config.get('start_window_days_back', 30)
    end_back = config.get('end_window_days_back', 0)
    do_live = config.get('do_live_trades', False)
    max_trades = config.get('max_trades_per_day', 5)

    # Determine date range
    now = datetime.now()
    start_date = now - timedelta(days=start_back) if start_back else None
    end_date = now - timedelta(days=end_back) if end_back else None

    if start_date and end_date and start_date >= end_date:
        print("Invalid date range from best_strategy.json; ignoring end_date.")
        end_date = None

    # Setup logging
    logger = setup_logging(verbose=False)
    if do_live:
        logger.info("Running in LIVE trading mode.")
    else:
        logger.info("Running in DRY RUN mode.")

    # Parse historical data if we have a log file
    log_file_path = os.path.abspath("btcusd.log")
    if not os.path.exists(log_file_path):
        print(f"No local log file '{log_file_path}'. Relying on real-time data only.")
        df = pd.DataFrame()
    else:
        df = parse_log_file(log_file_path, start_date, end_date)

    # Prepare DataFrame columns if data is loaded
    if not df.empty:
        df.rename(columns={'price': 'close'}, inplace=True)
        df['open'] = df['close']
        df['high'] = df['close']
        df['low'] = df['close']
        df['trades'] = 1
        if 'volume' not in df.columns:
            df['volume'] = df.get('amount', 0.0)

    # Create a data manager and load historical data
    data_manager = CryptoDataManager(["btcusd"], logger=logger)
    if not df.empty:
        data_manager.load_historical_data({'btcusd': df})

    # Set up order placer
    order_placer = OrderPlacer()
    data_manager.order_placer = order_placer

    # Start the interactive shell
    stop_event = threading.Event()
    shell = CryptoShell(
        data_manager=data_manager,
        order_placer=order_placer,
        logger=logger,
        verbose=False,
        live_trading=do_live,
        stop_event=stop_event,
        max_trades_per_day=max_trades
    )

    # Start the WebSocket thread for real-time data
    url = 'wss://ws.bitstamp.net'
    websocket_thread = threading.Thread(
        target=run_websocket, args=(url, ["btcusd"], data_manager, stop_event), daemon=True)
    websocket_thread.start()
    logger.debug("WebSocket thread started.")

    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting gracefully.")
        shell.do_quit(None)
    finally:
        stop_event.set()
        if websocket_thread.is_alive():
            websocket_thread.join()
        if shell.auto_trader and shell.auto_trader.running:
            shell.auto_trader.stop()
        if shell.chart_process and shell.chart_process.is_alive():
            shell.stop_dash_app()

if __name__ == '__main__':
    set_start_method('spawn')
    main()
