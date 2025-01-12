# src/tdr_core/data_manager.py

import pandas as pd
import logging
import threading
from datetime import datetime, timedelta

from indicators.technical_indicators import ensure_datetime_index

###############################################################################
# CryptoDataManager
###############################################################################
class CryptoDataManager:
    """
    Stores and manages trade data and candle data for both historical and live data.
    """
    def __init__(self, symbols, logger, verbose=False):
        self.data = {
            symbol: pd.DataFrame(
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']
            ) for symbol in symbols
        }
        self.candlesticks = {symbol: {} for symbol in symbols}
        self.candlestick_observers = []
        self.trade_observers = []
        self.logger = logger
        self.verbose = verbose

        # Will hold the most recent "close" price
        self.last_price = {symbol: None for symbol in symbols}
        self.order_placer = None
        self.last_trade = {symbol: None for symbol in symbols}
        self.next_trigger = {symbol: None for symbol in symbols}
        self.current_trends = {symbol: {} for symbol in symbols}

        # For staleness detection: track last trade time
        self.last_trade_time = {symbol: None for symbol in symbols}

    def load_historical_data(self, data_dict):
        """
        Load historical data for each symbol from a dictionary of DataFrames.
        """
        total_symbols = len(data_dict)
        for idx, (symbol, df) in enumerate(data_dict.items(), 1):
            self.data[symbol] = df.reset_index(drop=True)
            if not df.empty:
                self.last_price[symbol] = df.iloc[-1]['close']
                self.logger.debug(
                    f"Loaded historical data for {symbol}, last price: {self.last_price[symbol]}"
                )
            print(f"Loaded historical data for {symbol} ({idx}/{total_symbols})")

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
                'volume': 0.0,
                'trades': 1
            }
        else:
            candle = self.candlesticks[symbol][minute]
            candle['high'] = max(candle['high'], price)
            candle['low'] = min(candle['low'], price)
            candle['close'] = price
            candle['trades'] += 1

        # Update last price and trade time
        self.last_price[symbol] = price
        self.last_trade_time[symbol] = dt

        # Notify trade observers
        for observer in self.trade_observers:
            observer(symbol, price, timestamp, trade_reason)

    def get_current_price(self, symbol):
        """
        Return the *most recent* known price (live trades if available).
        """
        if self.last_price[symbol] is not None:
            return self.last_price[symbol]

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
