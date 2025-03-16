###############################################################################
# File Path: src/tdr_core/strategies.py
###############################################################################
# CHANGES to ensure that "go long" uses the 3 partial trades of 90% approach.
# We do minimal changes inside the old buy_in_three_parts / rsi_buy_in_three_parts
# methods to call the new partial_buy_3x_90pct so we meet the user's new requirement.
###############################################################################

import pandas as pd
import numpy as np
import json
import time
import logging
import threading
import os
from datetime import datetime, timedelta

from strategies.base_strategy import BaseStrategy
from indicators.technical_indicators import (
    ensure_datetime_index,
    add_moving_averages,
    generate_ma_signals,
    calculate_rsi,
    generate_rsi_signals,
    calculate_bollinger_bands,
    generate_bollinger_band_signals,
    calculate_macd,
    generate_macd_signals
)
from tdr_core.trade import Trade

class MACrossoverStrategy(BaseStrategy):
    """
    Basic Moving Average Crossover strategy with partial buy logic, daily limits, etc.
    """

    def __init__(
        self,
        data_manager,
        logger,
        live_trading=False,
        max_trades_per_day=5,
        initial_position=0,
        initial_balance_btc=0.0,
        initial_balance_usd=0.0,
        short_window=12,
        long_window=36,
        fee_percentage=0.0012,
        **kwargs
    ):
        super().__init__(
            data_manager=data_manager,
            logger=logger,
            live_trading=live_trading,
            max_trades_per_day=max_trades_per_day,
            initial_position=initial_position,
            initial_balance_btc=initial_balance_btc,
            initial_balance_usd=initial_balance_usd,
            fee_percentage=fee_percentage,
            **kwargs
        )
        self.short_window = short_window
        self.long_window = long_window

        self.initial_amount = kwargs.get("amount", 0.0)
        self.current_amount = self.initial_amount

        self.trade_log = []
        if self.live_trading:
            self.trade_log_file = 'trades.json'
        else:
            self.trade_log_file = 'non-live-trades.json'

        self.last_signal_time = None
        self.last_trade_reason = None
        self.last_trade_data_source = None
        self.last_trade_signal_timestamp = None
        self.next_trigger = None
        self.current_trends = {}
        self.df_ma = pd.DataFrame()
        self.strategy_start_time = datetime.now()

        self.position_cost_basis = 0.0
        self.position_size = 0.0
        self.theoretical_trade = None

        self.bar_size = '1H'
        data_manager.add_trade_observer(self.check_instant_signal)

        mtm_usd, _ = self.get_mark_to_market_values()
        self.max_mtm_usd = mtm_usd
        self.min_mtm_usd = mtm_usd
        self.max_balance_usd = self.balance_usd
        self.min_balance_usd = self.balance_usd
        self.max_balance_btc = self.balance_btc
        self.min_balance_btc = self.balance_btc

    def get_mark_to_market_values(self):
        current_price = self.data_manager.get_current_price('btcusd') or 0.0
        total_usd_value = self.balance_usd + (self.balance_btc * current_price)
        total_btc_value = self.balance_btc + (self.balance_usd / current_price if current_price else 0.0)
        return total_usd_value, total_btc_value

    def start(self):
        self.running = True
        self.strategy_thread = threading.Thread(target=self.run_strategy_loop, daemon=True)
        self.strategy_thread.start()
        self.logger.info("Strategy loop started.")

    def stop(self):
        self.running = False
        self.logger.info("Strategy loop stopped.")
        if not self.live_trading and self.trade_log:
            try:
                file_path = os.path.abspath(self.trade_log_file)
                with open(file_path, 'w') as f:
                    json.dump([t.to_dict() for t in self.trade_log], f, indent=2)
                self.logger.info(f"Trades logged to '{file_path}' (dry-run mode).")
            except Exception as e:
                self.logger.error(f"Failed to write trades: {e}")

    def run_strategy_loop(self):
        import time
        while self.running:
            df = self.data_manager.get_price_dataframe('btcusd')
            if not df.empty:
                try:
                    df = ensure_datetime_index(df)
                    df_resampled = df.resample(self.bar_size).agg({
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
                        df_ma = add_moving_averages(df_resampled.copy(), self.short_window, self.long_window, price_col='close')
                        df_ma = generate_ma_signals(df_ma)
                        df_ma['MA_Signal'] = df_ma['MA_Signal'].shift(1).fillna(0)

                        latest_signal = df_ma.iloc[-1]['MA_Signal']
                        signal_time = df_ma.index[-1]
                        current_price = df_ma.iloc[-1]['close']

                        self.next_trigger = self.determine_next_trigger(df_ma)
                        self.current_trends = self.get_current_trends(df_ma)
                        self.df_ma = df_ma

                        self.check_for_signals(latest_signal, current_price, signal_time)
                    else:
                        self.logger.debug("Not enough data to compute MAs.")
                except Exception as e:
                    self.logger.error(f"Error in strategy loop for MACrossoverStrategy: {e}")
            else:
                self.logger.debug(f"No data loaded for MACrossoverStrategy yet.")
            time.sleep(60)

    def determine_next_trigger(self, df_ma):
        if len(df_ma) < 2:
            return None
        last_signal = df_ma.iloc[-1]['MA_Signal']
        prev_signal = df_ma.iloc[-2]['MA_Signal'] if len(df_ma) >= 2 else 0
        if last_signal != prev_signal:
            if last_signal == 1:
                return "Next trigger: Potential SELL if short crosses below long."
            elif last_signal == -1:
                return "Next trigger: Potential BUY if short crosses above long."
        return "Next trigger: Awaiting next crossover signal."

    def get_current_trends(self, df_ma):
        if len(df_ma) < 2:
            return {}
        short_ma_curr = df_ma.iloc[-1]['Short_MA']
        short_ma_prev = df_ma.iloc[-2]['Short_MA']
        long_ma_curr = df_ma.iloc[-1]['Long_MA']
        long_ma_prev = df_ma.iloc[-2]['Long_MA']

        short_ma_slope = short_ma_curr - short_ma_prev
        long_ma_slope  = long_ma_curr - long_ma_prev

        return {
            'Short_MA_Slope': 'Upwards' if short_ma_slope > 0 else 'Downwards',
            'Long_MA_Slope': 'Upwards' if long_ma_slope > 0 else 'Downwards',
            'Price_Trend': 'Bullish' if short_ma_curr > long_ma_curr else 'Bearish',
            'Trend_Strength': abs(short_ma_curr - long_ma_curr) / long_ma_curr * 100 if long_ma_curr else 0
        }

    def check_instant_signal(self, symbol, price, timestamp, trade_reason):
        if not self.running:
            return
        if symbol != 'btcusd':
            return
        # bar-based approach used, no real-time action.

    def check_for_signals(self, latest_signal, current_price, signal_time):
        if self.last_signal_time == signal_time:
            return
        if not self.check_daily_limit():
            return

        # If buy signal & position <= 0 => partial buy
        if latest_signal == 1 and self.position <= 0:
            self.logger.info(f"Buy signal triggered at {current_price}")
            self.position = 1
            self.last_trade_reason = "MA Crossover: short above long."
            self.buy_in_three_parts(current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), signal_time)
            self.last_signal_time = signal_time

        # If sell signal & position >= 0 => single trade
        elif latest_signal == -1 and self.position >= 0:
            if not self.check_daily_limit():
                return
            self.logger.info(f"Sell signal triggered at {current_price}")
            self.position = -1
            trade_btc = round(self.balance_btc, 8)
            self.execute_trade("sell", current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                               signal_time, trade_btc, is_partial=False)
            # This single sell increments daily limit
            self.trade_count_today += 1
            self.last_signal_time = signal_time

    def buy_in_three_parts(self, price, timestamp_str, signal_time):
        """
        Updated to call the new partial_buy_3x_90pct from the base class.
        We preserve the name and comments, but the logic is replaced so that
        we do the 3 partial trades each for 90% of remaining USD.
        """
        self.logger.info(
            f"(MACrossoverStrategy) Using partial_buy_3x_90pct for buy_in_three_parts at price={price:.2f}"
        )
        self.partial_buy_3x_90pct(
            price=price,
            timestamp_str=timestamp_str,
            signal_time=signal_time
        )

    def execute_trade(self, trade_type, price, timestamp, signal_time, trade_btc, is_partial=False):
        super().execute_trade(
            trade_type=trade_type,
            price=price,
            timestamp=timestamp,
            signal_time=signal_time,
            trade_btc=trade_btc,
            is_partial=is_partial
        )

class RSITradingStrategy(BaseStrategy):
    """
    RSI-based strategy with partial buy logic, daily limits, etc.
    """

    def __init__(
        self,
        data_manager,
        logger,
        live_trading=False,
        max_trades_per_day=5,
        initial_position=0,
        initial_balance_btc=0.0,
        initial_balance_usd=0.0,
        rsi_window=14,
        overbought=70,
        oversold=30,
        **kwargs
    ):
        super().__init__(
            data_manager=data_manager,
            logger=logger,
            live_trading=live_trading,
            max_trades_per_day=max_trades_per_day,
            initial_position=initial_position,
            initial_balance_btc=initial_balance_btc,
            initial_balance_usd=initial_balance_usd,
            **kwargs
        )
        self.rsi_window = rsi_window
        self.overbought = overbought
        self.oversold = oversold

        self.initial_amount = kwargs.get("amount", 0.0)
        self.current_amount = self.initial_amount
        self.trade_log = []

        if self.live_trading:
            self.trade_log_file = 'trades.json'
        else:
            self.trade_log_file = 'non-live-trades.json'

        self.last_signal_time = None
        self.last_trade_reason = None
        self.last_trade_data_source = None
        self.last_trade_signal_timestamp = None
        self.next_trigger = None
        self.df_rsi = pd.DataFrame()
        self.strategy_start_time = datetime.now()

        self.bar_size = '1H'

        mtm_usd, _ = self.get_mark_to_market_values()
        self.max_mtm_usd = mtm_usd
        self.min_mtm_usd = mtm_usd
        self.max_balance_usd = self.balance_usd
        self.min_balance_usd = self.balance_usd
        self.max_balance_btc = self.balance_btc
        self.min_balance_btc = self.balance_btc
        self.daily_limit_reached_logged = False

        data_manager.add_trade_observer(self.check_instant_signal)

    def get_mark_to_market_values(self):
        current_price = self.data_manager.get_current_price('btcusd') or 0.0
        total_usd_value = self.balance_usd + (self.balance_btc * current_price)
        total_btc_value = self.balance_btc + (self.balance_usd / current_price if current_price else 0.0)
        return total_usd_value, total_btc_value

    def start(self):
        self.running = True
        self.strategy_thread = threading.Thread(target=self.run_strategy_loop, daemon=True)
        self.strategy_thread.start()
        self.logger.info("RSI strategy loop started.")

    def stop(self):
        self.running = False
        self.logger.info("RSI strategy loop stopped.")
        if not self.live_trading and self.trade_log:
            try:
                file_path = os.path.abspath(self.trade_log_file)
                with open(file_path, 'w') as f:
                    json.dump([t.to_dict() for t in self.trade_log], f, indent=2)
                self.logger.info(f"Trades logged to '{file_path}' (dry-run mode).")
            except Exception as e:
                self.logger.error(f"Failed to write trades: {e}")

    def run_strategy_loop(self):
        import time
        while self.running:
            df = self.data_manager.get_price_dataframe('btcusd')
            if not df.empty:
                try:
                    df = ensure_datetime_index(df)
                    df_resampled = df.resample(self.bar_size).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum',
                        'trades': 'sum',
                        'timestamp': 'last',
                        'source': 'last'
                    }).dropna()

                    if len(df_resampled) >= self.rsi_window:
                        df_rsi = df_resampled.copy()
                        df_rsi = calculate_rsi(df_rsi, window=self.rsi_window, price_col='close')
                        df_rsi['RSI_Signal'] = 0
                        df_rsi.loc[df_rsi['RSI'] < self.oversold, 'RSI_Signal'] = 1
                        df_rsi.loc[df_rsi['RSI'] > self.overbought, 'RSI_Signal'] = -1
                        df_rsi['RSI_Signal'] = df_rsi['RSI_Signal'].shift(1).fillna(0)

                        latest_signal = df_rsi.iloc[-1]['RSI_Signal']
                        signal_time = df_rsi.index[-1]
                        current_price = df_rsi.iloc[-1]['close']
                        self.df_rsi = df_rsi
                        self.check_for_signals(latest_signal, current_price, signal_time)
                    else:
                        self.logger.debug("Not enough data to compute RSI.")
                except Exception as e:
                    self.logger.error(f"Error in RSI strategy loop for RSI: {e}")
            else:
                self.logger.debug("No data loaded for RSI strategy yet.")
            time.sleep(60)

    def check_instant_signal(self, symbol, price, timestamp, trade_reason):
        if not self.running:
            return
        if symbol != 'btcusd':
            return
        # Skip real-time checks.

    def check_for_signals(self, latest_signal, current_price, signal_time):
        if self.last_signal_time == signal_time:
            return
        if not self.check_daily_limit():
            return

        if latest_signal == 1 and self.position <= 0:
            self.logger.info(f"RSI Buy signal triggered at {current_price}")
            self.position = 1
            self.last_trade_reason = f"RSI < {self.oversold}"
            self.rsi_buy_in_three_parts(current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), signal_time)
            self.last_signal_time = signal_time

        elif latest_signal == -1 and self.position >= 0:
            if not self.check_daily_limit():
                return
            self.logger.info(f"RSI Sell signal triggered at {current_price}")
            self.position = -1
            trade_btc = round(self.balance_btc, 8)
            self.execute_trade("sell", current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                               signal_time, trade_btc, is_partial=False)
            self.trade_count_today += 1
            self.last_signal_time = signal_time

    def rsi_buy_in_three_parts(self, price, timestamp_str, signal_time):
        """
        Updated to call partial_buy_3x_90pct so that going long is done
        in 3 partial trades, each using 90% of remaining USD.
        """
        self.logger.info(
            f"(RSITradingStrategy) Using partial_buy_3x_90pct for rsi_buy_in_three_parts at price={price:.2f}"
        )
        self.partial_buy_3x_90pct(
            price=price,
            timestamp_str=timestamp_str,
            signal_time=signal_time
        )

    def execute_trade(self, trade_type, price, timestamp, signal_time, trade_btc, is_partial=False):
        super().execute_trade(
            trade_type=trade_type,
            price=price,
            timestamp=timestamp,
            signal_time=signal_time,
            trade_btc=trade_btc,
            is_partial=is_partial
        )
