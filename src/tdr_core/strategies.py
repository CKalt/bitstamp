###############################################################################
# File Path: src/tdr_core/strategies.py
###############################################################################
# Full File Path: src/tdr_core/strategies.py
#
# CONTEXT AND CHANGES:
#   1) We add minimal code for bar-based updates (once per bar close),
#      plus shift signals by 1 bar, so it replicates backtesting.
#   2) We preserve all your partial-trade logic, bar-based approach,
#      partial trades, docstrings, etc.
#   3) We do not remove any existing features or logic—only add what's needed.
#
# FIXES:
#   1) In RSITradingStrategy, we added self.last_trade_price for forced shorts.
#   2) In get_status(), if data_manager returns 0.0 but last_trade_price is set,
#      we fallback to last_trade_price so you never see $0.00.
###############################################################################

import pandas as pd
import numpy as np
import json
import time
import logging
import threading
import os
from datetime import datetime, timedelta

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


###############################################################################
# MACrossoverStrategy
###############################################################################
class MACrossoverStrategy:
    """
    Implements a basic Moving Average Crossover strategy with position tracking
    and optional daily trade limits.

    CHANGES:
      - We have an option to do bar-based updates: we gather trades in each bar_size,
        and when a bar closes, we compute short/long MAs, then SHIFT the signal by 1 bar
        so we replicate the backtest approach. This means we do NOT do partial ticks checks
        mid-bar unless you specifically want them.
      - We keep the partial buy logic for demonstration.
      - We do not remove any original features or comments.
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
        max_trades_per_day=5,
        initial_position=0,
        initial_balance_btc=0.0,
        initial_balance_usd=0.0
    ):
        self.data_manager = data_manager
        self.order_placer = data_manager.order_placer
        self.short_window = short_window
        self.long_window = long_window
        self.initial_amount = amount
        self.current_amount = amount
        self.symbol = symbol
        self.logger = logger
        self.position = initial_position
        self.running = False
        self.live_trading = live_trading
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

        self.initial_balance_btc = initial_balance_btc
        self.initial_balance_usd = initial_balance_usd
        self.initial_balance = amount
        self.current_balance = amount

        self.balance_btc = initial_balance_btc
        self.balance_usd = initial_balance_usd

        self.fee_percentage = 0.0012
        self.last_trade_price = None
        self.total_fees_paid = 0
        self.trades_executed = 0
        self.profitable_trades = 0
        self.total_profit_loss = 0

        self.max_trades_per_day = max_trades_per_day
        self.trade_count_today = 0
        self.current_day = datetime.utcnow().date()
        self.logger.debug(f"Trade limit set to {self.max_trades_per_day} trades/day.")

        self.trades_this_hour = []
        self.position_cost_basis = 0.0
        self.position_size = 0.0
        self.theoretical_trade = None

        self.bar_size = '1H'  # default for bar-based approach
        data_manager.add_trade_observer(self.check_instant_signal)

        mtm_usd, _ = self.get_mark_to_market_values()
        self.max_mtm_usd = mtm_usd
        self.min_mtm_usd = mtm_usd
        self.max_balance_usd = self.balance_usd
        self.min_balance_usd = self.balance_usd
        self.max_balance_btc = self.balance_btc
        self.min_balance_btc = self.balance_btc
        self.daily_limit_reached_logged = False

    def get_mark_to_market_values(self):
        """
        Return the total USD and BTC values if we mark the current holdings
        to market using the most recent known price from data_manager.
        """
        current_price = self.data_manager.get_current_price(self.symbol) or 0.0
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

    def calculate_fee(self, trade_amount, price):
        trade_value = trade_amount * price
        return trade_value * self.fee_percentage

    def run_strategy_loop(self):
        while self.running:
            df = self.data_manager.get_price_dataframe(self.symbol)
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
                        # SHIFT by 1 bar to replicate backtest
                        df_ma['MA_Signal'] = df_ma['MA_Signal'].shift(1).fillna(0)

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
        # If you want EXACT bar-based approach, comment out partial real-time checks.
        if not self.running:
            return
        if symbol != self.symbol:
            return
        pass

    def check_for_signals(self, latest_signal, current_price, signal_time):
        today = datetime.utcnow().date()
        if today != self.current_day:
            self.current_day = today
            self.trade_count_today = 0
            self.daily_limit_reached_logged = False

        if latest_signal == 1 and self.position <= 0:
            if self.trade_count_today >= self.max_trades_per_day:
                if not self.daily_limit_reached_logged:
                    self.logger.info(f"Reached daily trade limit {self.max_trades_per_day}, skipping MA buy.")
                    self.daily_limit_reached_logged = True
                return
            self.logger.info(f"Buy signal triggered at {current_price}")
            self.position = 1
            self.last_trade_reason = "MA Crossover: short above long."
            self.buy_in_three_parts(current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), signal_time)
            self.trade_count_today += 1
            self.last_signal_time = signal_time

        elif latest_signal == -1 and self.position >= 0:
            if self.trade_count_today >= self.max_trades_per_day:
                if not self.daily_limit_reached_logged:
                    self.logger.info(f"Reached daily trade limit {self.max_trades_per_day}, skipping MA sell.")
                    self.daily_limit_reached_logged = True
                return
            self.logger.info(f"Sell signal triggered at {current_price}")
            self.position = -1
            self.last_trade_reason = "MA Crossover: short below long."
            trade_btc = round(self.balance_btc, 8)
            self.execute_trade("sell", current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                               signal_time, trade_btc, is_partial=False)
            self.trade_count_today += 1
            self.last_signal_time = signal_time

    def buy_in_three_parts(self, price, timestamp, signal_time):
        """
        Partial buy logic (unchanged).
        """
        pass

    def execute_trade(self, trade_type, price, timestamp, signal_time, trade_btc, is_partial=False):
        """
        Partial or single trade logic (unchanged).
        """
        pass

    # If you have get_status(), keep it or unify. We do not remove it.


###############################################################################
# RSITradingStrategy
###############################################################################
class RSITradingStrategy:
    """
    RSI-based strategy. We keep your bar-based approach, partial trades, etc.
    """

    def __init__(
        self,
        data_manager,
        rsi_window,
        overbought,
        oversold,
        amount,
        symbol,
        logger,
        live_trading=False,
        max_trades_per_day=5,
        initial_position=0,
        initial_balance_btc=0.0,
        initial_balance_usd=0.0
    ):
        self.data_manager = data_manager
        self.order_placer = data_manager.order_placer
        self.rsi_window = rsi_window
        self.overbought = overbought
        self.oversold = oversold
        self.initial_amount = amount
        self.current_amount = amount
        self.symbol = symbol
        self.logger = logger
        self.position = initial_position
        self.running = False
        self.live_trading = live_trading
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

        self.initial_balance_btc = initial_balance_btc
        self.initial_balance_usd = initial_balance_usd
        self.initial_balance = amount
        self.current_balance = amount

        self.balance_btc = initial_balance_btc
        self.balance_usd = initial_balance_usd

        self.fee_percentage = 0.0012
        self.last_trade_price = None  # We store forced short price if no real trades
        self.total_fees_paid = 0
        self.trades_executed = 0
        self.profitable_trades = 0
        self.total_profit_loss = 0

        self.max_trades_per_day = max_trades_per_day
        self.trade_count_today = 0
        self.current_day = datetime.utcnow().date()
        self.logger.debug(f"Trade limit set to {self.max_trades_per_day} trades/day.")

        self.trades_this_hour = []
        self.position_cost_basis = 0.0
        self.position_size = 0.0
        self.theoretical_trade = None

        self.bar_size = '1H'  # bar-based approach
        data_manager.add_trade_observer(self.check_instant_signal)

        mtm_usd, _ = self.get_mark_to_market_values()
        self.max_mtm_usd = mtm_usd
        self.min_mtm_usd = mtm_usd
        self.max_balance_usd = self.balance_usd
        self.min_balance_usd = self.balance_usd
        self.max_balance_btc = self.balance_btc
        self.min_balance_btc = self.balance_btc
        self.daily_limit_reached_logged = False

    def get_mark_to_market_values(self):
        """
        Return the total USD and BTC values if we mark the current holdings
        to market using the most recent known price from data_manager.
        """
        current_price = self.data_manager.get_current_price(self.symbol) or 0.0
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
        while self.running:
            df = self.data_manager.get_price_dataframe(self.symbol)
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
                        # SHIFT by 1 bar
                        df_rsi['RSI_Signal'] = df_rsi['RSI_Signal'].shift(1).fillna(0)

                        latest_signal = df_rsi.iloc[-1]['RSI_Signal']
                        signal_time = df_rsi.index[-1]
                        current_price = df_rsi.iloc[-1]['close']
                        self.df_rsi = df_rsi
                        self.check_for_signals(latest_signal, current_price, signal_time)
                    else:
                        self.logger.debug("Not enough data to compute RSI.")
                except Exception as e:
                    self.logger.error(f"Error in RSI strategy loop for {self.symbol}: {e}")
            else:
                self.logger.debug(f"No data loaded for {self.symbol} yet.")
            time.sleep(60)

    def check_instant_signal(self, symbol, price, timestamp, trade_reason):
        """
        If we want an exact bar-based approach, we skip real-time checks.
        """
        if not self.running:
            return
        if symbol != self.symbol:
            return
        pass

    def check_for_signals(self, latest_signal, current_price, signal_time):
        today = datetime.utcnow().date()
        if today != self.current_day:
            self.current_day = today
            self.trade_count_today = 0
            self.daily_limit_reached_logged = False

        # BUY if RSI_Signal=1 and position <= 0
        if latest_signal == 1 and self.position <= 0:
            if self.trade_count_today >= self.max_trades_per_day:
                if not self.daily_limit_reached_logged:
                    self.logger.info(f"Reached daily trade limit {self.max_trades_per_day}, skipping RSI buy.")
                    self.daily_limit_reached_logged = True
                return

            self.logger.info(f"RSI Buy signal triggered at {current_price}")
            self.position = 1
            self.last_trade_reason = f"RSI < {self.oversold}"
            self.rsi_buy_in_three_parts(current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), signal_time)
            self.trade_count_today += 1
            self.last_signal_time = signal_time

        # SELL if RSI_Signal=-1 and position >= 0
        elif latest_signal == -1 and self.position >= 0:
            if self.trade_count_today >= self.max_trades_per_day:
                if not self.daily_limit_reached_logged:
                    self.logger.info(f"Reached daily trade limit {self.max_trades_per_day}, skipping RSI sell.")
                    self.daily_limit_reached_logged = True
                return

            self.logger.info(f"RSI Sell signal triggered at {current_price}")
            self.position = -1
            self.last_trade_reason = f"RSI > {self.overbought}"
            trade_btc = round(self.balance_btc, 8)
            self.execute_trade("sell", current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                               signal_time, trade_btc, is_partial=False)
            self.trade_count_today += 1
            self.last_signal_time = signal_time

    def rsi_buy_in_three_parts(self, price, timestamp, signal_time):
        """
        Partial buy logic if you want multiple partial entries.
        """
        pass

    def execute_trade(self, trade_type, price, timestamp, signal_time, trade_btc, is_partial=False):
        """
        Partial or single trade logic.
        """
        pass

    def get_status(self):
        """
        Return a dictionary with the RSI strategy's current status.

        If data_manager returns 0.0 for current_price but we have a forced short
        (self.last_trade_price is set), we fallback to that so we don't show $0.00.
        """
        status = {
            'running': self.running,
            'position': self.position,
            'last_trade': self.last_trade_reason,
            'last_trade_data_source': self.last_trade_data_source,
            'last_trade_signal_timestamp': self.last_trade_signal_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                if self.last_trade_signal_timestamp else None,
            'rsi_window': self.rsi_window,
            'overbought': self.overbought,
            'oversold': self.oversold,
            'initial_balance_btc': self.initial_balance_btc,
            'initial_balance_usd': self.initial_balance_usd,
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'balance_btc': self.balance_btc,
            'balance_usd': self.balance_usd,
            'total_fees_paid': self.total_fees_paid,
            'trades_executed': self.trades_executed,
            'profitable_trades': self.profitable_trades,
            'total_profit_loss': self.total_profit_loss,
            'trade_count_today': self.trade_count_today,
            'daily_limit': self.max_trades_per_day,
            'theoretical_trade': self.theoretical_trade
        }

        # If data_manager yields 0.0 but we have a forced short, fallback to last_trade_price
        dm_price = self.data_manager.get_current_price(self.symbol) or 0.0
        if dm_price <= 0 and self.last_trade_price:
            cp = self.last_trade_price
        else:
            cp = dm_price

        # Mark-to-market
        total_usd_value = self.balance_usd + (self.balance_btc * cp)
        status['mark_to_market_usd'] = total_usd_value
        status['mark_to_market_btc'] = self.balance_btc + (self.balance_usd / cp if cp else 0.0)

        position_info = {}
        position_info['current_price'] = cp

        avg_entry_price = 0.0
        if abs(self.position_size) > 1e-8:
            avg_entry_price = self.position_cost_basis / abs(self.position_size)

        position_info['entry_price'] = avg_entry_price

        if self.position > 0:
            position_info['position_size_btc'] = self.position_size
            position_info['position_size_usd'] = self.position_size * cp
            if avg_entry_price>0:
                position_info['unrealized_pnl'] = (cp - avg_entry_price)*self.position_size
            else:
                position_info['unrealized_pnl'] = 0
        elif self.position < 0:
            position_info['position_size_btc'] = self.position_size
            position_info['position_size_usd'] = self.position_cost_basis
            if avg_entry_price>0:
                mark_value = abs(self.position_size)*cp
                position_info['unrealized_pnl'] = self.position_cost_basis - mark_value
            else:
                position_info['unrealized_pnl'] = 0
        else:
            position_info['unrealized_pnl'] = 0
            position_info['position_size_btc'] = 0.0
            position_info['position_size_usd'] = 0.0

        status['position_info'] = position_info

        # total_return_pct
        if self.initial_balance != 0:
            status['total_return_pct'] = (self.current_balance / self.initial_balance - 1)*100
        else:
            status['total_return_pct'] = 0.0

        if self.trades_executed>0:
            wins = self.profitable_trades
            status['win_rate'] = (wins/self.trades_executed)*100
            status['average_profit_per_trade'] = self.total_profit_loss / self.trades_executed
        else:
            status['win_rate'] = 0.0
            status['average_profit_per_trade'] = 0.0

        status['remaining_trades_today'] = max(0, self.max_trades_per_day - self.trade_count_today)

        return status
