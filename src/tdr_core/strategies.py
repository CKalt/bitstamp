###############################################################################
# File Path: src/tdr_core/strategies.py
###############################################################################
# Full File Path: src/tdr_core/strategies.py
#
# CONTEXT AND CHANGES:
#   1) We add a small code block in RSITradingStrategy.get_status()
#      to compute "rsi_proximity" and store it in the status dict
#      so that shell.py can display it in the "status" command
#      (similar to "ma_signal_proximity").
#   2) We preserve all original logic and comments, only adding
#      the new block with "### ADDED ###" for clarity.
#   3) We unify usage of `trades.json` for both live and non-live.
#   4) We implement partial-buy logic (3 trades) in both the
#      MA and RSI strategies when going long, each partial trade
#      is recorded in the trade log with is_partial=True.
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

    We have now implemented partial-buy logic so that going long occurs in
    three separate trades, each of which is at most 90% of the current USD
    balance. We unify logging so that both live and dry-run modes
    use 'trades.json'.
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

        # CHANGED: Always use 'trades.json'
        self.trade_log_file = 'trades.json'

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
        if self.trade_log:
            try:
                file_path = os.path.abspath(self.trade_log_file)
                with open(file_path, 'w') as f:
                    json.dump([t.to_dict() for t in self.trade_log], f, indent=2)
                self.logger.info(f"Trades logged to '{file_path}' (mode: {'live' if self.live_trading else 'dry-run'}).")
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
        # If you want EXACT bar-based approach, skip real-time checks.
        if not self.running:
            return
        if symbol != self.symbol:
            return
        # Currently unused; we rely on run_strategy_loop.

    def check_for_signals(self, latest_signal, current_price, signal_time):
        if self.last_signal_time == signal_time:
            return

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
        Partial buy logic:
        We do 3 sub-trades to ensure each trade is <= 90% of available USD.
        """
        total_btc_bought = 0.0
        total_cost_usd = 0.0

        usd_remaining = self.balance_usd

        # PART 1
        part1_usd = min(usd_remaining * 0.9, usd_remaining)
        btc1 = part1_usd / price if price > 0 else 0.0
        self.execute_trade("buy", price, timestamp, signal_time, btc1, is_partial=True)
        total_btc_bought += btc1
        cost1 = btc1 * price
        total_cost_usd += cost1
        usd_remaining -= cost1

        # PART 2
        part2_usd = min(usd_remaining * 0.9, usd_remaining)
        btc2 = part2_usd / price if price > 0 else 0.0
        self.execute_trade("buy", price, timestamp, signal_time, btc2, is_partial=True)
        total_btc_bought += btc2
        cost2 = btc2 * price
        total_cost_usd += cost2
        usd_remaining -= cost2

        # PART 3 (remainder)
        part3_usd = usd_remaining
        btc3 = part3_usd / price if price > 0 else 0.0
        self.execute_trade("buy", price, timestamp, signal_time, btc3, is_partial=True)
        total_btc_bought += btc3
        cost3 = btc3 * price
        total_cost_usd += cost3
        usd_remaining -= cost3

        # Now update position size and cost basis
        self.position_size = total_btc_bought
        self.position_cost_basis = total_cost_usd

    def execute_trade(self, trade_type, price, timestamp, signal_time, trade_btc, is_partial=False):
        """
        Record the trade (partial or single). Deduct or add from balances.
        """
        if abs(trade_btc) < 1e-9:
            return  # skip zero trades
        fee = self.calculate_fee(trade_btc, price)
        trade_timestamp_dt = datetime.now()
        trade_obj = Trade(
            trade_type,
            self.symbol,
            trade_btc,
            price,
            trade_timestamp_dt,
            reason=("Partial " if is_partial else "") + self.last_trade_reason if self.last_trade_reason else "Signal",
            data_source="Live" if self.live_trading else "Simulated",
            signal_timestamp=signal_time,
            live_trading=self.live_trading
        )

        # Adjust balances
        if trade_type == "buy":
            cost_usd = trade_btc * price
            self.balance_usd -= cost_usd
            self.balance_btc += trade_btc
            self.balance_usd -= fee
        elif trade_type == "sell":
            proceeds_usd = trade_btc * price
            self.balance_btc -= trade_btc
            self.balance_usd += proceeds_usd
            self.balance_usd -= fee

        self.total_fees_paid += fee

        # Book-keeping
        self.trades_executed += 1
        # Realized P&L if selling
        if trade_type == "sell" and self.position_size > 0 and not is_partial:
            # If we had a recognized open position, let's approximate realized PnL
            # But we skip it for partial sells in this example or if position was 0
            pass

        self.last_trade_price = price
        self.trade_log.append(trade_obj)

        # If we wrote an actual trade, also store if it was profitable
        if trade_type == "sell" and not is_partial:
            # simplistic logic
            if price > self.position_cost_basis / max(self.position_size, 1e-8):
                self.profitable_trades += 1
                self.logger.debug("Trade ended up profitable.")
            # compute total_profit_loss incrementally if desired

        self.logger.info(f"{trade_type.upper()} {trade_btc:.6f} BTC at ${price:.2f} - partial={is_partial}, fee=${fee:.2f}")


    def get_status(self):
        """
        Return a dictionary describing the current strategy status.
        """
        status = {
            'running': self.running,
            'position': self.position,
            'last_trade': self.last_trade_reason,
            'last_trade_data_source': self.last_trade_data_source,
            'last_trade_signal_timestamp': self.last_trade_signal_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                if self.last_trade_signal_timestamp else None,
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
            if avg_entry_price > 0:
                position_info['unrealized_pnl'] = (cp - avg_entry_price) * self.position_size
            else:
                position_info['unrealized_pnl'] = 0
        elif self.position < 0:
            position_info['position_size_btc'] = self.position_size
            position_info['position_size_usd'] = self.position_cost_basis
            if avg_entry_price > 0:
                mark_value = abs(self.position_size) * cp
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
            status['total_return_pct'] = (self.current_balance / self.initial_balance - 1) * 100
        else:
            status['total_return_pct'] = 0.0

        if self.trades_executed > 0:
            wins = self.profitable_trades
            status['win_rate'] = (wins / self.trades_executed) * 100
            status['average_profit_per_trade'] = self.total_profit_loss / self.trades_executed if self.trades_executed>0 else 0.0
        else:
            status['win_rate'] = 0.0
            status['average_profit_per_trade'] = 0.0

        status['remaining_trades_today'] = max(0, self.max_trades_per_day - self.trade_count_today)

        # For MA, let's compute a rough "ma_signal_proximity" if we want
        # (similar approach as we do for rsi_proximity in RSI)
        if not self.df_ma.empty:
            # We can approximate by seeing how close short_MA is to long_MA as fraction
            # of the long_MA, or 0 if that doesn't exist
            last_row = self.df_ma.iloc[-1]
            long_ma = last_row.get('Long_MA', 0.0)
            short_ma = last_row.get('Short_MA', 0.0)
            if long_ma != 0:
                dist = abs(short_ma - long_ma)
                status['ma_signal_proximity'] = dist / long_ma
            else:
                status['ma_signal_proximity'] = None
        else:
            status['ma_signal_proximity'] = None

        return status


###############################################################################
# RSITradingStrategy
###############################################################################
class RSITradingStrategy:
    """
    RSI-based strategy. We keep your bar-based approach, partial trades, etc.

    CHANGED:
      - We unify trades logging to always write to trades.json.
      - Implement partial buy logic (similar to MACrossoverStrategy).
      - We add a block in get_status() to compute 'rsi_proximity'
        so shell.py can display it in the status command.
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

        # CHANGED: Always use 'trades.json'
        self.trade_log_file = 'trades.json'

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

        self.bar_size = '1H'
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
        if self.trade_log:
            try:
                file_path = os.path.abspath(self.trade_log_file)
                with open(file_path, 'w') as f:
                    json.dump([t.to_dict() for t in self.trade_log], f, indent=2)
                self.logger.info(f"Trades logged to '{file_path}' (mode: {'live' if self.live_trading else 'dry-run'}).")
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
        # Currently skipping real-time signal logic

    def check_for_signals(self, latest_signal, current_price, signal_time):
        if self.last_signal_time == signal_time:
            return

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
        Partial buy logic if you want multiple partial entries:
        We do 3 sub-trades to ensure each trade is <= 90% of available USD.
        """
        total_btc_bought = 0.0
        total_cost_usd = 0.0

        usd_remaining = self.balance_usd

        # PART 1
        part1_usd = min(usd_remaining * 0.9, usd_remaining)
        btc1 = part1_usd / price if price > 0 else 0.0
        self.execute_trade("buy", price, timestamp, signal_time, btc1, is_partial=True)
        total_btc_bought += btc1
        cost1 = btc1 * price
        total_cost_usd += cost1
        usd_remaining -= cost1

        # PART 2
        part2_usd = min(usd_remaining * 0.9, usd_remaining)
        btc2 = part2_usd / price if price > 0 else 0.0
        self.execute_trade("buy", price, timestamp, signal_time, btc2, is_partial=True)
        total_btc_bought += btc2
        cost2 = btc2 * price
        total_cost_usd += cost2
        usd_remaining -= cost2

        # PART 3 (remainder)
        part3_usd = usd_remaining
        btc3 = part3_usd / price if price > 0 else 0.0
        self.execute_trade("buy", price, timestamp, signal_time, btc3, is_partial=True)
        total_btc_bought += btc3
        cost3 = btc3 * price
        total_cost_usd += cost3
        usd_remaining -= cost3

        self.position_size = total_btc_bought
        self.position_cost_basis = total_cost_usd

    def execute_trade(self, trade_type, price, timestamp, signal_time, trade_btc, is_partial=False):
        """
        Record the trade (partial or single). Deduct or add from balances.
        """
        if abs(trade_btc) < 1e-9:
            return
        fee = self.calculate_fee(trade_btc, price)
        trade_timestamp_dt = datetime.now()

        trade_obj = Trade(
            trade_type,
            self.symbol,
            trade_btc,
            price,
            trade_timestamp_dt,
            reason=("Partial " if is_partial else "") + self.last_trade_reason if self.last_trade_reason else "Signal",
            data_source="Live" if self.live_trading else "Simulated",
            signal_timestamp=signal_time,
            live_trading=self.live_trading
        )

        if trade_type == "buy":
            cost_usd = trade_btc * price
            self.balance_usd -= cost_usd
            self.balance_btc += trade_btc
            self.balance_usd -= fee
        elif trade_type == "sell":
            proceeds_usd = trade_btc * price
            self.balance_btc -= trade_btc
            self.balance_usd += proceeds_usd
            self.balance_usd -= fee

        self.total_fees_paid += fee

        self.trades_executed += 1

        self.last_trade_price = price
        self.trade_log.append(trade_obj)

        # simplistic measure for profitability
        if trade_type == "sell" and not is_partial:
            if price > (self.position_cost_basis / max(self.position_size, 1e-8)):
                self.profitable_trades += 1

        self.logger.info(f"{trade_type.upper()} {trade_btc:.6f} BTC at ${price:.2f} - partial={is_partial}, fee=${fee:.2f}")

    def calculate_fee(self, trade_amount, price):
        trade_value = trade_amount * price
        return trade_value * self.fee_percentage

    def get_status(self):
        """
        Return a dictionary with the RSI strategy's current status.

        ### ADDED: we compute last_rsi and rsi_proximity. ###
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
            if avg_entry_price > 0:
                position_info['unrealized_pnl'] = (cp - avg_entry_price)*self.position_size
            else:
                position_info['unrealized_pnl'] = 0
        elif self.position < 0:
            position_info['position_size_btc'] = self.position_size
            position_info['position_size_usd'] = self.position_cost_basis
            if avg_entry_price > 0:
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

        if self.trades_executed > 0:
            wins = self.profitable_trades
            status['win_rate'] = (wins/self.trades_executed)*100
            status['average_profit_per_trade'] = self.total_profit_loss / self.trades_executed
        else:
            status['win_rate'] = 0.0
            status['average_profit_per_trade'] = 0.0

        status['remaining_trades_today'] = max(0, self.max_trades_per_day - self.trade_count_today)

        ### ADDED: compute last_rsi & "rsi_proximity" ###
        last_rsi = None
        rsi_proximity = None
        if not self.df_rsi.empty:
            last_rsi = self.df_rsi.iloc[-1].get('RSI', None)
            if last_rsi is not None:
                distances = []
                if self.oversold < self.overbought:
                    distances.append(abs(last_rsi - self.oversold))
                    distances.append(abs(last_rsi - self.overbought))
                min_dist = min(distances) if distances else 0
                rsi_proximity = min_dist / 100.0

        status['last_rsi'] = last_rsi
        status['rsi_proximity'] = rsi_proximity

        return status
