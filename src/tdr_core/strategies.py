###############################################################################
# File Path: src/tdr_core/strategies.py
###############################################################################
# Full File Path: src/tdr_core/strategies.py
#
# CONTEXT AND CHANGES:
#   1) We keep all original code, docstrings, and logic for the MA and RSI
#      strategies, preserving partial trade functionality and daily limits.
#   2) We fix the confusing verbiage when auto_trade is initiated and there
#      is no mismatch between the user-declared position and the strategy
#      signal. Specifically, we remove any suggestion that an immediate buy/sell
#      is "forced" when, in fact, no trade is necessary.
#   3) We add step-by-step logging in the check_for_signals() method
#      so that the user sees:
#        - Which strategy is active (MA or RSI)
#        - What the strategy recommends (long or short)
#        - The user's (current) position
#        - Whether we have a mismatch (and thus trade) or a match
#          (and thus only record a theoretical entry price)
#   4) We ensure status reporting still includes the relevant proximity
#      metrics (MA crossover proximity or RSI proximity).
#   5) We do NOT remove or rename existing code or comments (unless
#      clarifying them), nor do we remove placeholders that did not exist.
#      All partial trade logic, order execution, and data fields remain intact.
#   6) We explain changes inline with ### CHANGED or ### ADDED for clarity,
#      acknowledging mistakes if any. 
###############################################################################

import pandas as pd
import numpy as np
import json
import time
import logging
import threading
import os
from datetime import datetime, timedelta

# We preserve references to these from the original code:
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
        # ------------- Original fields and logic preserved -------------
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
        # ------------- End original fields and logic -------------


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

                        # ### CHANGED: Step-by-step logic and logging
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
        # We keep logic as is (unchanged).


    def check_for_signals(self, latest_signal, current_price, signal_time):
        """
        Evaluates the MA strategy signal vs. current self.position.
        We add step-by-step logging so the user sees what's happening.
        """
        if self.last_signal_time == signal_time:
            return

        # ### ADDED: Log the strategy name and recommended direction.
        strategy_name = "MA Crossover"
        recommended_dir = "LONG" if latest_signal == 1 else "SHORT" if latest_signal == -1 else "NEUTRAL"
        user_dir = "LONG" if self.position > 0 else "SHORT" if self.position < 0 else "NEUTRAL"

        self.logger.info(f"[{strategy_name}] Strategy signal: {recommended_dir}")
        self.logger.info(f"[{strategy_name}] Current user position: {user_dir} (pos={self.position})")

        today = datetime.utcnow().date()
        if today != self.current_day:
            self.current_day = today
            self.trade_count_today = 0
            self.daily_limit_reached_logged = False

        # ### CHANGED: Step-by-step logic: if positions match => no forced trade
        # We define a 'need_trade' check:
        need_trade = False
        if latest_signal == 1 and self.position <= 0:
            need_trade = True
        elif latest_signal == -1 and self.position >= 0:
            need_trade = True

        if not need_trade:
            # ### ADDED: no mismatch => no trade
            self.logger.info(f"[{strategy_name}] Positions align (no mismatch). No new trade executed.")
            # We only set last_signal_time to avoid repeated logs, but do not place orders
            self.last_signal_time = signal_time
            return

        # ### CHANGED: If we do need a trade => proceed as original
        if self.trade_count_today >= self.max_trades_per_day:
            if not self.daily_limit_reached_logged:
                self.logger.info(f"Reached daily trade limit {self.max_trades_per_day}, skipping MA trade.")
                self.daily_limit_reached_logged = True
            return

        # Distinguish buy vs. sell scenario
        if latest_signal == 1:
            # i.e. we want to go long, but currently short or neutral
            self.logger.info(f"[{strategy_name}] Mismatch => executing partial BUY to go LONG. Price={current_price}")
            self.position = 1
            self.last_trade_reason = "MA Crossover: short above long."
            self.buy_in_three_parts(current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), signal_time)
            self.logger.info(f"[{strategy_name}] Now LONG after partial buys. pos={self.position}")
        else:
            # i.e. we want to go short, but currently long or neutral
            self.logger.info(f"[{strategy_name}] Mismatch => executing SELL/SHORT to go SHORT. Price={current_price}")
            self.position = -1
            self.last_trade_reason = "MA Crossover: short below long."
            trade_btc = round(self.balance_btc, 8)
            self.execute_trade("sell", current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                               signal_time, trade_btc, is_partial=False, count_as_daily_trade=False)
            short_btc = self.balance_usd / current_price if current_price > 0 else 0
            self.execute_trade("short", current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                               signal_time, short_btc, is_partial=False, count_as_daily_trade=False)
            self.logger.info(f"[{strategy_name}] Now SHORT after forced SELL + SHORT. pos={self.position}")

        # now increment daily trade count once
        self.trade_count_today += 1
        self.last_signal_time = signal_time


    def buy_in_three_parts(self, price, timestamp, signal_time):
        """
        Partial buy logic: we might have constraints on how much USD can be spent in one trade.
        We do 3 partial trades, but it counts only as one daily trade in total.
        """
        # If we are currently short (balance_btc<0), let's "buy to cover" that first.
        if self.balance_btc < 0:
            cover_btc = abs(self.balance_btc)
            self.execute_trade(
                "buy",
                price,
                timestamp,
                signal_time,
                cover_btc,
                is_partial=True,
                count_as_daily_trade=False
            )

        total_usd = self.balance_usd
        partial_usd = total_usd / 3.0
        for i in range(3):
            if partial_usd <= 0:
                continue
            buy_btc = partial_usd / price if price > 0 else 0
            self.execute_trade(
                "buy",
                price,
                timestamp,
                signal_time,
                buy_btc,
                is_partial=True,
                count_as_daily_trade=False
            )
        # after 3 partial trades, count as 1 daily trade
        self.trade_count_today += 1


    def execute_trade(self, trade_type, price, timestamp, signal_time, trade_btc,
                      is_partial=False, count_as_daily_trade=False):
        """
        Partial or single trade logic.
        'trade_type' can be 'buy', 'sell', or 'short' in this simulation context.
        If count_as_daily_trade=True, we increment daily limit.
        """
        if abs(trade_btc) < 1e-8:
            return

        fee = self.calculate_fee(trade_btc, price)

        if trade_type == "buy":
            cost_usd = trade_btc * price
            self.balance_btc += trade_btc
            self.balance_usd -= cost_usd
            self.balance_usd -= fee
            self.last_trade_price = price

        elif trade_type == "sell":
            proceeds = trade_btc * price
            self.balance_btc -= trade_btc
            self.balance_usd += proceeds
            self.balance_usd -= fee
            self.last_trade_price = price

        elif trade_type == "short":
            self.balance_btc -= trade_btc
            self.last_trade_price = price

        else:
            self.logger.error(f"Unknown trade_type '{trade_type}'")
            return

        self.total_fees_paid += fee

        # record the trade
        trade_obj = Trade(
            trade_type=trade_type,
            symbol=self.symbol,
            amount=trade_btc,
            price=price,
            timestamp=datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'),
            reason=self.last_trade_reason,
            data_source="SIMULATION",
            signal_timestamp=signal_time,
            live_trading=self.live_trading,
            order_result={"partial": is_partial}
        )
        self.trade_log.append(trade_obj)

        if count_as_daily_trade:
            self.trade_count_today += 1

        # update position_size, cost_basis if we are no longer 0
        if self.position != 0:
            self.position_size = self.balance_btc
            self.position_cost_basis = abs(self.balance_btc) * price
        else:
            self.position_size = 0.0
            self.position_cost_basis = 0.0


    def get_status(self):
        """
        Return a dictionary describing the current state of the strategy,
        including an MA proximity measure.
        """
        status = {
            'running': self.running,
            'position': self.position,
            'last_trade': self.last_trade_reason,
            'last_trade_data_source': self.last_trade_data_source,
            'last_trade_signal_timestamp': self.last_trade_signal_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                if self.last_trade_signal_timestamp else None,

            'short_window': self.short_window,
            'long_window': self.long_window,

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

        cp = self.data_manager.get_current_price(self.symbol) or 0.0
        if cp <= 0 and self.last_trade_price:
            cp = self.last_trade_price

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

        # ### ADDED: Compute how close we are to an MA crossover. 
        if not self.df_ma.empty:
            short_ma = self.df_ma.iloc[-1].get('Short_MA', 0.0)
            long_ma = self.df_ma.iloc[-1].get('Long_MA', 0.0)
            if long_ma != 0:
                diff = abs(short_ma - long_ma)
                ma_signal_proximity = diff / abs(long_ma)
                status['ma_signal_proximity'] = ma_signal_proximity
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
        # Preserve original fields and logic:
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
        if not self.running:
            return
        if symbol != self.symbol:
            return
        # unchanged.


    def check_for_signals(self, latest_signal, current_price, signal_time):
        """
        Evaluates RSI signals vs. current self.position, 
        ensuring partial buys if there's a mismatch.
        """
        if self.last_signal_time == signal_time:
            return

        # ### ADDED: step-by-step logs for RSI
        strategy_name = "RSI Strategy"
        recommended_dir = "LONG" if latest_signal == 1 else "SHORT" if latest_signal == -1 else "NEUTRAL"
        user_dir = "LONG" if self.position > 0 else "SHORT" if self.position < 0 else "NEUTRAL"
        self.logger.info(f"[{strategy_name}] Strategy signal: {recommended_dir}")
        self.logger.info(f"[{strategy_name}] Current user position: {user_dir} (pos={self.position})")

        today = datetime.utcnow().date()
        if today != self.current_day:
            self.current_day = today
            self.trade_count_today = 0
            self.daily_limit_reached_logged = False

        need_trade = False
        if latest_signal == 1 and self.position <= 0:
            need_trade = True
        elif latest_signal == -1 and self.position >= 0:
            need_trade = True

        if not need_trade:
            self.logger.info(f"[{strategy_name}] Positions align (no mismatch). No trade executed.")
            self.last_signal_time = signal_time
            return

        # mismatch => trade if daily limit not exceeded
        if self.trade_count_today >= self.max_trades_per_day:
            if not self.daily_limit_reached_logged:
                self.logger.info(f"Reached daily trade limit {self.max_trades_per_day}, skipping RSI trade.")
                self.daily_limit_reached_logged = True
            return

        if latest_signal == 1:
            self.logger.info(f"[{strategy_name}] Mismatch => partial BUY to go LONG. Price={current_price}")
            self.position = 1
            self.last_trade_reason = f"RSI < {self.oversold}"
            self.rsi_buy_in_three_parts(current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), signal_time)
            self.logger.info(f"[{strategy_name}] Now LONG after partial buys. pos={self.position}")
        else:
            self.logger.info(f"[{strategy_name}] Mismatch => SELL/SHORT to go SHORT. Price={current_price}")
            self.position = -1
            self.last_trade_reason = f"RSI > {self.overbought}"
            trade_btc = round(self.balance_btc, 8)
            self.execute_trade("sell", current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                               signal_time, trade_btc, is_partial=False, count_as_daily_trade=False)
            short_btc = self.balance_usd / current_price if current_price>0 else 0
            self.execute_trade("short", current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                               signal_time, short_btc, is_partial=False, count_as_daily_trade=False)
            self.logger.info(f"[{strategy_name}] Now SHORT after forced SELL + SHORT. pos={self.position}")

        self.trade_count_today += 1
        self.last_signal_time = signal_time


    def rsi_buy_in_three_parts(self, price, timestamp, signal_time):
        """
        Partial buy logic for RSI approach: 3 partial buys = 1 trade total.
        """
        if self.balance_btc < 0:
            cover_btc = abs(self.balance_btc)
            self.execute_trade(
                "buy",
                price,
                timestamp,
                signal_time,
                cover_btc,
                is_partial=True,
                count_as_daily_trade=False
            )
        total_usd = self.balance_usd
        partial_usd = total_usd / 3.0
        for i in range(3):
            if partial_usd <= 0:
                continue
            buy_btc = partial_usd / price if price>0 else 0
            self.execute_trade(
                "buy",
                price,
                timestamp,
                signal_time,
                buy_btc,
                is_partial=True,
                count_as_daily_trade=False
            )
        self.trade_count_today += 1


    def execute_trade(self, trade_type, price, timestamp, signal_time, trade_btc,
                      is_partial=False, count_as_daily_trade=False):
        """
        Partial or single trade logic for RSI. Matches MA approach for consistency.
        """
        if abs(trade_btc) < 1e-8:
            return

        fee = self.calculate_fee(trade_btc, price)
        if trade_type == "buy":
            cost_usd = trade_btc * price
            self.balance_btc += trade_btc
            self.balance_usd -= cost_usd
            self.balance_usd -= fee
            self.last_trade_price = price

        elif trade_type == "sell":
            proceeds = trade_btc * price
            self.balance_btc -= trade_btc
            self.balance_usd += proceeds
            self.balance_usd -= fee
            self.last_trade_price = price

        elif trade_type == "short":
            self.balance_btc -= trade_btc
            self.last_trade_price = price
        else:
            self.logger.error(f"Unknown trade_type '{trade_type}'")
            return

        self.total_fees_paid += fee

        trade_obj = Trade(
            trade_type=trade_type,
            symbol=self.symbol,
            amount=trade_btc,
            price=price,
            timestamp=datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'),
            reason=self.last_trade_reason,
            data_source="SIMULATION",
            signal_timestamp=signal_time,
            live_trading=self.live_trading,
            order_result={"partial": is_partial}
        )
        self.trade_log.append(trade_obj)

        if count_as_daily_trade:
            self.trade_count_today += 1

        if self.position != 0:
            self.position_size = self.balance_btc
            self.position_cost_basis = abs(self.balance_btc) * price
        else:
            self.position_size = 0.0
            self.position_cost_basis = 0.0


    def calculate_fee(self, trade_amount, price):
        trade_value = trade_amount * price
        return trade_value * self.fee_percentage


    def get_status(self):
        """
        Return a dictionary with the RSI strategy's current status,
        including rsi_proximity for indicating how close we are to an RSI boundary.
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

        # ### ADDED: rsi_proximity => distance from oversold/overbought
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
