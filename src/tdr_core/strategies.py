# src/tdr_core/strategies.py
# ----------------------------------------------------------------------------
# FULL FILE PATH: src/tdr_core/strategies.py
# ----------------------------------------------------------------------------
# CHANGES MADE:
#   1) Introduced position-based cost-basis tracking to fix the "entry price" and 
#      "unrealized PnL" inaccuracies in status.
#   2) Split trade logging into 'trades.json' (for live mode) vs. 
#      'non-live-trades.json' (for dry-run).
#   3) Marked changes with (CHANGED) / (NEW).
#   4) Preserved all original comments and logic except where necessary to 
#      incorporate cost-basis logic.
# ----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import json
import time
import logging
import threading
import os  # (CHANGED) Added to ensure file I/O for the new non-live trades file
from datetime import datetime, timedelta

# We reuse indicators:
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

        # (CHANGED) Decide which trades file to use (live vs. non-live).
        #           This is to keep real trades separate from the dry-run trades.
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

        # ---------------------------------------------------------------------
        # Storing the user's "initial BTC balance" and "initial USD balance"
        # separately. We also keep an internal 'amount' but that is used
        # for the internal P&L logic.
        # ---------------------------------------------------------------------
        self.initial_balance_btc = initial_balance_btc
        self.initial_balance_usd = initial_balance_usd

        # For backward-compat with existing P&L logic:
        self.initial_balance = amount
        self.current_balance = amount

        # Actual live/dry-run wallet balances:
        self.balance_btc = initial_balance_btc
        self.balance_usd = initial_balance_usd

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

        self.trades_this_hour = []

        # (CHANGED) Removed `self.initial_fill_price = 0.0`. 
        #           We'll track cost basis more accurately (below).

        # (NEW) Fields for cost-basis tracking (averaged if multiple partial buys):
        #       This helps accurately compute entry price & PnL.
        self.position_cost_basis = 0.0   # total USD spent if long
        self.position_size       = 0.0   # total BTC in the open position

        # Register real-time callback
        data_manager.add_trade_observer(self.check_instant_signal)

        # For staleness detection, etc. (unchanged)
        mtm_usd, _ = self.get_mark_to_market_values()
        self.max_mtm_usd = mtm_usd
        self.min_mtm_usd = mtm_usd
        self.max_balance_usd = self.balance_usd
        self.min_balance_usd = self.balance_usd
        self.max_balance_btc = self.balance_btc
        self.min_balance_btc = self.balance_btc

    def _clean_up_hourly_trades(self):
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        self.trades_this_hour = [t for t in self.trades_this_hour if t > one_hour_ago]

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
                # (CHANGED) We are now using `self.trade_log_file` which 
                #           is 'non-live-trades.json' if not in live_trading.
                with open(file_path, 'w') as f:
                    json.dump([t.to_dict() for t in self.trade_log], f, indent=2)
                self.logger.info(f"Trades logged to '{file_path}'")
            except Exception as e:
                self.logger.error(f"Failed to write trades: {e}")

    def calculate_fee(self, trade_amount, price):
        trade_value = trade_amount * price
        return trade_value * self.fee_percentage

    def run_strategy_loop(self):
        """
        Strategy loop that checks for signals every minute.
        """
        while self.running:
            df = self.data_manager.get_price_dataframe(self.symbol)
            if not df.empty:
                try:
                    df = ensure_datetime_index(df)
                    df_resampled = df.resample('1H').agg({
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

                        latest_signal = df_ma.iloc[-1]['MA_Signal']
                        signal_time = df_ma.index[-1]
                        current_price = df_ma.iloc[-1]['close']
                        signal_source = df_ma.iloc[-1]['source']

                        self.next_trigger = self.determine_next_trigger(df_ma)
                        self.current_trends = self.get_current_trends(df_ma)
                        self.df_ma = df_ma

                        # Check the signals (crossover logic)
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
        Return text describing the potential next trigger, if signals changed.
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
            'Trend_Strength': abs(short_ma_curr - long_ma_curr) / long_ma_curr * 100 if long_ma_curr else 0
        }

    def check_instant_signal(self, symbol, price, timestamp, trade_reason):
        """
        Real-time callback for each new trade. If there's a new crossover, act.
        """
        if not self.running:
            return
        if symbol != self.symbol:
            return

        df_live = self.data_manager.get_price_dataframe(symbol)
        if df_live.empty:
            return

        df_live = ensure_datetime_index(df_live)
        if len(df_live) < self.long_window:
            return

        df_ma = df_live.copy()
        df_ma['Short_MA'] = df_ma['close'].rolling(self.short_window).mean()
        df_ma['Long_MA']  = df_ma['close'].rolling(self.long_window).mean()
        df_ma.dropna(inplace=True)
        if df_ma.empty:
            return

        latest = df_ma.iloc[-1]
        short_ma_now = latest['Short_MA']
        long_ma_now = latest['Long_MA']
        signal_now = 1 if short_ma_now > long_ma_now else -1

        if len(df_ma) < 2:
            return
        prev = df_ma.iloc[-2]
        prev_signal = 1 if prev['Short_MA'] > prev['Long_MA'] else -1
        if signal_now == prev_signal:
            return

        signal_time = df_ma.index[-1]
        self.check_for_signals(signal_now, price, signal_time)

    def check_for_signals(self, latest_signal, current_price, signal_time):
        """
        If the new MA signal differs from our current position, execute trades.
        Now also controls the daily trade-limit check, so that multi-part buys
        only count as a single daily trade.
        """

        today = datetime.utcnow().date()
        if today != self.current_day:
            self.current_day = today
            self.trade_count_today = 0
            self.logger.debug("New day, resetting daily trade count.")

        if self.last_signal_time == signal_time:
            return

        # If we see a BUY signal
        if latest_signal == 1 and self.position <= 0:
            if self.trade_count_today >= self.max_trades_per_day:
                self.logger.info(
                    f"Reached daily trade limit {self.max_trades_per_day}, skipping trade."
                )
                return

            self.logger.info(f"Buy signal triggered at {current_price}")
            self.position = 1
            self.last_trade_reason = "MA Crossover: short above long."
            self.buy_in_three_parts(
                current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), signal_time
            )

            self.trade_count_today += 1
            self.last_signal_time = signal_time

        # If we see a SELL signal
        elif latest_signal == -1 and self.position >= 0:
            if self.trade_count_today >= self.max_trades_per_day:
                self.logger.info(
                    f"Reached daily trade limit {self.max_trades_per_day}, skipping trade."
                )
                return

            self.logger.info(f"Sell signal triggered at {current_price}")
            self.position = -1
            self.last_trade_reason = "MA Crossover: short below long."
            trade_btc = self.balance_btc
            trade_btc = round(trade_btc, 8)
            self.execute_trade(
                "sell",
                current_price,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                signal_time,
                trade_btc
            )

            self.trade_count_today += 1
            self.last_signal_time = signal_time

    def buy_in_three_parts(self, price, timestamp, signal_time):
        """
        Simulate a multi-part buy so we can keep within Bitstamp's 90% rule,
        but still only count as a single daily trade.
        """
        partial_btc_1 = self.get_89pct_btc_of_usd(price)
        self.execute_trade("buy", price, timestamp, signal_time, partial_btc_1)

        partial_btc_2 = self.get_89pct_btc_of_usd(price)
        self.execute_trade("buy", price, timestamp, signal_time, partial_btc_2)

        partial_btc_3 = self.get_89pct_btc_of_usd(price)
        self.execute_trade("buy", price, timestamp, signal_time, partial_btc_3)

    def get_89pct_btc_of_usd(self, price):
        available_usd = self.balance_usd * 0.89
        btc_approx = available_usd / (price * (1 + self.fee_percentage))
        return round(btc_approx, 8)

    # (CHANGED) execute_trade calls update_balance to handle cost basis
    def execute_trade(self, trade_type, price, timestamp, signal_timestamp, trade_btc):
        """
        Execute a single trade. Note that daily-limit checking and incrementing
        are now handled in check_for_signals() so that multi-part buys only 
        count as a single daily trade. We *do* still enforce the hourly limit.
        """
        self._clean_up_hourly_trades()
        max_trades_per_hour = 3
        if len(self.trades_this_hour) >= max_trades_per_hour:
            self.logger.info(f"Reached hourly trade limit {max_trades_per_hour}, skipping trade.")
            return

        data_source = 'historical' if signal_timestamp < self.strategy_start_time else 'live'
        from tdr_core.trade import Trade  # Local import to avoid circular references
        trade_info = Trade(
            trade_type,
            self.symbol,
            trade_btc,
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
                f"market-{trade_type}", self.symbol, trade_btc
            )
            self.logger.info(f"Executed LIVE {trade_type} order: {result}")
            trade_info.order_result = result
            if result.get("status") == "error":
                self.logger.error(f"Trade failed: {result}")
                self._log_failed_trade(trade_info)
                return
            self.update_balance(trade_type, price, trade_btc)
        else:
            self.logger.info(f"Executed DRY RUN {trade_type} order: {trade_info.to_dict()}")
            self.trade_log.append(trade_info)
            self.update_balance(trade_type, price, trade_btc)

        self.trades_this_hour.append(datetime.utcnow())
        self._log_successful_trade(trade_info)

    # (CHANGED) This now includes cost basis logic for the current position.
    def update_balance(self, trade_type, fill_price, fill_btc):
        """
        Update balance after a trade is executed (including fees and P&L).
        Now uses cost-basis tracking to fix the entry price & PnL issues.
        """
        fee = self.calculate_fee(fill_btc, fill_price)
        self.total_fees_paid += fee

        if trade_type == "buy":
            cost_usd = fill_btc * fill_price
            total_cost_usd = cost_usd + fee
            if total_cost_usd > self.balance_usd:
                possible_btc = self.balance_usd / (fill_price * (1 + self.fee_percentage))
                possible_btc = round(possible_btc, 8)
                fill_btc = possible_btc
                cost_usd = fill_btc * fill_price
                fee = self.calculate_fee(fill_btc, fill_price)
                total_cost_usd = cost_usd + fee
            self.balance_usd -= total_cost_usd
            self.balance_btc += fill_btc

            # If we are going long or adding to a long
            if self.position_size >= 0:
                self.position_cost_basis += (fill_btc * fill_price)
                self.position_size += fill_btc
            else:
                # If we were short (position_size < 0), 
                # you'd do partial close logic, etc. 
                # but if your code never does short + buy, 
                # we keep it minimal.
                pass

            if self.last_trade_price is not None and self.position == -1:
                # old code for short -> buy
                profit = fill_btc * (self.last_trade_price - fill_price) - fee
                self.current_balance += profit
                self.total_profit_loss += profit
                if profit > 0:
                    self.profitable_trades += 1

        elif trade_type == "sell":
            proceeds_usd = fill_btc * fill_price
            fee_sell = proceeds_usd * self.fee_percentage
            fee = fee_sell
            net_usd = proceeds_usd - fee

            if fill_btc > self.balance_btc:
                fill_btc = self.balance_btc
                fill_btc = round(fill_btc, 8)
                proceeds_usd = fill_btc * fill_price
                fee = proceeds_usd * self.fee_percentage
                net_usd = proceeds_usd - fee

            self.balance_btc -= fill_btc
            self.balance_usd += net_usd

            if self.position_size > 0:
                # partial or full close of a long
                if fill_btc > self.position_size:
                    fill_btc = self.position_size
                ratio = fill_btc / self.position_size
                cost_removed = ratio * self.position_cost_basis
                self.position_cost_basis -= cost_removed
                self.position_size -= fill_btc
                # If self.position_size is 0, we closed out the position fully.

            if self.last_trade_price is not None and self.position == 1:
                profit = fill_btc * (fill_price - self.last_trade_price) - fee
                self.current_balance += profit
                self.total_profit_loss += profit
                if profit > 0:
                    self.profitable_trades += 1

        self.last_trade_price = fill_price
        self.trades_executed += 1

        balance_ratio = self.current_balance / self.initial_balance if self.initial_balance else 1
        self.current_amount = self.initial_amount * balance_ratio

        # Update the max/min for USD and BTC after each trade
        if self.balance_usd > self.max_balance_usd:
            self.max_balance_usd = self.balance_usd
        if self.balance_usd < self.min_balance_usd:
            self.min_balance_usd = self.balance_usd
        if self.balance_btc > self.max_balance_btc:
            self.max_balance_btc = self.balance_btc
        if self.balance_btc < self.min_balance_btc:
            self.min_balance_btc = self.balance_btc

        # Also update max/min mark-to-market
        mtm_usd, _ = self.get_mark_to_market_values()
        if mtm_usd > self.max_mtm_usd:
            self.max_mtm_usd = mtm_usd
        if mtm_usd < self.min_mtm_usd:
            self.min_mtm_usd = mtm_usd

        self.logger.info(
            f"Trade completed - Balance: ${self.current_balance:.2f}, "
            f"Fees: ${fee:.2f}, Next trade amount: {self.current_amount:.8f}, "
            f"Total P&L: ${self.total_profit_loss:.2f} || "
            f"[BTC Balance: {self.balance_btc:.8f}, USD Balance: {self.balance_usd:.2f}]"
        )

    def get_mark_to_market_values(self):
        """
        Returns total notional in USD and BTC, based on the current market price.
        """
        current_price = self.data_manager.get_current_price(self.symbol) or 0.0
        total_usd_value = self.balance_usd + (self.balance_btc * current_price)
        total_btc_value = self.balance_btc + (self.balance_usd / current_price if current_price else 0.0)
        return total_usd_value, total_btc_value

    def get_status(self):
        """
        Return a dictionary summarizing the current status of this strategy,
        with a new 'position_info' block to show the current position in USD or BTC.
        Now also includes cost-basis-based entry price for accurate PnL.
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
            'initial_balance_btc': self.initial_balance_btc,
            'initial_balance_usd': self.initial_balance_usd,
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'balance_btc': self.balance_btc,
            'balance_usd': self.balance_usd,
            'total_return_pct': ((self.current_balance / self.initial_balance) - 1) * 100 if self.initial_balance != 0 else 0,
            'total_fees_paid': self.total_fees_paid,
            'trades_executed': self.trades_executed,
            'profitable_trades': self.profitable_trades,
            'win_rate': (self.profitable_trades / self.trades_executed * 100) if self.trades_executed else 0,
            'current_amount': self.current_amount,
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

        mtm_usd, mtm_btc = self.get_mark_to_market_values()
        status['mark_to_market_usd'] = mtm_usd
        status['mark_to_market_btc'] = mtm_btc

        if mtm_usd > self.max_mtm_usd:
            self.max_mtm_usd = mtm_usd
        if mtm_usd < self.min_mtm_usd:
            self.min_mtm_usd = mtm_usd

        status['max_balance_usd'] = self.max_balance_usd
        status['min_balance_usd'] = self.min_balance_usd
        status['max_balance_btc'] = self.max_balance_btc
        status['min_balance_btc'] = self.min_balance_btc
        status['max_mtm_usd'] = self.max_mtm_usd
        status['min_mtm_usd'] = self.min_mtm_usd

        # (CHANGED) We'll compute average entry price from the cost basis 
        #           if position_size > 0 (i.e. a long).
        position_info = {
            'current_price': self.data_manager.get_current_price(self.symbol) or 0.0,
            'entry_price': 0.0,  # will set below
            'position_size_btc': 0.0,
            'position_size_usd': 0.0,
            'unrealized_pnl': 0.0,
        }

        cp = position_info['current_price']

        if self.position == 1 and self.position_size > 0:
            # compute average entry from cost basis
            avg_entry_price = self.position_cost_basis / self.position_size if self.position_size else 0.0
            position_info['entry_price'] = avg_entry_price
            position_info['position_size_btc'] = self.position_size
            position_info['position_size_usd'] = self.position_size * cp
            cost_basis = self.position_cost_basis
            mark_value = self.position_size * cp
            position_info['unrealized_pnl'] = mark_value - cost_basis

        elif self.position == -1:
            # If short logic is used, do similar but for short cost-basis
            pass

        status['position_info'] = position_info
        return status
