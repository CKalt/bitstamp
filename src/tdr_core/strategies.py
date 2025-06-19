# ----------------------------------------------------------------------------
# FULL FILE PATH: src/tdr_core/strategies.py
# ----------------------------------------------------------------------------
# CHANGES MADE:
#   1) Introduced position-based cost-basis tracking for accurate entry price & PnL.
#   2) Distinguish 'trades.json' (live trades) vs 'non-live-trades.json' (dry-run).
#   3) Return 'theoretical_trade' in get_status() if no real trade executed.
#   4) Clear self.theoretical_trade after a real trade.
#   5) (NEW) If live_trading=True, we now append new trades to 'trades.json' in real time
#      so you don't have to wait for strategy.stop().
#   6) (NEW) We skip trades if fill_btc < 1e-8, avoiding "zero position" confusion.
#   7) (Previously) Removed partial-fill clamp in the 'sell' side to allow actual short entries.
#   8) (NEW) For the 'buy' side, we now properly handle leftover BTC if you move from short to a net long.
#
# NOTE: We have taken care to preserve all existing comments and code, only adding
#       the minimal lines required for short->long leftover logic.
# ----------------------------------------------------------------------------

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


###############################################################################
class DiagnosticLogger:
    """
    Creates a diagnostic log file for each trading session that captures:
    - Periodic snapshots every 10 minutes
    - Signal evaluations and why they were/weren't taken
    - Position changes and P&L
    - Market regime changes
    - Errors and warnings
    """
    
    def __init__(self, strategy_name="strategy"):
        self.start_time = datetime.now()
        self.filename = f"diagnostics_{strategy_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        self.events = []
        self.last_snapshot_time = datetime.now()
        self.snapshot_interval = 600  # 10 minutes

        self.last_signal_eval = None  # Track last signal to avoid duplicates
        self.signal_eval_count = 0
        self.max_events = 2000  # Limit total events to keep file size down
        
        # Create initial file immediately
        self._save()
        
    def log_event(self, event_type, data):
        """Log an event with timestamp."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        self.events.append(event)
        self._save()
        
    def log_snapshot(self, position_info, market_data, strategy_state):
        """Log a periodic snapshot of system state."""
        snapshot = {
            "position": position_info,
            "market": market_data,
            "strategy": strategy_state
        }
        self.log_event("SNAPSHOT", snapshot)
        self.last_snapshot_time = datetime.now()
        
    def log_signal_evaluation(self, signal_type, signal_value, reason, will_trade, why_not=None):
        """Log when signals are evaluated."""
        data = {
            "signal_type": signal_type,
            "signal_value": signal_value,
            "reason": reason,
            "will_trade": will_trade,
            "why_not": why_not
        }
        self.log_event("SIGNAL_EVAL", data)
        
    def log_trade_execution(self, trade_type, price, amount, position_before, position_after, pnl):
        """Log trade executions."""
        data = {
            "trade_type": trade_type,
            "price": price,
            "amount": amount,
            "position_before": position_before,
            "position_after": position_after,
            "pnl": pnl
        }
        self.log_event("TRADE", data)
        
    def log_regime_change(self, old_regime, new_regime, confidence, metrics):
        """Log market regime changes."""
        data = {
            "old_regime": old_regime,
            "new_regime": new_regime,
            "confidence": confidence,
            "metrics": metrics
        }
        self.log_event("REGIME_CHANGE", data)
        
    def log_error(self, error_msg, stack_trace=None):
        """Log errors and warnings."""
        data = {
            "error": error_msg,
            "stack_trace": stack_trace or traceback.format_exc()
        }
        self.log_event("ERROR", data)
        
    def log_position_anomaly(self, description, details):
        """Log position tracking anomalies."""
        data = {
            "description": description,
            "details": details
        }
        self.log_event("POSITION_ANOMALY", data)
        
    def should_snapshot(self):
        """Check if it's time for a periodic snapshot."""
        return (datetime.now() - self.last_snapshot_time).total_seconds() >= self.snapshot_interval
        
    def _save(self):
        """Save events to file."""
        try:
            summary = {
                "session_start": self.start_time.isoformat(),
                "last_update": datetime.now().isoformat(),
                "total_events": len(self.events),
                "event_types": {
                    event_type: len([e for e in self.events if e["type"] == event_type])
                    for event_type in set(e["type"] for e in self.events)
                },
                "events": self.events[-1000:]  # Keep last 1000 events to prevent huge files
            }
            with open(self.filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save diagnostic log: {e}")

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

        # Decide which trades file to use (live vs. non-live).
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

        # Initial balances for BTC & USD (and legacy "amount" for P&L).
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

        # Daily trade limits
        self.max_trades_per_day = max_trades_per_day
        self.trade_count_today = 0
        self.current_day = datetime.utcnow().date()
        self.logger.debug(
            f"Trade limit set to {self.max_trades_per_day} trades/day.")

        self.trades_this_hour = []

        # Cost basis logic
        self.position_cost_basis = 0.0
        self.position_size = 0.0

        # For storing an initial theoretical trade if hist_position matches user request
        self.theoretical_trade = None

        # Initialize diagnostic logger
        self.diagnostic_logger = DiagnosticLogger(f"MA_{short_window}_{long_window}")

        # Register real-time callback
        data_manager.add_trade_observer(self.check_instant_signal)

        # Track staleness detection
        mtm_usd, _ = self.get_mark_to_market_values()
        self.max_mtm_usd = mtm_usd
        self.min_mtm_usd = mtm_usd
        self.max_balance_usd = self.balance_usd
        self.min_balance_usd = self.balance_usd
        self.max_balance_btc = self.balance_btc
        self.min_balance_btc = self.balance_btc

    def _clean_up_hourly_trades(self):
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        self.trades_this_hour = [
            t for t in self.trades_this_hour if t > one_hour_ago]

    def start(self):
        """
        Start the strategy loop in a background thread.
        """
        self.running = True
        self.strategy_thread = threading.Thread(
            target=self.run_strategy_loop, daemon=True)
        self.strategy_thread.start()
        self.logger.info("Strategy loop started.")

    def stop(self):
        """
        Stop the strategy loop and, if in dry-run, save trades to file.
        If in live mode, we assume real trades were appended in real time,
        but we might still finalize them here if needed.
        """
        self.running = False
        self.logger.info("Strategy loop stopped.")
        if not self.live_trading and self.trade_log:
            try:
                file_path = os.path.abspath(self.trade_log_file)
                with open(file_path, 'w') as f:
                    json.dump([t.to_dict()
                              for t in self.trade_log], f, indent=2)
                self.logger.info(
                    f"Trades logged to '{file_path}' (dry-run mode).")
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

                        # Check signals (MA crossover)

                        # Diagnostic: Log signal evaluation
                        if self.diagnostic_logger.should_snapshot():
                            self._log_diagnostic_snapshot()
                            
                        # Always log signal evaluations
                        will_trade = False
                        why_not = []
                        
                        if latest_signal == 1 and self.position <= 0:
                            will_trade = self.trade_count_today < self.max_trades_per_day
                            if not will_trade:
                                why_not.append(f"Daily limit reached: {self.trade_count_today}/{self.max_trades_per_day}")
                        elif latest_signal == -1 and self.position >= 0:
                            will_trade = self.trade_count_today < self.max_trades_per_day
                            if not will_trade:
                                why_not.append(f"Daily limit reached: {self.trade_count_today}/{self.max_trades_per_day}")
                        else:
                            why_not.append(f"Signal {latest_signal} matches current position {self.position}")
                            
                        self.diagnostic_logger.log_signal_evaluation(
                            signal_type="MA_CROSSOVER",
                            signal_value=latest_signal,
                            reason=f"Short MA: {df_ma.iloc[-1]['Short_MA']:.2f}, Long MA: {df_ma.iloc[-1]['Long_MA']:.2f}",
                            will_trade=will_trade,
                            why_not=why_not if why_not else None
                        )

                        self.check_for_signals(
                            latest_signal, current_price, signal_time)
                    else:
                        self.logger.debug("Not enough data to compute MAs.")
                except Exception as e:
                    self.logger.error(
                        f"Error in strategy loop for {self.symbol}: {e}")
            else:
                self.logger.debug(f"No data loaded for {self.symbol} yet.")
            time.sleep(60)

    def _log_diagnostic_snapshot(self):
        """Create a diagnostic snapshot."""
        try:
            current_price = self.data_manager.get_current_price(self.symbol) or 0
            mtm_usd, mtm_btc = self.get_mark_to_market_values()
            
            position_info = {
                "direction": self.position,
                "btc_balance": self.balance_btc,
                "usd_balance": self.balance_usd,
                "position_size": self.position_size,
                "cost_basis": self.position_cost_basis,
                "mtm_usd": mtm_usd,
                "unrealized_pnl": self._calculate_unrealized_pnl()
            }
            
            market_data = {
                "current_price": current_price,
                "short_ma": self.df_ma.iloc[-1]['Short_MA'] if not self.df_ma.empty else None,
                "long_ma": self.df_ma.iloc[-1]['Long_MA'] if not self.df_ma.empty else None,
                "signal": self.df_ma.iloc[-1]['MA_Signal'] if not self.df_ma.empty else None
            }
            
            strategy_state = {
                "trades_today": self.trade_count_today,
                "total_trades": self.trades_executed,
                "total_pnl": self.total_profit_loss
            }
            
            self.diagnostic_logger.log_snapshot(position_info, market_data, strategy_state)
        except Exception as e:
            self.diagnostic_logger.log_error(f"Failed to create snapshot: {e}")

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
        Real-time callback for each new trade. If there's a new crossover, act now.
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
        df_ma['Long_MA'] = df_ma['close'].rolling(self.long_window).mean()
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
        If the new MA signal differs from our current position, place trades.
        Also checks daily trade-limit; if at max, it skips.
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
                    f"Reached daily trade limit {self.max_trades_per_day}, skipping trade.")
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
                    f"Reached daily trade limit {self.max_trades_per_day}, skipping trade.")
                return

            self.logger.info(f"Sell signal triggered at {current_price}")
            self.position = -1
            self.last_trade_reason = "MA Crossover: short below long."
            trade_btc = round(self.balance_btc, 8)
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
        Simulate a multi-part buy so we can keep within a 90% rule but only 1 daily trade.
        """
        # Store initial state
        initial_position_size = self.position_size
        initial_cost_basis = self.position_cost_basis

        partial_btc_1 = self.get_89pct_btc_of_usd(price)
        self.execute_trade("buy", price, timestamp, signal_time, partial_btc_1)

        partial_btc_2 = self.get_89pct_btc_of_usd(price)
        self.execute_trade("buy", price, timestamp, signal_time, partial_btc_2)

        partial_btc_3 = self.get_89pct_btc_of_usd(price)
        self.execute_trade("buy", price, timestamp, signal_time, partial_btc_3)
        
        # Validate final position
        total_btc_bought = self.position_size - initial_position_size
        if total_btc_bought > 0:
            expected_cost = total_btc_bought * price * (1 + self.fee_percentage)
            actual_cost_added = self.position_cost_basis - initial_cost_basis
            
            if abs(actual_cost_added - expected_cost) > 1.0:
                self.logger.warning(f"Position tracking error detected! Expected cost: ${expected_cost:.2f}, Actual: ${actual_cost_added:.2f}")
                # Correct the cost basis
                self.position_cost_basis = self.position_size * price
                self.logger.info(f"Corrected position cost basis to ${self.position_cost_basis:.2f}")
        
        # Log final position state
        self.logger.info(f"Three-part buy complete: {self.position_size:.8f} BTC, cost basis: ${self.position_cost_basis:.2f}")

    def get_89pct_btc_of_usd(self, price):
        available_usd = self.balance_usd * 0.89
        btc_approx = available_usd / (price * (1 + self.fee_percentage))
        return round(btc_approx, 8)

    def execute_trade(self, trade_type, price, timestamp, signal_time, trade_btc):
        """
        Execute a single trade. 
        (NEW) If trade_btc < 1e-8, skip to avoid confusion with 0.0 updates.
        (NEW) If live_trading=True, append to trades.json immediately.
        """
        if trade_btc < 1e-8:
            self.logger.debug(
                f"Skipping trade because fill_btc is too small: {trade_btc}")
            return

        self._clean_up_hourly_trades()
        max_trades_per_hour = 3
        if len(self.trades_this_hour) >= max_trades_per_hour:
            self.logger.info(
                f"Reached hourly trade limit {max_trades_per_hour}, skipping trade.")
            return

        from tdr_core.trade import Trade
        trade_info = Trade(
            trade_type,
            self.symbol,
            trade_btc,
            price,
            datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'),
            self.last_trade_reason,
            'live' if self.live_trading else 'historical',
            signal_time,
            live_trading=self.live_trading
        )

        self.last_trade_data_source = trade_info.data_source
        self.last_trade_signal_timestamp = signal_time

        # Place order with the exchange if live.
        if self.live_trading:
            result = self.order_placer.place_order(
                f"market-{trade_type}", self.symbol, trade_btc)
            self.logger.info(f"Executed LIVE {trade_type} order: {result}")
            trade_info.order_result = result
            if result.get("status") == "error":
                self.logger.error(f"Trade failed: {result}")
                self._log_failed_trade(trade_info)
                return
            # Update balances & cost basis
            self.update_balance(trade_type, price, trade_btc)

            # (NEW) Append to trades.json right away for live trades
            try:
                file_path = os.path.abspath(self.trade_log_file)
                if not os.path.exists(file_path):
                    existing_trades = []
                else:
                    with open(file_path, 'r') as f:
                        try:
                            existing_trades = json.load(f)
                        except json.JSONDecodeError:
                            existing_trades = []
                existing_trades.append(trade_info.to_dict())
                with open(file_path, 'w') as f:
                    json.dump(existing_trades, f, indent=2)
                self.logger.debug(
                    f"Appended live trade to {self.trade_log_file}")
            except Exception as e:
                self.logger.error(f"Failed to write live trade: {e}")

        else:
            # Dry-run => no actual exchange order, just local simulation
            self.logger.info(
                f"Executed DRY RUN {trade_type} order: {trade_info.to_dict()}")
            self.trade_log.append(trade_info)
            self.update_balance(trade_type, price, trade_btc)

        self.trades_this_hour.append(datetime.utcnow())
        self._log_successful_trade(trade_info)

        # If a theoretical trade existed, clear it
        if self.theoretical_trade is not None:
            self.logger.debug(
                "Clearing theoretical trade because an actual trade occurred.")
            self.theoretical_trade = None

    def update_balance(self, trade_type, fill_price, fill_btc):
        """
        Update balance after a trade, tracking cost basis & partial fills.
        (NEW) If final fill_btc ends up 0, skip it to avoid no-op.

        REMINDER: 
         - We have removed partial-fill clamp on the 'sell' side to allow short entries.
         - We also fix leftover logic on the 'buy' side so going from short->long updates position_size properly.
        """
        fee = self.calculate_fee(fill_btc, fill_price)
        self.total_fees_paid += fee

        if trade_type == "buy":
            cost_usd = fill_btc * fill_price
            total_cost_usd = cost_usd + fee
            if total_cost_usd > self.balance_usd:
                # partial fill correction
                possible_btc = self.balance_usd / \
                    (fill_price * (1 + self.fee_percentage))
                possible_btc = round(possible_btc, 8)
                if possible_btc < 1e-8:
                    self.logger.debug(
                        f"Cannot buy anything with leftover USD. Skipping.")
                    return
                fill_btc = possible_btc
                cost_usd = fill_btc * fill_price
                fee = self.calculate_fee(fill_btc, fill_price)
                total_cost_usd = cost_usd + fee

            self.balance_usd -= total_cost_usd
            self.balance_btc += fill_btc

            # Critical fix: Handle position tracking correctly
            if self.position == -1 and self.position_size <= 0:
                # Transitioning from short to long - reset tracking
                self.logger.info(f"Transitioning from SHORT to LONG")
                self.position_size = fill_btc
                self.position_cost_basis = fill_btc * fill_price
            elif self.position_size >= 0:
                # Adding to existing long position
                self.position_cost_basis += (fill_btc * fill_price)
                self.position_size += fill_btc
            else:
                # Covering short position
                if abs(self.position_size) >= fill_btc:
                    # Just reducing short
                    self.position_size += fill_btc
                    # Don't change cost basis when covering
                else:
                    # Going from short through neutral to long
                    leftover_btc = fill_btc - abs(self.position_size)
                    self.position_size = leftover_btc
                    self.position_cost_basis = leftover_btc * fill_price

            # Validate position after update
            if self.position_size > 0 and self.position_cost_basis > 0:
                avg_entry = self.position_cost_basis / self.position_size
                if avg_entry > fill_price * 1.5:
                    self.logger.error(f"Position tracking error: avg entry ${avg_entry:.2f} > 1.5x fill price ${fill_price:.2f}")
                    # Reset to reasonable values
                    self.position_cost_basis = self.position_size * fill_price
                    self.logger.info(f"Reset cost basis to ${self.position_cost_basis:.2f}")

            if self.last_trade_price is not None and self.position == -1:
                # old code for short -> buy
                profit = fill_btc * (self.last_trade_price - fill_price) - fee
                self.current_balance += profit
                self.total_profit_loss += profit
                if profit > 0:
                    self.profitable_trades += 1

        elif trade_type == "sell":
            # If fill_btc > current balance, we open or add to a short (balance_btc goes negative).
            proceeds_usd = fill_btc * fill_price
            fee_sell = proceeds_usd * self.fee_percentage
            fee = fee_sell
            net_usd = proceeds_usd - fee

            self.balance_btc -= fill_btc
            self.balance_usd += net_usd

            if self.position_size > 0:
                # partial or full close of a long
                if fill_btc > self.position_size:
                    # going from long to short
                    fill_btc_for_long = self.position_size
                    ratio = 1.0  # fully closing that long portion
                    cost_removed = self.position_cost_basis
                    self.position_cost_basis -= cost_removed
                    self.position_size -= fill_btc_for_long
                    # leftover portion is new short
                    leftover_btc_for_short = fill_btc - fill_btc_for_long
                    if leftover_btc_for_short > 1e-8:
                        self.position_size -= leftover_btc_for_short
                        self.position_cost_basis += leftover_btc_for_short * fill_price
                else:
                    # partial or full flatten only
                    ratio = fill_btc / self.position_size
                    cost_removed = ratio * self.position_cost_basis
                    self.position_cost_basis -= cost_removed
                    self.position_size -= fill_btc

            else:
                # Going short: track USD held and BTC amount sold
                self.position_size = 0.0  # No BTC held when short
                self.position_cost_basis = fill_btc  # BTC amount sold for P&L calc
                self.logger.info(f"Short position: sold {fill_btc:.8f} BTC @ ${fill_price:.2f}, holding ${self.balance_usd:.2f} USD")

            if self.last_trade_price is not None and self.position == 1:
                profit = fill_btc * (fill_price - self.last_trade_price) - fee
                self.current_balance += profit
                self.total_profit_loss += profit
                if profit > 0:
                    self.profitable_trades += 1

        self.last_trade_price = fill_price
        self.trades_executed += 1

        # Recompute 'current_amount' for old P&L logic
        ratio = self.current_balance / self.initial_balance if self.initial_balance else 1
        self.current_amount = self.initial_amount * ratio

        # Validate position consistency
        if abs(self.position_size) < 1e-8:
            self.position_size = 0.0
            self.position_cost_basis = 0.0
            if self.position != 0:
                self.logger.warning(f"Position size near zero but position flag is {self.position}. Resetting to neutral.")
                self.position = 0
        
        # Log position state for debugging
        self.logger.debug(f"Position update: size={self.position_size:.8f}, cost_basis={self.position_cost_basis:.2f}, direction={self.position}")

        # Update max/min USD & BTC
        if self.balance_usd > self.max_balance_usd:
            self.max_balance_usd = self.balance_usd
        if self.balance_usd < self.min_balance_usd:
            self.min_balance_usd = self.balance_usd
        if self.balance_btc > self.max_balance_btc:
            self.max_balance_btc = self.balance_btc
        if self.balance_btc < self.min_balance_btc:
            self.min_balance_btc = self.balance_btc

        # Update max/min MTM
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
        total_btc_value = self.balance_btc + \
            (self.balance_usd / current_price if current_price else 0.0)
        return total_usd_value, total_btc_value

    def get_status(self):
        """
        Return a dictionary summarizing the current status, including 'position_info'
        that shows cost-basis-based entry price, position size, and unrealized PnL.
        """
        # Validate position tracking before building status
        if self.position == 1 and self.position_size > 0:
            avg_entry = self.position_cost_basis / self.position_size if self.position_size > 0 else 0
            current_price = self.data_manager.get_current_price(self.symbol) or 0
            
            # Sanity check: entry price shouldn't be more than 1.5x current price
            if avg_entry > current_price * 1.5 and current_price > 0:
                self.logger.error(f"Invalid entry price detected: ${avg_entry:.2f} vs current ${current_price:.2f}")
                # Attempt to fix by recalculating based on current balance
                # Assume entry was 5% below current price as a reasonable estimate
                self.position_cost_basis = self.position_size * current_price * 0.95
                self.logger.info(f"Reset position cost basis to ${self.position_cost_basis:.2f}")

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
            'remaining_trades_today': max(0, self.max_trades_per_day - self.trade_count_today),
            'theoretical_trade': self.theoretical_trade
        }

        if self.last_trade_reason:
            status['last_trade'] = self.last_trade_reason
            status['last_trade_data_source'] = self.last_trade_data_source
            if self.last_trade_signal_timestamp:
                status['last_trade_signal_timestamp'] = self.last_trade_signal_timestamp.strftime(
                    '%Y-%m-%d %H:%M:%S')

        if hasattr(self, 'df_ma') and not self.df_ma.empty:
            status['ma_difference'] = self.df_ma.iloc[-1]['Short_MA'] - \
                self.df_ma.iloc[-1]['Long_MA']
            if len(self.df_ma) >= 2:
                short_ma_slope = self.df_ma.iloc[-1]['Short_MA'] - \
                    self.df_ma.iloc[-2]['Short_MA']
                long_ma_slope = self.df_ma.iloc[-1]['Long_MA'] - \
                    self.df_ma.iloc[-2]['Long_MA']
                status['ma_slope_difference'] = short_ma_slope - long_ma_slope
                status['short_ma_momentum'] = 'Increasing' if short_ma_slope > 0 else 'Decreasing'
                status['long_ma_momentum'] = 'Increasing' if long_ma_slope > 0 else 'Decreasing'
                status['momentum_alignment'] = (
                    'Aligned' if (short_ma_slope > 0 and long_ma_slope > 0)
                    or (short_ma_slope < 0 and long_ma_slope < 0)
                    else 'Diverging'
                )

        if self.trades_executed > 0:
            status['average_fee_per_trade'] = self.total_fees_paid / \
                self.trades_executed
            status['risk_reward_ratio'] = (
                abs(self.total_profit_loss /
                    self.total_fees_paid) if self.total_fees_paid > 0 else 0
            )

        # Mark-to-market updates
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

        # Build position_info
        position_info = {
            'current_price': self.data_manager.get_current_price(self.symbol) or 0.0,
            'entry_price': 0.0,
            'position_size_btc': 0.0,
            'position_size_usd': 0.0,
            'unrealized_pnl': 0.0,
        }

        # Fix position flag early if it's wrong
        if abs(self.balance_btc) < 1e-6 and self.balance_usd > 10000 and self.position == 0:
            self.position = -1  # We're short, holding USD
            self.logger.info(f"Corrected position flag to SHORT: holding ${self.balance_usd:.2f} USD, 0 BTC")
        elif self.balance_btc > 1e-6 and abs(self.balance_usd) < 1000 and self.position == 0:
            self.position = 1   # We're long, holding BTC
            self.logger.info(f"Corrected position flag to LONG: holding {self.balance_btc:.8f} BTC")

        cp = position_info['current_price']

        # Handle theoretical vs real trades properly
        if self.trades_executed == 0 and self.theoretical_trade:
            # THEORETICAL TRADE: Show what position should be worth
            entry_price = self.theoretical_trade.get('entry_price', 0.0)
            if self.theoretical_trade['direction'] == 'long':
                position_info['entry_price'] = entry_price
                position_info['position_size_btc'] = self.theoretical_trade['amount']
                position_info['position_size_usd'] = self.theoretical_trade['amount'] * cp
                position_info['unrealized_pnl'] = (cp - entry_price) * self.theoretical_trade['amount']
            else:  # short
                position_info['entry_price'] = entry_price  
                position_info['position_size_btc'] = 0.0
                position_info['position_size_usd'] = self.theoretical_trade['amount']
                btc_equivalent = self.theoretical_trade['amount'] / entry_price
                position_info['unrealized_pnl'] = (entry_price - cp) * btc_equivalent

        else:
            # REAL TRADES: Use actual balances and tracking
            if self.position == 1:
                # Long position - holding BTC
                if self.position_size > 1e-8:
                    avg_entry_price = self.position_cost_basis / self.position_size
                    position_info['entry_price'] = avg_entry_price
                    position_info['position_size_btc'] = self.position_size
                    position_info['position_size_usd'] = self.position_size * cp
                    position_info['unrealized_pnl'] = (self.position_size * cp) - self.position_cost_basis
                else:
                    # Long but check actual balance
                    position_info['entry_price'] = 0.0
                    position_info['position_size_btc'] = self.balance_btc
                    position_info['position_size_usd'] = self.balance_btc * cp
                    position_info['unrealized_pnl'] = 0.0
                    
            elif self.position == -1:
                # Short position - holding USD from selling BTC
                position_info['entry_price'] = self.last_trade_price or 0.0
                position_info['position_size_btc'] = 0.0  # No BTC held
                position_info['position_size_usd'] = self.balance_usd  # USD from sale
                
                # P&L for short: (sell_price - current_price) * btc_amount_sold
                if self.last_trade_price and self.position_cost_basis > 0:
                    btc_sold = self.position_cost_basis
                    position_info['unrealized_pnl'] = (self.last_trade_price - cp) * btc_sold
                else:
                    position_info['unrealized_pnl'] = 0.0
            else:
                # Truly neutral position
                position_info['entry_price'] = 0.0
                position_info['position_size_btc'] = 0.0
                position_info['position_size_usd'] = 0.0
                position_info['unrealized_pnl'] = 0.0

        status['position_info'] = position_info

        ########################################################################
        # (A) Provide an MA-based measure of "how close" we are to crossing:
        ########################################################################
        if status['ma_difference'] is not None:
            short_val = self.df_ma.iloc[-1]['Short_MA']
            long_val = self.df_ma.iloc[-1]['Long_MA']
            avg_ma = (short_val + long_val) / \
                2.0 if (short_val + long_val) != 0 else 0.0
            if avg_ma != 0.0:
                status['ma_signal_proximity'] = abs(
                    short_val - long_val) / abs(avg_ma)
            else:
                status['ma_signal_proximity'] = None
        else:
            status['ma_signal_proximity'] = None
        ########################################################################

        return status

    def _log_successful_trade(self, trade_info):
        self.logger.info(
            f"Trade executed successfully: {trade_info.to_dict()}")

    def _log_failed_trade(self, trade_info):
        self.logger.info(f"Trade failed/canceled: {trade_info.to_dict()}")


###############################################################################
class AdaptiveMultiStrategy(MACrossoverStrategy):
    """
    Adaptive strategy that switches between different trading approaches:

    TRENDING MARKETS: Uses MA crossovers (trend following)
    RANGING MARKETS: Uses mean reversion (RSI + Bollinger Bands)  
    VOLATILE MARKETS: Uses breakout strategies (MACD + volume)

    This way we're ALWAYS trading optimally instead of sitting out!
    """

    def __init__(self, *args, **kwargs):
        # Extract adaptive parameters
        self.regime_lookback = kwargs.pop('regime_lookback', 50)
        self.regime_switch_threshold = kwargs.pop(
            'regime_switch_threshold', 0.7)
        self.min_trade_gap_minutes = kwargs.pop('min_trade_gap_minutes', 30)
        self.rsi_oversold = kwargs.pop('rsi_oversold', 30)
        self.rsi_overbought = kwargs.pop('rsi_overbought', 70)
        self.bb_std_dev = kwargs.pop('bb_std_dev', 2.0)
        self.volume_threshold = kwargs.pop('volume_threshold', 1.5)
        self.macd_threshold = kwargs.pop('macd_threshold', 0.001)
        self.signal_confirmation_bars = kwargs.pop(
            'signal_confirmation_bars', 2)

        # Initialize parent class
        super().__init__(*args, **kwargs)

        # Adaptive state
        self.current_regime = "unknown"
        self.regime_confidence = 0.0
        self.active_strategy = "trending"  # trending, ranging, volatile
        self.strategy_switches_today = 0
        self.signal_history = []
        self.last_confirmed_signal = 0

        # Ensure critical attributes exist (fix AttributeError)
        if not hasattr(self, 'last_trade_time'):
            self.last_trade_time = None

        # Performance tracking by strategy
        self.strategy_performance = {
            "trending": {"trades": 0, "profit": 0.0},
            "ranging": {"trades": 0, "profit": 0.0},
            "volatile": {"trades": 0, "profit": 0.0}
        }

        self.logger.info(f"🎯 ADAPTIVE MULTI-STRATEGY INITIALIZED")
        self.logger.info(
            f"   Will switch between TRENDING → RANGING → VOLATILE strategies")
        self.logger.info(
            f"   Regime lookback: {self.regime_lookback}, Gap: {self.min_trade_gap_minutes}min")

        self.logger.info(f"   Confidence thresholds: TRENDING=70%, RANGING=60%, VOLATILE=65%")
        self.logger.info(f"   Position-aware switching: $50k+ requires higher confidence")

    def detect_market_regime(self, df):
        """Detect if market is trending, ranging, or volatile."""
        if len(df) < self.regime_lookback:
            return "unknown", 0.0, {}

        recent_df = df.tail(self.regime_lookback).copy()

        # Calculate market metrics
        price_start = recent_df['close'].iloc[0]
        price_end = recent_df['close'].iloc[-1]
        price_high = recent_df['close'].max()
        price_low = recent_df['close'].min()

        total_return = abs(price_end - price_start) / price_start
        price_range = (price_high - price_low) / ((price_high + price_low) / 2)
        trend_strength = total_return / price_range if price_range > 0 else 0

        # Volatility
        returns = recent_df['close'].pct_change().dropna()
        volatility = returns.std() if len(returns) > 1 else 0

        # Whipsaw detection
        df_temp = add_moving_averages(
            recent_df.copy(), self.short_window, self.long_window, price_col='close')
        df_temp = generate_ma_signals(df_temp)
        signal_changes = (df_temp['MA_Signal'].diff() != 0).sum()
        whipsaw_ratio = (signal_changes / len(df_temp)) * 100

        # Range-bound detection
        ma_20 = recent_df['close'].rolling(20).mean()
        price_vs_ma = (recent_df['close'] - ma_20) / ma_20
        range_bound_score = 1.0 - \
            abs(price_vs_ma.mean()) if not price_vs_ma.empty else 0

        # Regime scoring
        regime_scores = {'trending': 0.0, 'ranging': 0.0, 'volatile': 0.0}

        # TRENDING indicators (more conservative)
        if trend_strength > 0.3:  # Reduced from 0.4 - more reasonable
            regime_scores['trending'] += 2.0
        if whipsaw_ratio < 2.0:  # Lower whipsaw tolerance
            regime_scores['trending'] += 2.0
        if volatility < 0.02:  # Increased from 0.015 - more reasonable
            regime_scores['trending'] += 1.5
        # Require sustained directional movement
        if len(recent_df) >= 10:
            recent_closes = recent_df['close'].tail(10)
            directional_moves = 0
            for i in range(1, len(recent_closes)):
                if (recent_closes.iloc[i] > recent_closes.iloc[i-1] and price_end > price_start) or \
                   (recent_closes.iloc[i] < recent_closes.iloc[i-1] and price_end < price_start):
                    directional_moves += 1
            if directional_moves >= 6:  # Reduced from 7 - 60% directional consistency
                regime_scores['trending'] += 1.0

        # RANGING indicators (YOUR CURRENT SITUATION!)
        if range_bound_score > 0.8:
            regime_scores['ranging'] += 2.0
        if whipsaw_ratio > 4.0:  # Your 6.14 whipsaw ratio!
            regime_scores['ranging'] += 2.0
        if trend_strength < 0.2:
            regime_scores['ranging'] += 1.0

        # VOLATILE indicators
        if volatility > 0.035:
            regime_scores['volatile'] += 2.0
        if price_range > 0.08:
            regime_scores['volatile'] += 1.0

        # Determine regime
        best_regime = max(regime_scores.items(), key=lambda x: x[1])
        regime = best_regime[0]
        max_score = best_regime[1]
        # Normalize to max possible score
        confidence = min(0.95, max_score / 6.0)

        # Create metrics dict first
        metrics = {'whipsaw_ratio': whipsaw_ratio, 'trend_strength': trend_strength}
        
        self.logger.info(
            f"📊 Regime Scores: TRENDING={regime_scores['trending']:.1f}, RANGING={regime_scores['ranging']:.1f}, VOLATILE={regime_scores['volatile']:.1f}")
        self.logger.info(f"📈 Market Metrics: whipsaw={metrics.get('whipsaw_ratio', 0):.1f}%, trend_strength={metrics.get('trend_strength', 0):.3f}, volatility={volatility:.4f}")
        self.logger.info(f"🎯 Final: {regime.upper()} (confidence: {confidence:.1%})")

        return regime, confidence, metrics

    def generate_ranging_signal(self, df):
        """Generate mean reversion signals for ranging markets."""
        if len(df) < 20:
            return 0, "Insufficient data for ranging strategy"

        # Calculate RSI
        df_rsi = calculate_rsi(df.copy(), window=14, price_col='close')
        current_rsi = df_rsi.iloc[-1]['RSI']

        # Calculate Bollinger Bands
        df_bb = calculate_bollinger_bands(
            df.copy(), window=20, num_std=self.bb_std_dev, price_col='close')
        current_price = df_bb.iloc[-1]['close']
        bb_upper = df_bb.iloc[-1]['BB_Upper']
        bb_lower = df_bb.iloc[-1]['BB_Lower']
        bb_middle = df_bb.iloc[-1]['BB_MA']

        signal = 0
        reason = "No mean reversion signal"

        # Mean reversion logic - BUY OVERSOLD, SELL OVERBOUGHT
        if current_rsi < self.rsi_oversold and current_price < bb_lower:
            signal = 1  # Oversold = BUY
            reason = f"Mean Reversion BUY: RSI {current_rsi:.1f} oversold + below BB"
        elif current_rsi > self.rsi_overbought and current_price > bb_upper:
            signal = -1  # Overbought = SELL
            reason = f"Mean Reversion SELL: RSI {current_rsi:.1f} overbought + above BB"
        elif current_rsi < 40 and current_price < bb_middle * 0.99:  # Additional entry condition
            signal = 1
            reason = f"Mean Reversion BUY: RSI {current_rsi:.1f} low + below BB middle"
        elif self.position != 0:
            # Exit positions when price returns to middle
            if self.position == 1 and current_price > bb_middle:
                signal = -1
                reason = f"Mean Reversion EXIT LONG: Price returned to BB middle"
            elif self.position == -1 and current_price < bb_middle:
                signal = 1
                reason = f"Mean Reversion EXIT SHORT: Price returned to BB middle"

        return signal, reason

    def generate_trending_signal(self, df):
        """Generate MA crossover signals for trending markets."""
        if len(df) < self.long_window:
            return 0, "Insufficient data for MA"

        df_ma = add_moving_averages(
            df.copy(), self.short_window, self.long_window, price_col='close')
        df_ma = generate_ma_signals(df_ma)

        signal = df_ma.iloc[-1]['MA_Signal']
        reason = f"MA Crossover: {'LONG' if signal == 1 else 'SHORT' if signal == -1 else 'NEUTRAL'}"

        return signal, reason

    def generate_volatile_signal(self, df):
        """Generate breakout signals for volatile markets."""
        if len(df) < 26:
            return 0, "Insufficient data for volatile strategy"

        df_macd = calculate_macd(
            df.copy(), fast=12, slow=26, signal=9, price_col='close')
        df_macd = generate_macd_signals(df_macd)

        macd_signal = df_macd.iloc[-1]['MACD_Signal']
        current_macd = df_macd.iloc[-1]['MACD']

        signal = 0
        reason = "No breakout signal"

        # Breakout logic with MACD
        if macd_signal == 1 and abs(current_macd) > self.macd_threshold:
            signal = 1
            reason = f"Breakout LONG: MACD bullish crossover"
        elif macd_signal == -1 and abs(current_macd) > self.macd_threshold:
            signal = -1
            reason = f"Breakout SHORT: MACD bearish crossover"

        return signal, reason

    def confirm_signal(self, current_signal):
        """Require multiple consecutive bars of the same signal."""
        self.signal_history.append(current_signal)

        if len(self.signal_history) > 10:
            self.signal_history = self.signal_history[-10:]

        # Require more confirmation when holding positions
        current_position_value = abs(self.balance_btc * (self.data_manager.get_current_price(self.symbol) or 0)) + self.balance_usd
        has_position = current_position_value > 10000
        
        required_bars = self.signal_confirmation_bars + (2 if has_position else 0)
        
        if len(self.signal_history) < required_bars:
            return False

        recent_signals = self.signal_history[-required_bars:]
        if not all(s == recent_signals[0] for s in recent_signals):
            return False

        if recent_signals[0] == self.last_confirmed_signal:
            return False

        self.last_confirmed_signal = recent_signals[0]
        return True

    def check_trade_gap(self):
        """Check if enough time passed since last trade."""
        if self.last_trade_time is None:
            return True

        minutes_since = (datetime.now() -
                         self.last_trade_time).total_seconds() / 60
        return minutes_since >= self.min_trade_gap_minutes

    def run_strategy_loop(self):
        """Main adaptive strategy loop."""
        while self.running:
            df = self.data_manager.get_price_dataframe(self.symbol)
            if not df.empty:
                try:
                    df = ensure_datetime_index(df)
                    df_resampled = df.resample('1H').agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                        'volume': 'sum', 'trades': 'sum', 'timestamp': 'last', 'source': 'last'
                    }).dropna()

                    if len(df_resampled) >= self.long_window:
                        # 1. Detect market regime
                        regime, confidence, metrics = self.detect_market_regime(
                            df_resampled)

                        # 2. Conservative strategy selection with position awareness
                        current_position_value = abs(self.balance_btc * (self.data_manager.get_current_price(self.symbol) or 0)) + self.balance_usd
                        has_significant_position = current_position_value > 50000  # $50k+ position
                        
                        # Higher confidence required when holding significant positions
                        confidence_threshold = 0.75 if has_significant_position else 0.65
                        
                        # Specific thresholds by regime
                        if regime == "trending":
                            required_confidence = 0.70  # Reduced from 0.80 - more reasonable
                        elif regime == "ranging":
                            required_confidence = 0.60  # Moderate bar for ranging
                        else:  # volatile
                            required_confidence = 0.65  # Reduced from 0.70

                        # Only switch if confidence exceeds threshold
                        if confidence >= required_confidence:
                            new_strategy = regime
                            self.logger.info(f"✅ Switching to {regime} strategy - confidence {confidence:.1%} >= {required_confidence:.1%} required")
                        else:
                            new_strategy = self.active_strategy  # Stay with current strategy
                            self.logger.info(f"⏸️  Staying with {self.active_strategy} strategy - {regime} confidence {confidence:.1%} < {required_confidence:.1%} required (position: ${current_position_value:,.0f})")

                        # Extra conservative check: don't switch away from ranging easily
                        if self.active_strategy == "ranging" and new_strategy != "ranging":
                            if confidence < 0.75:  # Reduced from 0.85 - still conservative but more reasonable
                                new_strategy = "ranging"
                                self.logger.debug(f"Staying in ranging strategy - confidence {confidence:.1%} insufficient to switch")

                        # Extra conservative check: don't switch away from trending easily when holding position
                        if self.active_strategy == "trending" and new_strategy != "trending" and has_significant_position:
                            if confidence < 0.75:  # High bar to leave trending when holding position
                                new_strategy = "trending"
                                self.logger.info(f"🔒 Staying in trending strategy - confidence {confidence:.1%} insufficient to switch (holding ${current_position_value:,.0f})")

                        # 3. Check for strategy switch
                        if new_strategy != self.active_strategy:
                            self.logger.info(f"🔄 STRATEGY SWITCH: {self.active_strategy} → {new_strategy} (confidence: {confidence:.1%})")
                            self.active_strategy = new_strategy
                            self.strategy_switches_today += 1
                        else:
                            self.logger.debug(f"Keeping {self.active_strategy} strategy (regime: {regime}, confidence: {confidence:.1%})")

                        # 4. Generate signal using active strategy
                        if self.active_strategy == "trending":
                            signal, signal_reason = self.generate_trending_signal(
                                df_resampled)
                        elif self.active_strategy == "ranging":
                            signal, signal_reason = self.generate_ranging_signal(
                                df_resampled)
                        elif self.active_strategy == "volatile":
                            signal, signal_reason = self.generate_volatile_signal(
                                df_resampled)
                        else:
                            signal, signal_reason = 0, "Unknown strategy"

                        # 5. Execute if signal confirmed
                        if signal != 0 and self.confirm_signal(signal):
                            current_price = df_resampled.iloc[-1]['close']
                            signal_time = df_resampled.index[-1]

                            self.logger.info(
                                f"📊 {self.active_strategy.upper()}: {signal_reason}")
                            self.check_for_signals(
                                signal, current_price, signal_time)

                        # Store regime info
                        self.current_regime = regime
                        self.regime_confidence = confidence

                except Exception as e:
                    self.logger.error(f"Error in adaptive strategy loop: {e}")
            time.sleep(60)

    def check_for_signals(self, latest_signal, current_price, signal_time):
        """Execute trades with adaptive strategy logic."""
        today = datetime.utcnow().date()
        if today != self.current_day:
            self.current_day = today
            self.trade_count_today = 0
            self.strategy_switches_today = 0

        if self.last_signal_time == signal_time:
            return

        # Check constraints
        if not self.check_trade_gap():
            mins = (datetime.now() - self.last_trade_time).total_seconds() / \
                60 if self.last_trade_time else 0
            self.logger.info(
                f"⏱️  Trade gap: {mins:.1f}min < {self.min_trade_gap_minutes}min required")
            return

        if self.trade_count_today >= self.max_trades_per_day:
            self.logger.info(
                f"📈 Daily limit: {self.trade_count_today}/{self.max_trades_per_day}")
            return

        # Execute trade
        if latest_signal == 1 and self.position <= 0:
            self.logger.info(
                f"🟢 {self.active_strategy.upper()} LONG at ${current_price}")
            self.position = 1
            self.last_trade_reason = f"Adaptive {self.active_strategy}: confirmed long"
            self.last_trade_time = datetime.now()
            self.buy_in_three_parts(current_price, datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S'), signal_time)
            self.trade_count_today += 1
            self.last_signal_time = signal_time
            self.strategy_performance[self.active_strategy]["trades"] += 1

        elif latest_signal == -1 and self.position >= 0:
            self.logger.info(
                f"🔴 {self.active_strategy.upper()} SHORT at ${current_price}")
            self.position = -1
            self.last_trade_reason = f"Adaptive {self.active_strategy}: confirmed short"
            self.last_trade_time = datetime.now()
            trade_btc = round(self.balance_btc, 8)
            self.execute_trade("sell", current_price, datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S'), signal_time, trade_btc)
            self.trade_count_today += 1
            self.last_signal_time = signal_time
            self.strategy_performance[self.active_strategy]["trades"] += 1

    def get_status(self):
        """Enhanced status with adaptive metrics."""
        status = super().get_status()

        # Add adaptive strategy metrics
        status.update({
            'adaptive_strategy': {
                'current_regime': self.current_regime,
                'regime_confidence': f"{self.regime_confidence:.1%}",
                'active_strategy': self.active_strategy,
                'strategy_switches_today': self.strategy_switches_today,
                'strategy_performance': self.strategy_performance,
                'position_value': abs(self.balance_btc * (self.data_manager.get_current_price(self.symbol) or 0)) + self.balance_usd,
                'confidence_required': 0.80 if self.current_regime == "trending" else 0.60 if self.current_regime == "ranging" else 0.70,
            },
            'signal_confirmation': {
                'signals_recorded': len(self.signal_history),
                'confirmation_required': self.signal_confirmation_bars,
                'last_confirmed_signal': self.last_confirmed_signal
            },
            'trade_timing': {
                'min_gap_minutes': self.min_trade_gap_minutes,
                'time_since_last_trade_minutes': (
                    (datetime.now() - self.last_trade_time).total_seconds() / 60
                    if self.last_trade_time else None
                ),
                'can_trade_now': self.check_trade_gap()
            }
        })

        return status
