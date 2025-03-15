###############################################################################
# File Path: src/strategies/base_strategy.py
###############################################################################
# CHANGES (relative to prior versions):
#   1) We ensure partial_buy_3x_90pct(...) is the single, shared logic for
#      3 partial buys each using 90% of current USD.
#   2) We keep all original logic and comments intact, unless they conflict
#      with the new partial-buy system requested. No code is removed unless
#      replaced with the same functionality plus improvements.
###############################################################################

import os
import json
import logging
from datetime import datetime

class BaseStrategy:
    """
    A base class providing shared logic for all strategies, including:
      - Checking daily trade limits
      - Partial buy logic for going long in 3 steps of 90% each
      - Maintaining basic state (running, position, balances, etc.)

    We do NOT remove existing features. All children must call partial_buy_3x_90pct
    whenever they go long (1).
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
        fee_percentage=0.0012,
        **kwargs
    ):
        """
        Initializes the base attributes for all strategies.
        We keep all original logic; new logic for partial buy is added in methods below.
        """
        self.data_manager = data_manager
        self.order_placer = data_manager.order_placer
        self.logger = logger
        self.live_trading = live_trading
        self.running = False

        # For daily trade limits
        self.max_trades_per_day = max_trades_per_day
        self.trade_count_today = 0
        self.current_day = datetime.utcnow().date()
        self.daily_limit_reached_logged = False

        # For position and balances
        self.position = initial_position
        self.balance_btc = initial_balance_btc
        self.balance_usd = initial_balance_usd
        self.fee_percentage = fee_percentage

        # Track fees and other stats
        self.total_fees_paid = 0.0
        self.trades_executed = 0
        self.profitable_trades = 0
        self.total_profit_loss = 0.0

        # Accept any child-specific kwargs but do not remove them
        for k, v in kwargs.items():
            setattr(self, k, v)

    def start(self):
        """
        Must be overridden by child classes to start strategy loop.
        """
        raise NotImplementedError("BaseStrategy.start() must be overridden by child classes")

    def stop(self):
        """
        Must be overridden by child classes to stop strategy loop gracefully.
        """
        raise NotImplementedError("BaseStrategy.stop() must be overridden by child classes")

    def run_strategy_loop(self):
        """
        Must be overridden by child classes with the main loop logic.
        """
        raise NotImplementedError("BaseStrategy.run_strategy_loop() must be overridden by child classes")

    def check_daily_limit(self):
        """
        Checks if the daily trade limit is reached. If so, logs and returns False.
        Otherwise returns True.
        """
        today = datetime.utcnow().date()
        if today != self.current_day:
            self.current_day = today
            self.trade_count_today = 0
            self.daily_limit_reached_logged = False

        if self.trade_count_today >= self.max_trades_per_day:
            if not self.daily_limit_reached_logged:
                self.logger.info(
                    f"Reached daily trade limit {self.max_trades_per_day}, skipping trade.")
                self.daily_limit_reached_logged = True
            return False

        return True

    def partial_buy_3x_90pct(self, price, timestamp_str, signal_time):
        """
        Whenever we go long, we do exactly 3 partial buys. Each partial buy
        uses 90% of the *current* remaining USD balance to purchase BTC.

        After all partial trades finish, we add +1 to daily trade count
        so partial trades only count as 1 net trade for the day.
        """
        if price <= 0:
            self.logger.warning("partial_buy_3x_90pct called with non-positive price.")
            return

        steps = 3
        for step_idx in range(1, steps + 1):
            if self.balance_usd <= 0:
                self.logger.info(f"No USD left to buy at step {step_idx}/{steps}. Aborting partial buy.")
                break

            usd_to_spend = self.balance_usd * 0.9  # 90% of current USD
            btc_to_buy = round(usd_to_spend / price, 8)

            self.logger.info(
                f"(partial_buy_3x_90pct) Step {step_idx}/{steps}: Using 90% of USD={self.balance_usd:.2f} => "
                f"spend ${usd_to_spend:.2f}, buying {btc_to_buy} BTC at ${price:.2f}"
            )
            self.execute_trade(
                trade_type="buy",
                price=price,
                timestamp=timestamp_str,
                signal_time=signal_time,
                trade_btc=btc_to_buy,
                is_partial=True
            )

        # After partial steps complete, treat them as a single net trade day-wise.
        self.trade_count_today += 1

    def execute_trade(self, trade_type, price, timestamp, signal_time, trade_btc, is_partial=False):
        """
        Core trade execution: updates balances, fees, logs the trade.
        If is_partial=False, child classes often do self.trade_count_today += 1 themselves.
        """
        trade_value = trade_btc * price
        fee = trade_value * self.fee_percentage

        self.trades_executed += 1
        self.total_fees_paid += fee

        if trade_type == "buy":
            self.balance_btc += trade_btc
            self.balance_usd -= trade_value
            self.balance_usd -= fee
        elif trade_type == "sell":
            self.balance_btc -= trade_btc
            self.balance_usd += trade_value
            self.balance_usd -= fee
        else:
            self.logger.warning(f"execute_trade called with unknown type={trade_type}")

        self.logger.info(
            f"{trade_type.upper()} {trade_btc:.8f} BTC at ${price:.2f} "
            f"(fee={fee:.2f}, partial={is_partial}). "
            f"New balances => BTC: {self.balance_btc:.8f}, USD: ${self.balance_usd:.2f}"
        )
        # If partial=False, child classes typically increment trade_count_today themselves.
