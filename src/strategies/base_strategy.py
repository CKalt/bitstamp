###############################################################################
# File Path: src/strategies/base_strategy.py
###############################################################################
# CHANGES:
#   1) Added a default get_status() method so that every child strategy
#      (MA, RSI, RAMM, etc.) can call or inherit it. This fixes the AttributeError
#      when do_status calls `self.auto_trader.get_status()`.
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
      - A default get_status() to avoid "no attribute get_status" errors
      - Maintaining basic state (running, position, balances, etc.)
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

        # Additional child-specific kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Provide placeholders that child classes often define
        self.strategy_start_time = datetime.now()
        self.theoretical_trade = None
        self.position_size = 0.0
        self.position_cost_basis = 0.0

    def start(self):
        raise NotImplementedError("BaseStrategy.start() must be overridden by child classes")

    def stop(self):
        raise NotImplementedError("BaseStrategy.stop() must be overridden by child classes")

    def run_strategy_loop(self):
        raise NotImplementedError("BaseStrategy.run_strategy_loop() must be overridden by child classes")

    def check_daily_limit(self):
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
        if price <= 0:
            self.logger.warning("partial_buy_3x_90pct called with non-positive price.")
            return
        steps = 3
        for step_idx in range(1, steps + 1):
            if self.balance_usd <= 0:
                self.logger.info(f"No USD left to buy at step {step_idx}/{steps}. Aborting partial buy.")
                break
            usd_to_spend = self.balance_usd * 0.9
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
        # partial trades => count as 1 daily trade
        self.trade_count_today += 1

    def execute_trade(self, trade_type, price, timestamp, signal_time, trade_btc, is_partial=False):
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

    def get_status(self):
        """
        A default get_status() to avoid AttributeError when do_status calls self.auto_trader.get_status().

        Child strategies can override or extend this to provide more fields
        (like RSI, MAs, or specialized metrics).
        """
        # Basic mark-to-market
        current_price = self.data_manager.get_current_price('btcusd') or 0.0
        mark_to_market_usd = self.balance_usd + (self.balance_btc * current_price)
        if current_price == 0:
            mark_to_market_btc = 0.0
        else:
            mark_to_market_btc = self.balance_btc + (self.balance_usd / current_price)

        # If we had an "initial_balance" concept, we try to get it, else 0
        initial_balance = getattr(self, 'initial_balance', 0.0)

        # Basic position info
        position_info = {}
        if abs(self.position_size) > 1e-8:
            avg_entry_price = (self.position_cost_basis / abs(self.position_size)) if abs(self.position_cost_basis) > 1e-8 else 0.0
        else:
            avg_entry_price = 0.0
        current_pos_val = self.position_size * current_price

        if self.position > 0:
            position_info['position_size_btc'] = self.position_size
            position_info['position_size_usd'] = current_pos_val
            if avg_entry_price > 0:
                position_info['unrealized_pnl'] = (current_price - avg_entry_price)*self.position_size
            else:
                position_info['unrealized_pnl'] = 0.0
        elif self.position < 0:
            position_info['position_size_btc'] = self.position_size
            # cost basis is positive money spent. negative position means short
            position_info['position_size_usd'] = abs(self.position_cost_basis)
            if avg_entry_price > 0:
                mark_value = abs(self.position_size)*current_price
                position_info['unrealized_pnl'] = self.position_cost_basis - mark_value
            else:
                position_info['unrealized_pnl'] = 0.0
        else:
            position_info['position_size_btc'] = 0.0
            position_info['position_size_usd'] = 0.0
            position_info['unrealized_pnl'] = 0.0

        position_info['current_price'] = current_price
        position_info['entry_price'] = avg_entry_price

        # The main dictionary
        status_dict = {
            'running': self.running,
            'position': self.position,
            'trade_count_today': self.trade_count_today,
            'max_trades_per_day': self.max_trades_per_day,
            'remaining_trades_today': max(0, self.max_trades_per_day - self.trade_count_today),
            'balance_btc': self.balance_btc,
            'balance_usd': self.balance_usd,
            'mark_to_market_usd': mark_to_market_usd,
            'mark_to_market_btc': mark_to_market_btc,
            'initial_balance': initial_balance,
            'current_balance': mark_to_market_usd,  # convenience
            'total_fees_paid': self.total_fees_paid,
            'trades_executed': self.trades_executed,
            'theoretical_trade': self.theoretical_trade,
            'position_info': position_info
        }

        # Example total_return_pct if initial_balance was used
        if initial_balance != 0.0:
            status_dict['total_return_pct'] = (mark_to_market_usd / initial_balance - 1)*100
        else:
            status_dict['total_return_pct'] = 0.0

        # We can add more placeholders to avoid error if do_status tries to display them
        status_dict['profitable_trades'] = getattr(self, 'profitable_trades', 0)
        status_dict['total_profit_loss'] = getattr(self, 'total_profit_loss', 0.0)
        # Win rate, etc.
        if self.trades_executed > 0:
            status_dict['win_rate'] = (self.profitable_trades / self.trades_executed)*100 if self.profitable_trades else 0.0
            status_dict['average_profit_per_trade'] = status_dict['total_profit_loss'] / self.trades_executed if self.trades_executed else 0.0
        else:
            status_dict['win_rate'] = 0.0
            status_dict['average_profit_per_trade'] = 0.0

        # Typically child strategy sets self.strategy_start_time
        session_duration = datetime.now() - getattr(self, 'strategy_start_time', datetime.now())
        hours = session_duration.total_seconds()/3600
        # Some status calls might want risk_reward_ratio, etc. We'll default to 0
        status_dict['risk_reward_ratio'] = 0.0
        status_dict['last_trade'] = getattr(self, 'last_trade_reason', None)
        status_dict['last_trade_data_source'] = getattr(self, 'last_trade_data_source', None)
        status_dict['last_trade_signal_timestamp'] = getattr(self, 'last_trade_signal_timestamp', None)

        # Additional placeholders that do_status might expect
        status_dict['next_trigger'] = getattr(self, 'next_trigger', None)
        status_dict['current_trends'] = getattr(self, 'current_trends', {})
        status_dict['ma_difference'] = None
        status_dict['ma_slope_difference'] = None
        status_dict['short_ma_momentum'] = None
        status_dict['long_ma_momentum'] = None
        status_dict['momentum_alignment'] = None
        status_dict['last_rsi'] = None
        status_dict['rsi_window'] = getattr(self, 'rsi_window', 14)
        status_dict['overbought'] = getattr(self, 'overbought', 70)
        status_dict['oversold'] = getattr(self, 'oversold', 30)
        status_dict['rsi_proximity'] = None
        status_dict['ma_signal_proximity'] = None

        return status_dict
