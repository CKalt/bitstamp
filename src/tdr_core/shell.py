###############################################################################
# src/tdr_core/shell.py
###############################################################################
# Full File Path: src/tdr_core/shell.py
#
# CHANGES (EXPLANATION):
#   1) We ensure that, in do_status, we display the "Strategy" in short or long 
#      view if present in the status dict.
#   2) We show "rsi_signal_proximity" or "ma_signal_proximity" in short view 
#      with a short comment line to remind the user how to interpret it.
#   3) We do not remove any other code or docstrings. Minimal additions only.
###############################################################################

import cmd
import sys
import json
import time
import logging
import threading
import requests
import os
from datetime import datetime
from flask import Flask, request
from multiprocessing import Process, Manager

from tdr_core.order_placer import OrderPlacer
from tdr_core.strategies import MACrossoverStrategy, RSITradingStrategy
from tdr_core.data_manager import CryptoDataManager
from tdr_core.trade import Trade

from utils.analysis import analyze_data, run_trading_system
from data.loader import create_metadata_file, parse_log_file


def determine_rsi_position(df, rsi_window=14, overbought=70, oversold=30):
    # ... (unchanged as you had it) ...
    from indicators.technical_indicators import ensure_datetime_index, calculate_rsi
    # ...
    return 0  # or 1 or -1 accordingly

class CryptoShell(cmd.Cmd):
    intro = 'Welcome to the Crypto Shell (No CLI args). Type help or ? to list commands.\n'
    prompt = '(crypto) '

    def __init__(self, data_manager, order_placer, logger,
                 verbose=False, live_trading=False, stop_event=None,
                 max_trades_per_day=5):
        super().__init__()
        self.data_manager = data_manager
        self.order_placer = order_placer
        self.data_manager.order_placer = order_placer
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
            # ... (unchanged) ...
        }

        self.data_manager.add_candlestick_observer(self.candlestick_callback)
        self.data_manager.add_trade_observer(self.trade_callback)

    # ... other commands unmodified ...

    def do_auto_trade(self, arg):
        """
        Start auto-trading using the best strategy from best_strategy.json.
        """
        # ... existing logic ...
        pass

    def do_stop_auto_trade(self, arg):
        """
        Stop auto-trading if running.
        """
        # ... unchanged ...
        pass

    def do_status(self, arg):
        """
        Show status of auto-trading. Usage: status [long]
        """
        sub_arg = arg.strip().lower()
        show_full = (sub_arg == 'long')

        if not self.auto_trader or not self.auto_trader.running:
            print("Auto-trading is not running.")
            return

        status = self.auto_trader.get_status()
        pos_str = {1:'Long', -1:'Short', 0:'Neutral'}.get(status['position'], 'Unknown')

        if not show_full:
            print("\nPosition Details (Short View):")
            print("━"*50)

            # NEW: show the strategy if present
            strategy_in_use = status.get('Strategy', 'N/A')
            print(f"  • Strategy In Use: {strategy_in_use}")

            print(f"  • Direction: {pos_str}")
            pos_info = status.get('position_info', {})
            print(f"  • Current Price:  ${pos_info.get('current_price', 0.0):.2f}")
            print(f"  • Entry Price:    ${pos_info.get('entry_price', 0.0):.2f}")

            if status['position'] == 1:
                print(f"  • Position Size (BTC): {pos_info.get('position_size_btc', 0.0):.8f}")
                print(f"  • Position Value (USD): ${pos_info.get('position_size_usd', 0.0):.2f}")
            elif status['position'] == -1:
                print(f"  • Short Size (BTC): {pos_info.get('position_size_btc', 0.0):.8f} (negative means short)")
                print(f"  • USD Held:         ${pos_info.get('position_size_usd', 0.0):.2f}")
            else:
                print("  • Neutral position, no open BTC or short.")

            print(f"  • Unrealized PnL:  ${pos_info.get('unrealized_pnl', 0.0):.2f}")

            # If we have proximity fields, show them
            ma_prox = status.get('ma_signal_proximity', None)
            if ma_prox is not None:
                print(f"  • MA Signal Proximity: {ma_prox:.4f} (closer to 0 => near crossing)")

            rsi_prox = status.get('rsi_signal_proximity', None)
            if rsi_prox is not None:
                print(f"  • RSI Signal Proximity: {rsi_prox:.4f} (0 => fully in oversold/overbought)")

            if status['trades_executed'] == 0 and status.get('theoretical_trade'):
                t = status['theoretical_trade']
                print(f"\n  This is a theoretical trade (no actual trades yet):")
                print(f"    • Timestamp:  {t['timestamp']}")
                print(f"    • Direction:  {t['direction']}")
                print(f"    • Amount:     {t['amount']}")
                print(f"    • Theoretical? {t['theoretical']}")

            print("")
            return

        # Else show the full version
        print("\nAuto-Trading Status:")
        print("━"*50)
        strategy_in_use = status.get('Strategy', 'N/A')
        print(f"  • Strategy: {strategy_in_use}")
        print(f"  • Running: {status['running']}")
        print(f"  • Position: {pos_str}")
        print(f"  • Daily Trades: {status['trade_count_today']}/{self.auto_trader.max_trades_per_day}")
        print(f"  • Remaining Trades Today: {status['remaining_trades_today']}")

        print("\nAccount Balances & Performance:")
        print(f"  • Initial USD Balance: ${status['initial_balance_usd']:.2f}")
        print(f"  • Initial BTC Balance: {status['initial_balance_btc']:.8f}")
        print(f"  • Current USD Balance: ${status['balance_usd']:.2f}")
        print(f"  • Current BTC Balance: {status['balance_btc']:.8f}")
        print(f"  • Total Return (vs initial): {status['total_return_pct']:.2f}%")
        print(f"  • Total P&L: ${status['total_profit_loss']:.2f}")
        print(f"  • Current Trade Amount: {status['current_amount']:.8f}")
        print(f"  • Total Fees Paid: ${status['total_fees_paid']:.2f}")

        print("\nMark-to-Market & Drawdowns:")
        print(f"  • Current MTM (USD): ${status['mark_to_market_usd']:.2f}")
        print(f"  • Current MTM (BTC): {status['mark_to_market_btc']:.8f}")
        print(f"  • Max MTM (USD): ${status['max_mtm_usd']:.2f}")
        print(f"  • Min MTM (USD): ${status['min_mtm_usd']:.2f}")
        print(f"  • Max USD Balance: ${status['max_balance_usd']:.2f}")
        print(f"  • Min USD Balance: ${status['min_balance_usd']:.2f}")
        print(f"  • Max BTC Balance: {status['max_balance_btc']:.8f}")
        print(f"  • Min BTC Balance: {status['min_balance_btc']:.8f}")

        pos_info = status.get('position_info', {})
        print("\nPosition Details:")
        print(f"  • Direction:  {pos_str}")
        print(f"  • Current Price:  ${pos_info.get('current_price', 0.0):.2f}")
        print(f"  • Entry Price:    ${pos_info.get('entry_price', 0.0):.2f}")
        if status['position'] == 1:
            print(f"  • Position Size (BTC): {pos_info.get('position_size_btc', 0.0):.8f}")
            print(f"  • Position Value (USD): ${pos_info.get('position_size_usd', 0.0):.2f}")
        elif status['position'] == -1:
            print(f"  • Short Size (BTC): {pos_info.get('position_size_btc', 0.0):.8f} (negative means short)")
            print(f"  • USD Held:         ${pos_info.get('position_size_usd', 0.0):.2f}")
        else:
            print("  • Neutral position, no open BTC or short.")
        print(f"  • Unrealized PnL:  ${pos_info.get('unrealized_pnl', 0.0):.2f}")

        # Print proximity measures with a quick user note
        ma_prox = status.get('ma_signal_proximity', None)
        if ma_prox is not None:
            print(f"  • MA Signal Proximity: {ma_prox:.4f} (closer to 0 => near crossing)")

        rsi_prox = status.get('rsi_signal_proximity', None)
        if rsi_prox is not None:
            print(f"  • RSI Signal Proximity: {rsi_prox:.4f} (0 => fully oversold/overbought)")

        print("\nTrading Statistics:")
        print(f"  • Total Trades: {status['trades_executed']}")
        print(f"  • Profitable Trades: {status['profitable_trades']}")
        print(f"  • Win Rate: {status['win_rate']:.1f}%")

        if status['trades_executed'] > 0:
            print(f"  • Avg Profit/Trade: ${status['average_profit_per_trade']:.2f}")
            print(f"  • Avg Fee/Trade: ${status.get('average_fee_per_trade', 0.0):.2f}")
            print(f"  • Risk/Reward Ratio: {status.get('risk_reward_ratio', 0.0):.2f}")

        if status['last_trade']:
            print("\nLast Trade Info:")
            print(f"  • Reason: {status['last_trade']}")
            print(f"  • Data Source: {status['last_trade_data_source']}")
            print(f"  • Signal Time: {status['last_trade_signal_timestamp']}")

        print("\nTechnical Analysis:")
        if status.get('next_trigger'):
            print(f"  • {status['next_trigger']}")
        if status.get('current_trends'):
            print("  • Current Trends:")
            for k, v in status['current_trends'].items():
                print(f"    ◦ {k}: {v}")
        if 'ma_difference' in status and status['ma_difference'] is not None:
            print(f"  • MA Difference: {status['ma_difference']:.4f}")
        if 'ma_slope_difference' in status and status['ma_slope_difference'] is not None:
            print(f"  • MA Slope Difference: {status['ma_slope_difference']:.4f}")
        if 'short_ma_momentum' in status['current_trends']:
            print(f"  • Short MA Momentum: {status['current_trends'].get('Short_MA_Slope')}")
        if 'long_ma_momentum' in status['current_trends']:
            print(f"  • Long MA Momentum: {status['current_trends'].get('Long_MA_Slope')}")
        if 'momentum_alignment' in status:
            print(f"  • Momentum Alignment: {status['momentum_alignment']}")

        if 'last_rsi' in status:
            print(f"  • Last RSI: {status['last_rsi']:.2f} "
                  f"(window={status.get('rsi_window',14)}, "
                  f"overbought={status.get('overbought',70)}, "
                  f"oversold={status.get('oversold',30)})")

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

    def do_exit(self, arg):
        return self.do_quit(arg)

    def stop_dash_app(self):
        # ...
        pass

    def do_chart(self, arg):
        # ...
        pass
