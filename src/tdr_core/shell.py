###############################################################################
# src/tdr_core/shell.py
###############################################################################
# Full File Path: src/tdr_core/shell.py
#
# CHANGES (EXPLANATION):
#   1) Previously, we only mentioned the scenario of being short while 
#      strategy indicates short and skipping a redundant trade. But we 
#      must do the same for long positions:
#        - If the user is already long, and the strategy also says go long, 
#          we skip forcing an immediate buy. 
#        - In other words, if hist_position == desired_position (whether +1 
#          or -1), we skip the forced trade.
#   2) We have already implemented "if hist_position == desired_position: skip", 
#      which covers *both* short->short and long->long cases. 
#   3) We preserve all other code and comments. We only reiterate in the docstring 
#      that this logic applies for both short->short and long->long. 
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

# We'll need references to modules from our codebase:
from tdr_core.order_placer import OrderPlacer
from tdr_core.strategies import MACrossoverStrategy, RSITradingStrategy
from tdr_core.data_manager import CryptoDataManager
from tdr_core.trade import Trade

# Original references from tdr.py
from utils.analysis import analyze_data, run_trading_system
from data.loader import create_metadata_file, parse_log_file

def determine_rsi_position(df, rsi_window=14, overbought=70, oversold=30):
    """
    A minimal function to guess the final RSI-based position from the last row 
    of DF. If RSI < oversold => position=1, if RSI>overbought => position=-1, else 0.

    This logic ensures that if we are already 'long' (1) or 'short' (-1) 
    per the last RSI signal, we skip forcing an immediate trade if the user 
    also chooses the same position. This covers both short->short and long->long.
    """
    if df.empty or len(df) < rsi_window:
        return 0
    from indicators.technical_indicators import ensure_datetime_index, calculate_rsi
    df_copy = ensure_datetime_index(df.copy())
    df_copy = df_copy.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'trades': 'sum',
        'timestamp': 'last'
    }).dropna()
    if len(df_copy) < rsi_window:
        return 0

    df_copy = calculate_rsi(df_copy, window=rsi_window, price_col='close')
    last_rsi = df_copy.iloc[-1]['RSI']
    if last_rsi < oversold:
        return 1
    elif last_rsi > overbought:
        return -1
    return 0


###############################################################################
class CryptoShell(cmd.Cmd):
    """
    An interactive command-based shell for controlling the Crypto trading system.
    """
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
            'price': 'price btcusd',
            'range': 'range btcusd 30',
            'buy': 'buy btcusd 0.001',
            'sell': 'sell btcusd 0.001',
            'candles': 'candles btcusd',
            'ticker': 'ticker btcusd',
            'example': 'example price',
            'limit_buy': 'limit_buy btcusd 0.001 50000 daily_order=true',
            'limit_sell': 'limit_sell btcusd 0.001 60000 ioc_order=true',
            'auto_trade': 'auto_trade 2.47btc long',
            'stop_auto_trade': 'stop_auto_trade',
            'status': 'status [long]',
            'chart': 'chart btcusd 1H'
        }

        # Register callbacks
        self.data_manager.add_candlestick_observer(self.candlestick_callback)
        self.data_manager.add_trade_observer(self.trade_callback)

    def emptyline(self):
        pass

    def do_example(self, arg):
        """
        Show an example usage of a command: example <command>
        """
        command = arg.strip().lower()
        if command in self.examples:
            print("Example usage of '{}':".format(command))
            print("  {}".format(self.examples[command]))
        else:
            print("No example for '{}'. Available commands:".format(command))
            print(", ".join(self.examples.keys()))

    def do_price(self, arg):
        """
        Show current price for a symbol, plus the last WebSocket update timestamp:
          price <symbol>
        """
        symbol = arg.strip().lower()
        if not symbol:
            print("Usage: price <symbol>")
            return
        price = self.data_manager.get_current_price(symbol)
        if price is not None:
            last_update_time = self.data_manager.last_trade_time.get(symbol)
            if last_update_time:
                update_str = last_update_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                update_str = "unknown (no trades yet)"
            print(f"Current price of {symbol}: ${price:.2f} (last update: {update_str})")
        else:
            print(f"No data for {symbol}")

    def do_range(self, arg):
        """
        Show min and max price in last N minutes: range <symbol> <minutes>
        """
        args = arg.split()
        if len(args) != 2:
            print("Usage: range <symbol> <minutes>")
            return
        symbol, minutes = args[0].lower(), int(args[1])
        min_price, max_price = self.data_manager.get_price_range(symbol, minutes)
        if min_price is not None and max_price is not None:
            print(f"Price range for {symbol} over last {minutes} minutes:")
            print(f"Min: ${min_price:.2f}, Max: ${max_price:.2f}")
        else:
            print(f"No data for {symbol} in that timeframe")

    def do_buy(self, arg):
        """
        Place a market buy order: buy <symbol> <amount>
        """
        args = arg.split()
        if len(args) != 2:
            print("Usage: buy <symbol> <amount>")
            return
        symbol, amount = args[0].lower(), float(args[1])
        result = self.order_placer.place_order("market-buy", symbol, amount)
        print(json.dumps(result, indent=2))

    def do_sell(self, arg):
        """
        Place a market sell order: sell <symbol> <amount>
        """
        args = arg.split()
        if len(args) != 2:
            print("Usage: sell <symbol> <amount>")
            return
        symbol, amount = args[0].lower(), float(args[1])
        result = self.order_placer.place_order("market-sell", symbol, amount)
        print(json.dumps(result, indent=2))

    def do_candles(self, arg):
        """
        Toggle 1-minute candlestick printout: candles <symbol>
        """
        symbol = arg.strip().lower()
        if not symbol:
            print("Usage: candles <symbol>")
            return
        if symbol in self.candlestick_output:
            del self.candlestick_output[symbol]
            print(f"Stopped 1-minute candlestick output for {symbol}")
        else:
            self.candlestick_output[symbol] = True
            print(f"Started 1-minute candlestick output for {symbol}")

    def do_ticker(self, arg):
        """
        Toggle real-time trade output: ticker <symbol>
        """
        symbol = arg.strip().lower()
        if not symbol:
            print("Usage: ticker <symbol>")
            return
        if symbol in self.ticker_output:
            del self.ticker_output[symbol]
            print(f"Stopped real-time trade output for {symbol}")
        else:
            self.ticker_output[symbol] = True
            print(f"Started real-time trade output for {symbol}")

    def candlestick_callback(self, symbol, minute, candle):
        """
        Callback for candlestick updates if toggled on via candles <symbol>.
        """
        if symbol in self.candlestick_output:
            ts_str = datetime.fromtimestamp(candle['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{symbol} - {ts_str}: "
                  f"Open={candle['open']:.2f}, High={candle['high']:.2f}, "
                  f"Low={candle['low']:.2f}, Close={candle['close']:.2f}, "
                  f"Volume={candle['volume']}, Trades={candle['trades']}")

    def trade_callback(self, symbol, price, timestamp, trade_reason):
        """
        Callback for trade updates if toggled on via ticker <symbol>.
        """
        if symbol in self.ticker_output:
            ts_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{symbol} - {ts_str}: Price=${price:.2f}")

    def do_verbose(self, arg):
        """
        Enable verbose logging to console or to a specified log file: verbose [logfile]
        """
        arg = arg.strip()
        if not arg:
            if not self.verbose:
                self.logger.setLevel(logging.DEBUG)
                debug_handlers = [
                    h for h in self.logger.handlers
                    if isinstance(h, logging.StreamHandler) and h.level == logging.DEBUG
                ]
                if not debug_handlers:
                    debug_stream_handler = logging.StreamHandler(sys.stderr)
                    debug_stream_handler.setLevel(logging.DEBUG)
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    debug_stream_handler.setFormatter(formatter)
                    self.logger.addHandler(debug_stream_handler)
                self.data_manager.set_verbose(True)
                self.verbose = True
                print("Verbose mode enabled.")
            else:
                print("Verbose mode is already enabled.")
        else:
            log_file = arg
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    self.logger.removeHandler(handler)
            try:
                log_file_path = os.path.abspath(log_file)
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                self.data_manager.set_verbose(True)
                self.verbose = True
                print(f"Verbose logs being written to {log_file_path}.")
            except Exception as e:
                print(f"Failed to open log file {log_file}: {e}")

    def parse_order_options(self, args):
        options = {}
        for arg in args:
            if '=' in arg:
                key, value = arg.split('=', 1)
                if key in ['daily_order', 'ioc_order', 'fok_order', 'moc_order', 'gtd_order']:
                    options[key] = (value.lower() == 'true')
                elif key == 'expire_time':
                    try:
                        options[key] = int(value)
                    except ValueError:
                        print(f"Invalid value for {key}: {value} (should be int).")
                elif key == 'client_order_id':
                    options[key] = value
                elif key == 'limit_price':
                    try:
                        options[key] = float(value)
                    except ValueError:
                        print(f"Invalid value for {key}: {value} (should be float).")
        return options

    def do_limit_buy(self, arg):
        """
        Place a limit buy order: limit_buy <symbol> <amount> <price> [options]
        """
        args = arg.split()
        if len(args) < 3:
            print("Usage: limit_buy <symbol> <amount> <price> [options]")
            return
        symbol, amount, price = args[0].lower(), float(args[1]), float(args[2])
        options = self.parse_order_options(args[3:])
        result = self.order_placer.place_limit_buy_order(symbol, amount, price, **options)
        print(json.dumps(result, indent=2))

    def do_limit_sell(self, arg):
        """
        Place a limit sell order: limit_sell <symbol> <amount> <price> [options]
        """
        args = arg.split()
        if len(args) < 3:
            print("Usage: limit_sell <symbol> <amount> <price> [options]")
            return
        symbol, amount, price = args[0].lower(), float(args[1]), float(args[2])
        options = self.parse_order_options(args[3:])
        result = self.order_placer.place_limit_sell_order(symbol, amount, price, **options)
        print(json.dumps(result, indent=2))

    def parse_position_str(self, pos_str):
        """
        Convert 'long'|'short'|'neutral' to +1|-1|0.
        """
        pos_str = pos_str.lower()
        if pos_str == 'long':
            return 1
        elif pos_str == 'short':
            return -1
        elif pos_str == 'neutral':
            return 0
        else:
            return None

    def do_auto_trade(self, arg):
        """
        Start auto-trading using the best strategy from best_strategy.json.
        
        Usage:
          auto_trade <amount><btc|usd> <long|short|neutral>
          
        Examples:
          auto_trade 2.47btc long
          auto_trade 234462usd short
          
        If hist_position == desired_position (e.g. short->short or long->long),
        then we skip forcing an immediate trade. Only trade if we are going in the 
        opposite direction (hist_position != desired_position).
        """
        if self.auto_trader and self.auto_trader.running:
            print("Auto-trading is already running. Stop it first.")
            return

        args_list = arg.split()
        if len(args_list) != 2:
            print("Usage: auto_trade <amount><btc|usd> <long|short|neutral>")
            return

        balance_str = args_list[0].lower()
        pos_str = args_list[1].lower()

        desired_position = self.parse_position_str(pos_str)
        if desired_position is None:
            print("Position must be 'long', 'short', or 'neutral'.")
            return

        import re
        pattern = re.compile(r'^(\d+(\.\d+)?)(btc|usd)$')
        match = pattern.match(balance_str)
        if not match:
            print("Balance argument must be like 2.47btc or 234462usd.")
            return

        amount_num = float(match.group(1))
        amount_unit = match.group(3)
        if amount_unit == 'btc' and desired_position != 1:
            print("Error: If specifying BTC balance, you must start in a 'long' position.")
            return
        if amount_unit == 'usd' and desired_position != -1:
            print("Error: If specifying USD balance, you must start in a 'short' position.")
            return

        file_path = os.path.abspath('best_strategy.json')
        if not os.path.exists(file_path):
            print(f"Error: '{file_path}' not found.")
            return

        with open(file_path, 'r') as f:
            best_strategy_params = json.load(f)

        strategy_name = best_strategy_params.get('Strategy')

        if strategy_name == 'MA':
            short_window = int(best_strategy_params.get('Short_Window', 12))
            long_window  = int(best_strategy_params.get('Long_Window', 36))
            do_live      = best_strategy_params.get('do_live_trades', False)
            max_trades_day = best_strategy_params.get('max_trades_per_day', 5)

            df = self.data_manager.get_price_dataframe('btcusd').copy()
            if 'close' not in df.columns and 'price' in df.columns:
                df.rename(columns={'price': 'close'}, inplace=True)
            if df.empty:
                hist_position = 0
            else:
                from tdr import determine_initial_position  # minimal local import
                hist_position = determine_initial_position(df, short_window, long_window)

            initial_balance_btc = 0.0
            initial_balance_usd = 0.0
            if desired_position == 1:
                initial_balance_btc = amount_num
            elif desired_position == -1:
                initial_balance_usd = amount_num

            self.auto_trader = MACrossoverStrategy(
                self.data_manager,
                short_window,
                long_window,
                amount_num,
                'btcusd',
                self.logger,
                live_trading=do_live,
                max_trades_per_day=max_trades_day,
                initial_position=desired_position,
                initial_balance_btc=initial_balance_btc,
                initial_balance_usd=initial_balance_usd
            )

            current_market_price = self.data_manager.get_current_price('btcusd') or 0.0

            # If hist_position == desired_position => skip forced immediate trade
            if desired_position == hist_position:
                self.logger.info(f"(auto_trade) MA: No forced immediate trade since desired_position == hist_position == {desired_position}.")
            else:
                # Force trade if there's a mismatch
                if desired_position == 1 and hist_position != 1 and current_market_price > 0:
                    buy_btc = amount_num / current_market_price
                    trade_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.logger.info(f"(auto_trade) Forcing immediate BUY for {buy_btc:.6f} BTC at ${current_market_price:.2f}.")
                    self.auto_trader.execute_trade(
                        "buy",
                        current_market_price,
                        trade_ts,
                        datetime.now(),
                        buy_btc
                    )
                elif desired_position == -1 and hist_position != -1 and current_market_price > 0:
                    sell_btc = amount_num / current_market_price
                    trade_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.logger.info(f"(auto_trade) Forcing immediate SELL for {sell_btc:.6f} BTC at ${current_market_price:.2f}.")
                    self.auto_trader.execute_trade(
                        "sell",
                        current_market_price,
                        trade_ts,
                        datetime.now(),
                        sell_btc
                    )

            self.auto_trader.start()
            print(f"Auto-trading started with {balance_str}, position={pos_str}, "
                  f"MA strategy (Short={short_window}, Long={long_window}), do_live_trades={do_live}")

        elif strategy_name == 'RSI':
            rsi_window = int(best_strategy_params.get('RSI_Window', 14))
            overbought = float(best_strategy_params.get('Overbought', 70))
            oversold   = float(best_strategy_params.get('Oversold', 30))
            do_live    = best_strategy_params.get('do_live_trades', False)
            max_trades_day = best_strategy_params.get('max_trades_per_day', 5)

            df = self.data_manager.get_price_dataframe('btcusd').copy()
            if 'close' not in df.columns and 'price' in df.columns:
                df.rename(columns={'price': 'close'}, inplace=True)

            # Determine a simple historical RSI-based position 
            # (long if last RSI<oversold, short if RSI>overbought, else 0)
            if df.empty:
                hist_position = 0
            else:
                from tdr_core.shell import determine_rsi_position  # local import
                hist_position = determine_rsi_position(df, rsi_window, overbought, oversold)

            initial_balance_btc = 0.0
            initial_balance_usd = 0.0
            if desired_position == 1:
                initial_balance_btc = amount_num
            elif desired_position == -1:
                initial_balance_usd = amount_num

            self.auto_trader = RSITradingStrategy(
                self.data_manager,
                rsi_window,
                overbought,
                oversold,
                amount_num,
                'btcusd',
                self.logger,
                live_trading=do_live,
                max_trades_per_day=max_trades_day,
                initial_position=desired_position,
                initial_balance_btc=initial_balance_btc,
                initial_balance_usd=initial_balance_usd
            )

            current_market_price = self.data_manager.get_current_price('btcusd') or 0.0

            # If hist_position == desired_position => skip forced immediate trade
            if desired_position == hist_position:
                self.logger.info(f"(auto_trade) RSI: No forced immediate trade since desired_position == hist_position == {desired_position}.")
            else:
                # Force trade if there's a mismatch
                if desired_position == 1 and hist_position != 1 and current_market_price > 0:
                    buy_btc = amount_num / current_market_price
                    trade_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.logger.info(f"(auto_trade) RSI: Forcing immediate BUY for {buy_btc:.6f} BTC at ${current_market_price:.2f}.")
                    self.auto_trader.execute_trade(
                        "buy",
                        current_market_price,
                        trade_ts,
                        datetime.now(),
                        buy_btc
                    )
                elif desired_position == -1 and hist_position != -1 and current_market_price > 0:
                    sell_btc = amount_num / current_market_price
                    trade_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.logger.info(f"(auto_trade) RSI: Forcing immediate SELL for {sell_btc:.6f} BTC at ${current_market_price:.2f}.")
                    self.auto_trader.execute_trade(
                        "sell",
                        current_market_price,
                        trade_ts,
                        datetime.now(),
                        sell_btc
                    )

            self.auto_trader.start()
            print(f"Auto-trading started with {balance_str}, position={pos_str}, "
                  f"RSI strategy (Window={rsi_window}, Overbought={overbought}, Oversold={oversold}), do_live_trades={do_live}")

        else:
            print(f"Best strategy is not 'MA' or 'RSI'; it's {strategy_name}.")
            print("Currently supported: 'MA', 'RSI'.")
            return

    def do_stop_auto_trade(self, arg):
        """
        Stop auto-trading if running.
        """
        if self.auto_trader and self.auto_trader.running:
            self.auto_trader.stop()
            print("Auto-trading stopped.")
        else:
            print("No auto-trading is running.")

    def do_status(self, arg):
        """
        Show status of auto-trading. Usage: status [long]
        
        By default (no arg or anything not "long"), we show a short version:
          - Position Details with direction, plus theoretical trade block if relevant.
        If user types "status long", we show the entire original block.
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

            if status['trades_executed'] == 0 and status.get('theoretical_trade'):
                t = status['theoretical_trade']
                print(f"\n  This is a theoretical trade (no actual trades yet):")
                print(f"    • Timestamp:  {t['timestamp']}")
                print(f"    • Direction:  {t['direction']}")
                print(f"    • Amount:     {t['amount']}")
                print(f"    • Theoretical? {t['theoretical']}")

            proximity = status.get('ma_signal_proximity')
            if proximity is not None:
                print(f"\n  • MA Crossover Proximity: {proximity*100:.2f}%")
                print("    (Closer to 0% means closer to flipping from short->long or long->short)")

            print("")
            return

        # Otherwise, show the full (long) status:
        print("\nAuto-Trading Status:")
        print("━"*50)
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
        if status['next_trigger']:
            print(f"  • {status['next_trigger']}")
        if status.get('current_trends'):
            print("  • Current Trends:")
            for k, v in status['current_trends'].items():
                print(f"    ◦ {k}: {v}")
        if 'ma_difference' in status and status['ma_difference'] is not None:
            print(f"  • MA Difference: {status['ma_difference']:.4f}")
        if 'ma_slope_difference' in status and status['ma_slope_difference'] is not None:
            print(f"  • MA Slope Difference: {status['ma_slope_difference']:.4f}")
        if 'short_ma_momentum' in status:
            print(f"  • Short MA Momentum: {status['short_ma_momentum']}")
        if 'long_ma_momentum' in status:
            print(f"  • Long MA Momentum: {status['long_ma_momentum']}")
        if 'momentum_alignment' in status:
            print(f"  • Momentum Alignment: {status['momentum_alignment']}")

        # RSI debug info:
        if 'last_rsi' in status:
            print(f"  • Last RSI: {status['last_rsi']:.2f} (window={status.get('rsi_window',14)}, overbought={status.get('overbought',70)}, oversold={status.get('oversold',30)})")

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
