# ----------------------------------------------------------------------------
# FULL FILE PATH: src/tdr_core/shell.py
# ----------------------------------------------------------------------------
# CHANGES MADE:
#   1) Introduced a small block in do_auto_trade() to derive the final signal
#      from the loaded best_strategy.json if it's "MA". This ensures that if
#      the best strategy's final signal is already short, we do NOT force a SELL
#      when the user says "auto_trade 192000usd short."
#   2) Preserved existing logic, code, and comments. Only minimal lines added.
#   3) INTEGRATION: Added support for the new charting module with port and alt_strategy_file support
#   4) INTEGRATION: Updated do_chart() method to support: chart <symbol> <bar_size> <port> <alt_strategy_file>
#   5) INTEGRATION: Replaced embedded run_dash_app with import from tdr_core.charting
#   6) INTEGRATION: Updated stop_dash_app to handle dynamic ports
# ----------------------------------------------------------------------------
# BUG FIXES IN THIS VERSION:
#   1) Fixed position display to never show "Neutral" - system is always LONG or SHORT
#   2) Fixed entry price calculation for short positions
# ----------------------------------------------------------------------------

import cmd
import sys
import json
import time
import logging
import threading
import requests
import os
import glob
from datetime import datetime
from flask import Flask, request
from multiprocessing import Process, Manager

# Enable tab completion
try:
    import readline
except ImportError:
    # readline not available on Windows
    pass
else:
    # Enable tab completion
    readline.parse_and_bind("tab: complete")

# We'll need references to modules from our codebase:
from tdr_core.strategies import MACrossoverStrategy, AdaptiveMultiStrategy
from tdr_core.command_interface import CommandInterface
from indicators.technical_indicators import ensure_datetime_index, add_moving_averages, generate_ma_signals

###############################################################################


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
        self.chart_port = None  # Track the port for shutdown
        self.stop_event = stop_event
        self.manager = Manager()
        self.data_manager_dict = self.manager.dict()
        self.max_trades_per_day = max_trades_per_day
        
        # Initialize command interface
        self.command_interface = None

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
            'resume_auto_trade': 'resume_auto_trade 1.55612586btc long 107374',
            'stop_auto_trade': 'stop_auto_trade',
            'status': 'status [long]',
            'chart': 'chart btcusd 1H 8051 alt_strategy-1.json',
            'summary_diagnostics': 'summary_diagnostics [filename.json]',
            'position_history': 'position_history',
            'auto_resume': 'auto_resume'
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
            print(
                f"Current price of {symbol}: ${price:.2f} (last update: {update_str})")
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
        min_price, max_price = self.data_manager.get_price_range(
            symbol, minutes)
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
            ts_str = datetime.fromtimestamp(
                candle['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{symbol} - {ts_str}: "
                  f"Open={candle['open']:.2f}, High={candle['high']:.2f}, "
                  f"Low={candle['low']:.2f}, Close={candle['close']:.2f}, "
                  f"Volume={candle['volume']}, Trades={candle['trades']}")

    def trade_callback(self, symbol, price, timestamp, trade_reason):
        """
        Callback for trade updates if toggled on via ticker <symbol>.
        """
        if symbol in self.ticker_output:
            ts_str = datetime.fromtimestamp(
                timestamp).strftime('%Y-%m-%d %H:%M:%S')
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
                    formatter = logging.Formatter(
                        '%(asctime)s - %(levelname)s - %(message)s')
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
                formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s')
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
                        print(
                            f"Invalid value for {key}: {value} (should be int).")
                elif key == 'client_order_id':
                    options[key] = value
                elif key == 'limit_price':
                    try:
                        options[key] = float(value)
                    except ValueError:
                        print(
                            f"Invalid value for {key}: {value} (should be float).")
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
        result = self.order_placer.place_limit_buy_order(
            symbol, amount, price, **options)
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
        result = self.order_placer.place_limit_sell_order(
            symbol, amount, price, **options)
        print(json.dumps(result, indent=2))

    def parse_position_str(self, pos_str):
        """
        Convert 'long'|'short' to +1|-1.
        Note: This system is never neutral (0).
        """
        pos_str = pos_str.lower()
        if pos_str == 'long':
            return 1
        elif pos_str == 'short':
            return -1
        else:
            return None

    def do_auto_trade(self, arg):
        """
        Start auto-trading using the best strategy from best_strategy.json.

        Usage:
          auto_trade <amount><btc|usd> <long|short>

        Examples:
          auto_trade 2.47btc long
          auto_trade 234462usd short
        """
        if self.auto_trader and self.auto_trader.running:
            print("Auto-trading is already running. Stop it first.")
            return

        args_list = arg.split()
        if len(args_list) != 2:
            print("Usage: auto_trade <amount><btc|usd> <long|short>")
            return

        balance_str = args_list[0].lower()
        pos_str = args_list[1].lower()

        desired_position = self.parse_position_str(pos_str)
        if desired_position is None:
            print("Position must be 'long' or 'short'.")
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
            print(
                "Error: If specifying BTC balance, you must start in a 'long' position.")
            return
        if amount_unit == 'usd' and desired_position != -1:
            print(
                "Error: If specifying USD balance, you must start in a 'short' position.")
            return

        file_path = os.path.abspath('best_strategy.json')
        if not os.path.exists(file_path):
            print(f"Error: '{file_path}' not found.")
            return

        with open(file_path, 'r') as f:
            best_strategy_params = json.load(f)

        strategy_name = best_strategy_params.get('Strategy')
        # <-- NEW: Show the current strategy
        print(f"Auto-trading initiated with strategy: {strategy_name}")

        short_window = int(best_strategy_params.get('Short_Window', 12))
        long_window = int(best_strategy_params.get('Long_Window', 36))
        do_live = best_strategy_params.get('do_live_trades', False)
        max_trades_day = best_strategy_params.get('max_trades_per_day', 5)

        # Read adaptive strategy parameters from config
        regime_threshold = float(best_strategy_params.get('regime_switch_threshold', 0.7))
        confirmation_bars = int(best_strategy_params.get('signal_confirmation_bars', 2))
        min_trade_gap = int(best_strategy_params.get('min_trade_gap_minutes', 30))
        regime_lookback = int(best_strategy_params.get('regime_lookback', 50))

# ------------------------------------------------------------------------
# NEW: Optional backwards‑compatibility switch.
# If you put  "auto_align_position": true  in best_strategy.json,
# the old "instant realignment" behaviour is retained.  Otherwise
# the programme will honour your requested starting posture and
# *not* reverse the position at start‑up.  The strategy will make
# the next move when its rules say so.
# ------------------------------------------------------------------------
        auto_align = best_strategy_params.get('auto_align_position', False)

        # Get price DataFrame for the chosen symbol
        df = self.data_manager.get_price_dataframe('btcusd').copy()
        if 'close' not in df.columns and 'price' in df.columns:
            df.rename(columns={'price': 'close'}, inplace=True)

        # Use conservative signal determination - wait for actual strategy to decide
        # Instead of creating temporary strategy, use simple MA analysis as baseline
        if not df.empty:
            df_for_analysis = ensure_datetime_index(df)
            df_resampled = df_for_analysis.resample('1H').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                'volume': 'sum', 'trades': 'sum', 'timestamp': 'last', 'source': 'last'
            }).dropna()

            if len(df_resampled) >= long_window:
                # Use simple MA crossover for initialization - let running strategy handle complexity
                df_ma = add_moving_averages(
                    df_resampled.copy(), short_window, long_window, price_col='close')
                df_ma = generate_ma_signals(df_ma)

                if not df_ma.empty:
                    hist_position = int(df_ma.iloc[-1]['MA_Signal'])
                    ma_short = df_ma.iloc[-1]['Short_MA']
                    ma_long = df_ma.iloc[-1]['Long_MA']

                    # Conservative initialization: require significant MA separation
                    ma_separation = abs(ma_short - ma_long) / \
                        ma_long if ma_long > 0 else 0
                    if ma_separation < 0.002:  # Less than 0.2% separation = too close to call
                        hist_position = 0  # Stay neutral if MAs are too close
                        self.logger.info(
                            f"MAs too close ({ma_separation:.3%} separation) - staying neutral for initialization")

                    direction_name = 'LONG' if hist_position == 1 else 'SHORT' if hist_position == -1 else 'NEUTRAL'
                    self.logger.info(
                        f"Simple MA analysis for initialization: {direction_name} (MA separation: {ma_separation:.3%})")
                    self.logger.info(
                        f"Note: Running strategy will use full adaptive logic with conservative thresholds")
                else:
                    hist_position = 0
            else:
                hist_position = 0
                self.logger.info("Insufficient data for MA analysis")
        else:
            hist_position = 0
            self.logger.info("No data available for analysis")

        initial_balance_btc = 0.0
        initial_balance_usd = 0.0
        if desired_position == 1:
            initial_balance_btc = amount_num
        elif desired_position == -1:
            initial_balance_usd = amount_num

        self.auto_trader = AdaptiveMultiStrategy(
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
            initial_balance_usd=initial_balance_usd,
            # Adaptive strategy parameters
            regime_lookback=regime_lookback,  # From config or default 50
            signal_confirmation_bars=confirmation_bars,  # From config or default 2
            min_trade_gap_minutes=min_trade_gap,  # From config or default 30
            regime_switch_threshold=regime_threshold,  # From config or default 0.7
            # Mean reversion for your whipsaw situation
            rsi_oversold=35,                 # Your RSI is 35.66!
            rsi_overbought=65,
            bb_std_dev=2.0,
            # Breakout parameters
            volume_threshold=1.5,
            macd_threshold=0.001
        )
        
        # Handle resume position tracking IMMEDIATELY after strategy creation
        if hasattr(self, '_resume_entry_price') and self._resume_entry_price:
            entry_price = self._resume_entry_price
            if desired_position == 1:  # LONG position
                # For LONG: position_size should be BTC amount, cost_basis = BTC * entry_price
                self.auto_trader.position_size = amount_num
                self.auto_trader.position_cost_basis = amount_num * entry_price
                self.auto_trader.last_trade_price = entry_price
                self.logger.info(f"Resume: Set LONG position tracking - {amount_num} BTC @ ${entry_price:.2f}, cost basis ${self.auto_trader.position_cost_basis:.2f}")
            elif desired_position == -1:  # SHORT position
                # For SHORT: position_size should be negative BTC sold, cost_basis = USD received
                btc_sold = amount_num / entry_price
                self.auto_trader.position_size = -btc_sold
                self.auto_trader.position_cost_basis = amount_num  # USD amount
                self.auto_trader.last_trade_price = entry_price
                self.logger.info(f"Resume: Set SHORT position tracking - {btc_sold:.8f} BTC sold @ ${entry_price:.2f}, holding ${amount_num:.2f} USD")
                
            # Sync to data_manager for consistent display
            if hasattr(self.data_manager, 'position_size'):
                self.data_manager.position_size = self.auto_trader.position_size
                self.data_manager.position_cost_basis = self.auto_trader.position_cost_basis
                self.data_manager.position = desired_position
                self.logger.info(f"Resume: Synced position to data_manager")
        
        # Log the auto_trade command to diagnostics
        if hasattr(self.auto_trader, 'diagnostic_logger'):
            self.auto_trader.diagnostic_logger.log_event("AUTO_TRADE_COMMAND", {
                "command": f"auto_trade {arg}",
                "parsed": {
                    "amount": amount_num,
                    "unit": amount_unit,
                    "position": pos_str,
                    "desired_position": desired_position,
                    "initial_balance_btc": initial_balance_btc,
                    "initial_balance_usd": initial_balance_usd,
                    "strategy": strategy_name,
                    "live_trading": do_live,
                    "max_trades_per_day": max_trades_day
                }
            })
 
        current_market_price = self.data_manager.get_current_price(
            'btcusd') or 0.0

        # If the user starts "long" and hist_position is also long => theoretical
        if desired_position == 1 and hist_position == 1 and current_market_price > 0:
            if self.auto_trader.position_size < 1e-8:  # i.e. 0.0
                self.auto_trader.position_size = amount_num
                self.auto_trader.position_cost_basis = amount_num * current_market_price
                self.logger.info(
                    f"(auto_trade) Setting cost basis to {self.auto_trader.position_cost_basis:.2f} "
                    f"for an initial LONG of {amount_num} BTC at ${current_market_price:.2f}."
                )
                self.auto_trader.theoretical_trade = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'direction': 'long',
                    'amount': amount_num,
                    'theoretical': True
                }

                # Log theoretical trade
                self.auto_trader.diagnostic_logger.log_event("THEORETICAL_TRADE", {
                    "reason": "User position matches system recommendation",
                    "position": "LONG",
                    "amount_btc": amount_num,
                    "entry_price": current_market_price,
                    "cost_basis": self.auto_trader.position_cost_basis,
                    "note": "No actual trade needed - positions aligned"
                })

        # If the user starts "short" and hist_position is also short => theoretical
        if desired_position == -1 and hist_position == -1 and current_market_price > 0:
            short_btc = amount_num / current_market_price
            if self.auto_trader.position_size > -1e-8 and short_btc > 0:
                self.auto_trader.position_size = -short_btc  # Should be -1.67
                # FIX: Cost basis should be BTC amount * price for proper entry price calculation
                self.auto_trader.position_cost_basis = short_btc * current_market_price
                self.logger.info(
                    f"(auto_trade) Setting position_size to {self.auto_trader.position_size:.6f} BTC and "
                    f"cost basis to {self.auto_trader.position_cost_basis:.2f} "
                    f"for an initial SHORT of {short_btc:.6f} BTC (=-{short_btc:.6f}) at ${current_market_price:.2f}."
                )
                self.auto_trader.theoretical_trade = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'direction': 'short',
                    'amount': amount_num,
                    'theoretical': True
                }

                # Log theoretical trade
                self.auto_trader.diagnostic_logger.log_event("THEORETICAL_TRADE", {
                    "reason": "User position matches system recommendation",
                    "position": "SHORT",
                    "amount_usd": amount_num,
                    "btc_equivalent": short_btc,
                    "entry_price": current_market_price,
                    "cost_basis": self.auto_trader.position_cost_basis,
                    "note": "No actual trade needed - positions aligned"
                })

        # Handle all four initialization scenarios
        if desired_position == hist_position:
            # Cases 1 & 3: Positions match - initialize tracking with THEORETICAL trade
            if desired_position == 1:  # Case 1: Both long
                self.auto_trader.position = 1
                self.auto_trader.position_size = amount_num
                self.auto_trader.position_cost_basis = amount_num * current_market_price
                self.auto_trader.balance_btc = amount_num
                self.auto_trader.balance_usd = 0.0

                # Set theoretical trade for entry price tracking
                self.auto_trader.theoretical_trade = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'direction': 'long',
                    'amount': amount_num,
                    'entry_price': current_market_price,
                    'theoretical': True
                }
                self.logger.info(
                    f"Case 1: LONG matches system. Theoretical entry: {amount_num:.8f} BTC @ ${current_market_price:.2f}")

            elif desired_position == -1:  # Case 3: Both short
                short_btc = amount_num / current_market_price
                self.auto_trader.position = -1
                # FIX: Track short position properly
                self.auto_trader.position_size = -short_btc  # Negative BTC for short
                # Cost basis = BTC sold * price for correct entry calculation
                self.auto_trader.position_cost_basis = short_btc * current_market_price
                self.auto_trader.balance_btc = 0.0
                self.auto_trader.balance_usd = amount_num

                # Set theoretical trade for entry price tracking
                self.auto_trader.theoretical_trade = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'direction': 'short',
                    'amount': amount_num,
                    'entry_price': current_market_price,
                    'theoretical': True
                }
                self.logger.info(
                    f"Case 3: SHORT matches system. Theoretical entry: ${amount_num:.2f} @ ${current_market_price:.2f}")

        else:
# ------------------------------------------------------------------------
# Positions differ between what you asked for (desired_position) and what
# the last MA snapshot suggested (hist_position).
#
# • If auto_align_position == False (the new default) we *do not* fire an
#   immediate reversing trade.  Instead we book your holdings as a
#   theoretical entry and let the adaptive strategy decide when (or if)
#   to flip.
#
# • If auto_align_position == True the old behaviour is preserved.
# ------------------------------------------------------------------------
            if not auto_align:
                self.logger.info(
                    "Initial desired position differs from MA suggestion, "
                    "but 'auto_align_position' is False.  Starting in "
                    "theoretical mode without an immediate hedge; the "
                    "running strategy will realign organically."
                )

                # Treat the mismatch as if positions already match
                hist_position = desired_position

                # --- Theoretical LONG initialisation (mirrors Case 1) ---
                if desired_position == 1:
                    self.auto_trader.position = 1
                    self.auto_trader.position_size = amount_num
                    self.auto_trader.position_cost_basis = amount_num * current_market_price
                    self.auto_trader.balance_btc = amount_num
                    self.auto_trader.balance_usd = 0.0
                    self.auto_trader.theoretical_trade = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'direction': 'long',
                        'amount': amount_num,
                        'entry_price': current_market_price,
                        'theoretical': True
                    }

                # --- Theoretical SHORT initialisation (mirrors Case 3) ---
                elif desired_position == -1:
                    short_btc = amount_num / current_market_price
                    self.auto_trader.position = -1
                    self.auto_trader.position_size = 0.0
                    self.auto_trader.position_cost_basis = amount_num
                    self.auto_trader.balance_btc = 0.0
                    self.auto_trader.balance_usd = amount_num
                    self.auto_trader.theoretical_trade = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'direction': 'short',
                        'amount': amount_num,
                        'entry_price': current_market_price,
                        'theoretical': True
                    }

            # --- Original realignment logic (only if auto_align is True) -------------
            else:
                # Auto-align is True, execute realignment
                if desired_position == 1 and hist_position == -1:
                    # Case 2: User long, system says short - sell all BTC
                    self.auto_trader.position = -1
                    self.auto_trader.balance_btc = amount_num
                    self.auto_trader.balance_usd = 0.0
                    self.auto_trader.position_size = 0.0  # Will be set by trade execution
                    self.auto_trader.position_cost_basis = 0.0  # Will be set by trade execution

                    self.logger.info(
                        f"Case 2: User LONG but system says SHORT. Selling {amount_num:.8f} BTC")

                    # Log position mismatch and corrective trade
                    self.auto_trader.diagnostic_logger.log_event("POSITION_CORRECTION", {
                        "reason": "User position disagrees with system",
                        "user_position": "LONG",
                        "system_position": "SHORT",
                        "action": "SELL all BTC to align with system",
                        "amount_btc": amount_num,
                        "current_price": current_market_price
                    })

                    trade_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.auto_trader.execute_trade(
                        "sell", current_market_price, trade_ts, datetime.now(), amount_num)

                elif desired_position == -1 and hist_position == 1:
                    # Case 4: User short, system says long - execute 3-part buy
                    short_btc = amount_num / current_market_price
                    self.auto_trader.position = 1
                    self.auto_trader.balance_btc = 0.0  # Starting short (USD only)
                    self.auto_trader.balance_usd = amount_num
                    self.auto_trader.position_size = 0.0  # Will be set by trade execution
                    self.auto_trader.position_cost_basis = 0.0  # Will be set by trade execution

                    self.logger.info(
                        f"Case 4: User SHORT but system says LONG. Executing 3-part buy from ${amount_num:.2f}")
                        
                    # Log position mismatch and corrective trade
                    self.auto_trader.diagnostic_logger.log_event("POSITION_CORRECTION", {
                        "reason": "User position disagrees with system",
                        "user_position": "SHORT",
                        "system_position": "LONG",
                        "action": "BUY in 3 parts to align with system",
                        "amount_usd": amount_num,
                        "btc_to_buy": short_btc,
                        "current_price": current_market_price,
                        "note": "Will execute 3 separate buys to work around 90% rule"
                    })

                    trade_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.auto_trader.buy_in_three_parts(
                        current_market_price, trade_ts, datetime.now())
                    
        # Log initial status after auto_trade setup
        self._log_full_status_to_diagnostics()

        self.auto_trader.start()
        # Fix: Use proper capitalization for position display
        position_display = {1: 'LONG', -1: 'SHORT'}.get(desired_position, 'UNKNOWN')
        print(f"Auto-trading started with {balance_str}, position={position_display}, "
              f"MA strategy (Short={short_window}, Long={long_window}), do_live_trades={do_live}")

    def do_stop_auto_trade(self, arg):
        """
        Stop auto-trading if running.
        """
        if self.auto_trader and self.auto_trader.running:
            self.auto_trader.stop()
            print("Auto-trading stopped.")
        else:
            print("No auto-trading is running.")

    def do_resume_auto_trade(self, arg):
        """
        Resume auto-trading from saved state or with manual parameters.
        
        Usage: 
          resume_auto_trade                              # Use saved resume-auto-trade.json
          resume_auto_trade 1.55612586btc long 107374   # Manual LONG position
          resume_auto_trade 167500usd short 107263      # Manual SHORT position
          
        If no arguments provided, reads from resume-auto-trade.json and asks for confirmation.
        """
        try:
            parts = arg.strip().split() if arg.strip() else []
            
            # If no arguments, try to load from resume file
            if len(parts) == 0:
                import json
                import os
                resume_file = os.path.abspath('resume-auto-trade.json')
                
                if not os.path.exists(resume_file):
                    print(f"No resume file found at {resume_file}")
                    print("Please provide manual parameters or ensure auto-trading has saved state.")
                    return
                    
                # Load and display resume data
                with open(resume_file, 'r') as f:
                    resume_data = json.load(f)
                
                print("\n" + "="*60)
                print("RESUME AUTO-TRADE FROM SAVED STATE")
                print("="*60)
                print(f"Position:      {resume_data['position']}")
                print(f"Amount:        {resume_data['amount']:.8f} {resume_data['unit'].upper()}")
                print(f"Entry Price:   ${resume_data['entry_price']:.2f}")
                print(f"Current Price: ${resume_data['current_price']:.2f}")
                print(f"Unrealized P&L: ${resume_data['unrealized_pnl']:.2f}")
                print(f"Last Updated:  {resume_data['timestamp']}")
                print(f"\nCommand: {resume_data['command']}")
                print("="*60)
                
                # Ask for confirmation
                response = input("\nDo you want to resume with these values? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    print("Resume cancelled.")
                    return
                
                # Execute the saved command
                amount_str = f"{resume_data['amount']:.8f}{resume_data['unit']}"
                position_str = resume_data['position'].lower()
                entry_price_str = str(int(resume_data['entry_price']))
                parts = [amount_str, position_str, entry_price_str]
            
            elif len(parts) != 3:
                print("Usage: resume_auto_trade [<amount><unit> <position> <entry_price>]")
                print("Examples:")
                print("  resume_auto_trade                        # Use saved state")
                print("  resume_auto_trade 1.55612586btc long 107374")
                print("  resume_auto_trade 167500usd short 107263")
                return
                
            amount_str, position_str, entry_price_str = parts
            entry_price = float(entry_price_str)
            
            # Parse amount and unit
            import re
            match = re.match(r'([0-9.]+)(btc|usd)', amount_str.lower())
            if not match:
                print(f"Invalid amount format: {amount_str}")
                print("Use format like '1.5btc' or '167500usd'")
                return
                
            amount = float(match.group(1))
            unit = match.group(2)
            position_str = position_str.lower()
            
            # Validate position matches unit
            if unit == 'btc' and position_str != 'long':
                print("Error: BTC amount requires 'long' position")
                return
            elif unit == 'usd' and position_str != 'short':
                print("Error: USD amount requires 'short' position")
                return
                
            # Call auto_trade with special resume flag
            auto_trade_cmd = f"{amount_str} {position_str}"
            print(f"Resuming auto-trade: {auto_trade_cmd} with entry price ${entry_price:.2f}")
            
            # Store the entry price for the auto_trade to use
            self._resume_entry_price = entry_price
            
            # Execute auto_trade
            self.do_auto_trade(auto_trade_cmd)
            
            # Position tracking is now handled inside do_auto_trade when _resume_entry_price is set
            if self.auto_trader:
                print(f"✅ Position tracking updated with entry price ${entry_price:.2f}")
                
            # Clean up
            if hasattr(self, '_resume_entry_price'):
                delattr(self, '_resume_entry_price')
                
        except Exception as e:
            print(f"Error in resume_auto_trade: {e}")
            import traceback
            traceback.print_exc()

    def do_fix_position_tracking(self, arg):
        """
        Fix position tracking when entry price is incorrect.
        
        Usage: fix_position_tracking <entry_price>
        Example: fix_position_tracking 106950
        """
        if not self.auto_trader:
            print("No auto trader running")
            return
            
        try:
            entry_price = float(arg.strip())
            current_price = self.data_manager.get_current_price('btcusd') or 0.0
            
            if self.auto_trader.position == -1:  # SHORT position
                # For SHORT: Calculate BTC sold based on USD balance and entry price
                usd_balance = self.auto_trader.balance_usd
                btc_sold = usd_balance / entry_price
                
                self.auto_trader.position_size = -btc_sold
                self.auto_trader.position_cost_basis = usd_balance
                self.auto_trader.last_trade_price = entry_price
                
                # Calculate actual P&L
                pnl = (entry_price - current_price) * btc_sold
                
                print(f"✅ Fixed SHORT position tracking:")
                print(f"   Entry: ${entry_price:.2f}")
                print(f"   BTC sold: {btc_sold:.8f}")
                print(f"   USD held: ${usd_balance:.2f}")
                print(f"   Current price: ${current_price:.2f}")
                print(f"   Actual P&L: ${pnl:.2f}")
                
            elif self.auto_trader.position == 1:  # LONG position
                # For LONG: position_size is BTC amount, cost_basis = BTC * entry_price
                btc_balance = self.auto_trader.balance_btc
                
                self.auto_trader.position_size = btc_balance
                self.auto_trader.position_cost_basis = btc_balance * entry_price
                self.auto_trader.last_trade_price = entry_price
                
                # Calculate actual P&L
                pnl = (current_price - entry_price) * btc_balance
                
                print(f"✅ Fixed LONG position tracking:")
                print(f"   Entry: ${entry_price:.2f}")
                print(f"   BTC held: {btc_balance:.8f}")
                print(f"   Current price: ${current_price:.2f}")
                print(f"   Actual P&L: ${pnl:.2f}")
                
        except Exception as e:
            print(f"Error fixing position: {e}")
            import traceback
            traceback.print_exc()

    def do_save_resume_state(self, arg):
        """
        Manually save the current position state to resume-auto-trade.json.
        This is also done automatically after trades and every 10 minutes.
        
        Usage: save_resume_state
        """
        if not self.auto_trader:
            print("No auto trader running")
            return
            
        try:
            self.auto_trader.save_resume_state()
            print("✅ Resume state saved to resume-auto-trade.json")
            
            # Show what was saved
            import json
            import os
            resume_file = os.path.abspath('resume-auto-trade.json')
            if os.path.exists(resume_file):
                with open(resume_file, 'r') as f:
                    data = json.load(f)
                print(f"\nSaved position: {data['position']} {data['amount']:.8f} {data['unit']} @ ${data['entry_price']:.2f}")
                print(f"Resume command: {data['command']}")
        except Exception as e:
            print(f"Error saving resume state: {e}")

    def do_position_history(self, arg):
        """
        Query position history from the server.
        Shows current position, last saved position, and recent history.
        
        Usage: position_history
        """
        if self.mode == 'server':
            print("This command is only available in client mode")
            return
            
        try:
            response = requests.get(f"{self.server_url}/api/position_history", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Show current position if active
                if data.get('current_position') and data['current_position'].get('active'):
                    pos = data['current_position']
                    print("\n📍 Current Active Position:")
                    print(f"  • Position: {pos['position'].upper()}")
                    print(f"  • Amount: {pos['amount']:.8f}")
                    print(f"  • Entry Price: ${pos['entry_price']:.2f}")
                    print(f"  • Current Price: ${pos['current_price']:.2f}")
                    print(f"  • Unrealized P&L: ${pos['unrealized_pnl']:.2f}")
                
                # Show last saved position
                if data.get('last_position'):
                    last = data['last_position']
                    print("\n💾 Last Saved Position:")
                    print(f"  • Time: {last['timestamp']}")
                    print(f"  • Position: {last['position']}")
                    print(f"  • Amount: {last['amount']:.8f} {last['unit']}")
                    print(f"  • Entry Price: ${last['entry_price']:.2f}")
                    print(f"  • Resume Command: {last['command']}")
                
                # Show recent history
                if data.get('history'):
                    print("\n📊 Recent Position History:")
                    for i, entry in enumerate(reversed(data['history'][-5:]), 1):
                        print(f"\n  [{i}] {entry['timestamp']}")
                        print(f"      {entry['position']} {entry['amount']:.8f} {entry['unit']} @ ${entry['entry_price']:.2f}")
                        if 'unrealized_pnl' in entry:
                            print(f"      P&L: ${entry['unrealized_pnl']:.2f}")
            else:
                print(f"Error: Server returned status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to server: {e}")
        except Exception as e:
            print(f"Error: {e}")

    def do_auto_resume(self, arg):
        """
        Automatically resume trading based on the last saved position from the server.
        This will query the server's position history and execute the resume command.
        
        Usage: auto_resume
        
        Note: This command will be ignored if auto-trading is already active.
        """
        if self.mode == 'server':
            # For server mode, check local files
            if self.auto_trader:
                print("Auto-trader is already running. Resume command ignored.")
                return
                
            try:
                import os
                resume_file = os.path.abspath('resume-auto-trade.json')
                if os.path.exists(resume_file):
                    with open(resume_file, 'r') as f:
                        data = json.load(f)
                    
                    print(f"Found saved position: {data['position']} {data['amount']:.8f} {data['unit']} @ ${data['entry_price']:.2f}")
                    print(f"Executing: {data['command']}")
                    
                    # Extract the command arguments
                    parts = data['command'].split()
                    if len(parts) >= 4 and parts[0] == 'resume_auto_trade':
                        resume_args = ' '.join(parts[1:])
                        self.do_resume_auto_trade(resume_args)
                    else:
                        print("Error: Invalid resume command format")
                else:
                    print("No saved position found")
                    
            except Exception as e:
                print(f"Error: {e}")
        else:
            # Client mode - query server
            if self.auto_trader:
                print("Auto-trader is already running. Resume command ignored.")
                return
                
            try:
                response = requests.get(f"{self.server_url}/api/position_history", timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check if auto-trader is already running on server
                    if data.get('current_position') and data['current_position'].get('active'):
                        print("Auto-trader is already running on server. Resume command ignored.")
                        return
                    
                    # Use last saved position
                    if data.get('last_position'):
                        last = data['last_position']
                        print(f"\n🔄 Auto-resuming from last position:")
                        print(f"  • Time: {last['timestamp']}")
                        print(f"  • Position: {last['position']} {last['amount']:.8f} {last['unit']} @ ${last['entry_price']:.2f}")
                        print(f"  • Command: {last['command']}")
                        
                        # Extract the command arguments
                        parts = last['command'].split()
                        if len(parts) >= 4 and parts[0] == 'resume_auto_trade':
                            resume_args = ' '.join(parts[1:])
                            print("\nExecuting resume command...")
                            self.do_resume_auto_trade(resume_args)
                        else:
                            print("Error: Invalid resume command format in saved data")
                    else:
                        print("No saved position found on server")
                else:
                    print(f"Error: Server returned status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Error connecting to server: {e}")
            except Exception as e:
                print(f"Error: {e}")

    def do_set_trade_limit(self, arg):
        """
        Temporarily adjust the daily trade limit for the auto trader.
        This change applies only to the current day and resets to default on the next day.
        
        Usage: set_trade_limit <number>
        Example: set_trade_limit 10
        """
        if not self.auto_trader:
            print("No auto trader running")
            return
            
        try:
            new_limit = int(arg.strip())
            if new_limit < 1:
                print("Trade limit must be at least 1")
                return
                
            old_limit = self.auto_trader.max_trades_per_day
            default_limit = self.auto_trader.default_max_trades_per_day
            self.auto_trader.max_trades_per_day = new_limit
            
            print(f"✅ Daily trade limit changed from {old_limit} to {new_limit}")
            print(f"   Default limit: {default_limit} (will reset tomorrow)")
            print(f"   Trades today: {self.auto_trader.trade_count_today}")
            print(f"   Remaining: {new_limit - self.auto_trader.trade_count_today}")
            
        except ValueError:
            print("❌ Invalid number. Usage: set_trade_limit <number>")
        except Exception as e:
            print(f"❌ Error setting trade limit: {e}")
    
    def do_enable_commands(self, arg):
        """
        Enable external command interface for Claude Code or other systems.
        This creates a command queue in the 'commands/' directory.
        
        Usage: enable_commands
        """
        if self.command_interface is None:
            self.command_interface = CommandInterface(self, self.logger)
            self.command_interface.start()
            print("✅ Command interface enabled")
            print("   Monitoring: commands/pending/")
            print("   Processed commands: commands/processed/")
            print("   Failed commands: commands/failed/")
            print("\nTo send commands externally:")
            print("   python send_tdr_command.py status")
            print("   python send_tdr_command.py strategy_diagnostics")
        else:
            print("Command interface is already enabled")
    
    def do_disable_commands(self, arg):
        """
        Disable external command interface.
        
        Usage: disable_commands
        """
        if self.command_interface:
            self.command_interface.stop()
            self.command_interface = None
            print("✅ Command interface disabled")
        else:
            print("Command interface is not enabled")

    def do_fix_position(self, arg):
        """
        Manually fix position tracking when entry price is incorrect.
        Usage: fix_position <entry_price>
        Example: fix_position 107263
        
        This corrects the position_cost_basis to match the actual entry price.
        """
        if not self.auto_trader:
            print("Auto-trader is not active.")
            return
            
        try:
            entry_price = float(arg.strip())
            if entry_price <= 0:
                print("Entry price must be positive.")
                return
                
            status = self.auto_trader.get_status()
            
            if status['position'] == -1:
                # For short position, calculate BTC amount from USD balance
                if hasattr(self.auto_trader, 'balance_usd') and self.auto_trader.balance_usd > 0:
                    # Estimate BTC sold based on current USD balance and entry price
                    btc_sold = self.auto_trader.balance_usd / entry_price
                    self.auto_trader.position_size = -btc_sold
                    self.auto_trader.position_cost_basis = btc_sold * entry_price
                    
                    print(f"✅ Fixed SHORT position tracking:")
                    print(f"   Entry Price: ${entry_price:.2f}")
                    print(f"   BTC Sold: {btc_sold:.8f}")
                    print(f"   USD Held: ${self.auto_trader.balance_usd:.2f}")
                    print(f"   Cost Basis: ${self.auto_trader.position_cost_basis:.2f}")
                    
                    # Log the correction
                    self.auto_trader.diagnostic_logger.log_position_anomaly(
                        "Manual position correction",
                        {"entry_price": entry_price, "btc_sold": btc_sold, "reason": "User correction"}
                    )
                else:
                    print("Cannot determine position size from current balances.")
            else:
                print("Position correction only supported for SHORT positions currently.")
                
        except ValueError:
            print("Invalid entry price. Usage: fix_position <entry_price>")

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
        # BUG FIX: Never show "Neutral" - this system is always LONG or SHORT
        pos_str = {1: 'Long', -1: 'Short'}.get(status['position'], 'Unknown')

        # If user asked for the short version (default):
        if not show_full:
            print("\nPosition Details (Short View):")
            print("━"*50)
            print(f"  • Direction: {pos_str}")
            pos_info = status.get('position_info', {})
            print(
                f"  • Current Price:  ${pos_info.get('current_price', 0.0):.2f}")
            print(
                f"  • Entry Price:    ${pos_info.get('entry_price', 0.0):.2f}")

            if status['position'] == 1:
                print(
                    f"  • Position Size (BTC): {pos_info.get('position_size_btc', 0.0):.8f}")
                print(
                    f"  • Position Value (USD): ${pos_info.get('position_size_usd', 0.0):.2f}")
            elif status['position'] == -1:
                print(
                    f"  • Short Position (holding USD): ${pos_info.get('position_size_usd', 0.0):.2f}")
                # BUG FIX: Show proper BTC equivalent for short positions
                if pos_info.get('entry_price', 0) > 0:
                    btc_equivalent = pos_info.get('position_size_usd', 0.0) / pos_info.get('entry_price', 1)
                    print(
                        f"  • BTC Equivalent: {btc_equivalent:.8f} BTC")
            else:
                # This should never happen
                print("  • ERROR: System in undefined state")

            print(
                f"  • Unrealized PnL:  ${pos_info.get('unrealized_pnl', 0.0):.2f}")

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
                print(
                    "    (Closer to 0% means closer to flipping from short->long or long->short)")

            # Show adaptive strategy info in short view
            if hasattr(self.auto_trader, 'current_regime'):
                print(f"\n  🎯 Adaptive Strategy:")
                print(
                    f"  • Market Regime: {self.auto_trader.current_regime.upper()}")
                print(
                    f"  • Confidence: {self.auto_trader.regime_confidence:.1%}")
                print(
                    f"  • Active Strategy: {self.auto_trader.active_strategy.upper()}")
                if self.auto_trader.strategy_switches_today > 0:
                    print(
                        f"  • Strategy Switches Today: {self.auto_trader.strategy_switches_today}")

                # Show current strategy performance
                current_strategy_perf = self.auto_trader.strategy_performance.get(
                    self.auto_trader.active_strategy, {})
                trades = current_strategy_perf.get('trades', 0)
                if trades > 0:
                    print(
                        f"  • {self.auto_trader.active_strategy.upper()} Performance: {trades} trades executed")

            print("")
            return

        # Otherwise, show the full (long) status:
        print("\nAuto-Trading Status:")
        print("━"*50)
        print(f"  • Running: {status['running']}")
        # Redefine pos_str for the full status section
        pos_str = {1: 'Long', -1: 'Short'}.get(status['position'], 'Unknown')
        print(f"  • Position: {pos_str}")
        print(
            f"  • Daily Trades: {status['trade_count_today']}/{self.auto_trader.max_trades_per_day}")
        print(
            f"  • Remaining Trades Today: {status['remaining_trades_today']}")

        print("\nAccount Balances & Performance:")
        print(f"  • Initial USD Balance: ${status['initial_balance_usd']:.2f}")
        print(f"  • Initial BTC Balance: {status['initial_balance_btc']:.8f}")
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
            print(
                f"  • Position Size (BTC): {pos_info.get('position_size_btc', 0.0):.8f}")
            print(
                f"  • Position Value (USD): ${pos_info.get('position_size_usd', 0.0):.2f}")
        elif status['position'] == -1:
            print(
                f"  • Short Position (holding USD): ${pos_info.get('position_size_usd', 0.0):.2f}")
            # BUG FIX: Show proper BTC equivalent for short positions
            if pos_info.get('entry_price', 0) > 0:
                btc_equivalent = pos_info.get('position_size_usd', 0.0) / pos_info.get('entry_price', 1)
                print(
                    f"  • BTC Equivalent: {btc_equivalent:.8f} BTC")
        else:
            # This should never happen
            print("  • ERROR: System in undefined state")
        print(
            f"  • Unrealized PnL:  ${pos_info.get('unrealized_pnl', 0.0):.2f}")

        print("\nTrading Statistics:")
        print(f"  • Total Trades: {status['trades_executed']}")
        print(f"  • Profitable Trades: {status['profitable_trades']}")
        print(f"  • Win Rate: {status['win_rate']:.1f}%")

        if status['trades_executed'] > 0:
            print(
                f"  • Avg Profit/Trade: ${status['average_profit_per_trade']:.2f}")
            print(
                f"  • Avg Fee/Trade: ${status.get('average_fee_per_trade', 0.0):.2f}")
            print(
                f"  • Risk/Reward Ratio: {status.get('risk_reward_ratio', 0.0):.2f}")

        if status['last_trade']:
            print("\nLast Trade Info:")
            print(f"  • Reason: {status['last_trade']}")
            print(f"  • Data Source: {status['last_trade_data_source']}")
            print(f"  • Signal Time: {status['last_trade_signal_timestamp']}")

        print("\nTechnical Analysis:")
        if status['next_trigger']:
            print(f"  • {status['next_trigger']}")
        if status['current_trends']:
            print("  • Current Trends:")
            for k, v in status['current_trends'].items():
                print(f"    ◦ {k}: {v}")
        if status['ma_difference'] is not None:
            print(f"  • MA Difference: {status['ma_difference']:.4f}")
        if status['ma_slope_difference'] is not None:
            print(
                f"  • MA Slope Difference: {status['ma_slope_difference']:.4f}")
        if 'short_ma_momentum' in status:
            print(f"  • Short MA Momentum: {status['short_ma_momentum']}")
        if 'long_ma_momentum' in status:
            print(f"  • Long MA Momentum: {status['long_ma_momentum']}")
        if 'momentum_alignment' in status:
            print(f"  • Momentum Alignment: {status['momentum_alignment']}")

        prox = status.get('ma_signal_proximity')
        if prox is not None:
            print(f"  • MA Signal Proximity: {prox*100:.2f}%")
            print("    (Closer to 0% => near a crossover)")

        if status['trades_executed'] == 0:
            print("\nNo trades yet, stats are limited.")
        elif status['win_rate'] < 40:
            print("Warning: Win rate is below 40%. Consider reviewing parameters.")
        if status['current_balance'] < status['initial_balance']*0.9:
            print("Warning: Balance is over 10% below initial.")
        if status['remaining_trades_today'] <= 1:
            print("Warning: Approaching daily trade limit!")

        # Show comprehensive adaptive strategy info in long view
        if hasattr(self.auto_trader, 'current_regime'):
            print("\nAdaptive Strategy Details:")
            print("━"*30)
            print(
                f"  • Current Market Regime: {self.auto_trader.current_regime.upper()}")
            print(
                f"  • Regime Confidence: {self.auto_trader.regime_confidence:.1%}")
            print(
                f"  • Active Trading Strategy: {self.auto_trader.active_strategy.upper()}")
            print(
                f"  • Strategy Switches Today: {self.auto_trader.strategy_switches_today}")
            print(
                f"  • Signal Confirmation: {len(getattr(self.auto_trader, 'signal_history', []))}/{getattr(self.auto_trader, 'signal_confirmation_bars', 2)} bars")
            print(
                f"  • Min Trade Gap: {getattr(self.auto_trader, 'min_trade_gap_minutes', 30)} minutes")

            # Show performance by strategy
            print(f"\n  Strategy Performance Breakdown:")
            for strategy_name, perf in self.auto_trader.strategy_performance.items():
                trades = perf.get('trades', 0)
                profit = perf.get('profit', 0.0)
                status_icon = "🔵" if strategy_name == self.auto_trader.active_strategy else "⚪"
                print(
                    f"    {status_icon} {strategy_name.upper()}: {trades} trades, ${profit:.2f} profit")

            # Explain why current strategy was chosen
            if self.auto_trader.current_regime == "ranging":
                print(f"\n  📊 RANGING MARKET DETECTED:")
                print(f"     • High whipsaw ratio detected (MA crossovers failing)")
                print(f"     • Switched to MEAN REVERSION strategy")
                print(f"     • Will buy oversold conditions, sell overbought")
            elif self.auto_trader.current_regime == "trending":
                print(f"\n  📈 TRENDING MARKET DETECTED:")
                print(f"     • Clear directional movement")
                print(f"     • Using MA CROSSOVER strategy")
            elif self.auto_trader.current_regime == "volatile":
                print(f"\n  ⚡ VOLATILE MARKET DETECTED:")
                print(f"     • High volatility with volume spikes")
                print(f"     • Using BREAKOUT strategy")

        session_duration = datetime.now() - self.auto_trader.strategy_start_time
        hours = session_duration.total_seconds() / 3600
        print(f"\nSession Duration: {hours:.1f} hours\n")
        print("━"*50)

    def do_quit(self, arg):
        """
        Quit the program, shutting down threads and processes gracefully.
        """
        print("Quitting...")

        # Try to save diagnostic log on exit
        if self.auto_trader and hasattr(self.auto_trader, 'diagnostic_logger'):
            try:
                self.auto_trader.diagnostic_logger.close()
                print(f"Diagnostic log saved: {self.auto_trader.diagnostic_logger.filename}")
            except Exception as e:
                print(f"Failed to save diagnostic log: {e}")
                # Try crash recovery save
                try:
                    self.auto_trader.diagnostic_logger.crash_recovery()
                    print("Performed crash recovery save")
                except:
                    pass
 
        # Stop auto trader first
        if self.auto_trader and self.auto_trader.running:
            print("Stopping auto trader...")
            self.auto_trader.stop()
            
        # Stop command interface
        if self.command_interface:
            print("Stopping command interface...")
            self.command_interface.stop()
            
        # Stop chart process
        if self.chart_process and self.chart_process.is_alive():
            print("Stopping chart processp...")
            self.stop_dash_app()
            
        # Set stop event for websocket threads
        if self.stop_event:
            print("Signaling websocket threads to stop...")
            self.stop_event.set()
            
        print("Shutdown complete.")
        return True
        
    def _log_full_status_to_diagnostics(self):
        """Log a full status report to diagnostics."""
        if not self.auto_trader or not hasattr(self.auto_trader, 'diagnostic_logger'):
            return
            
        try:
            status = self.auto_trader.get_status()
            
            # Create a comprehensive status dump
            full_status = {
                "timestamp": datetime.now().isoformat(),
                "running": status['running'],
                "position": status['position'],
                "balances": {
                    "btc": status['balance_btc'],
                    "usd": status['balance_usd'],
                    "mtm_usd": status['mark_to_market_usd'],
                    "mtm_btc": status['mark_to_market_btc']
                },
                "position_info": status.get('position_info', {}),
                "performance": {
                    "total_return_pct": status['total_return_pct'],
                    "total_pnl": status['total_profit_loss'],
                    "trades_executed": status['trades_executed'],
                    "win_rate": status['win_rate']
                },
                "trading": {
                    "trades_today": status['trade_count_today'],
                    "remaining_trades": status['remaining_trades_today']
                }
            }
            
            self.auto_trader.diagnostic_logger.log_event("STATUS_REPORT", full_status)
            
        except Exception as e:
            self.logger.error(f"Failed to log status: {e}")

    def stop_dash_app(self):
        """
        If a Dash app is running in a separate process, attempt to shut it down.
        Uses the stored chart_port to connect to the correct instance.
        """
        if self.chart_process and self.chart_process.is_alive():
            try:
                # Use the stored port for shutdown, default to 8050 if not set
                port = self.chart_port or 8050
                print(f"Attempting to shut down Dash app on port {port}...")
                try:
                    response = requests.get(f'http://127.0.0.1:{port}/shutdown', timeout=2)
                    print(f"Shutdown request sent, response: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"Shutdown request failed: {e}")
                
                # Give it a moment to shut down gracefully
                self.chart_process.join(timeout=3)
                
                # If still alive, terminate it
                if self.chart_process.is_alive():
                    print("Dash app didn't shut down gracefully, terminating...")
                    self.chart_process.terminate()
                    self.chart_process.join(timeout=2)

                print("Dash app shut down.")
            except Exception as e:
                print("Failed to shut down Dash app:", e)
                # Force terminate if all else fails
                if self.chart_process.is_alive():
                    self.chart_process.terminate()

    def do_exit(self, arg):
        """
        Alias for 'quit'.
        """
        return self.do_quit(arg)

    def do_chart(self, arg):
        """
        Show a Dash-based chart with strategy comparison support.

        Usage: chart [symbol] [bar_size] [port] [alt_strategy_file]

        Examples:
          chart btcusd 1H
          chart btcusd 1H 8051
          chart btcusd 1H 8051 alt_strategy-1.json
        """
        args = arg.split()
        symbol = 'btcusd'
        bar_size = '1H'
        port = 8050
        alt_strategy_file = None

        # Parse arguments
        if len(args) >= 1:
            symbol = args[0].strip().lower()
        if len(args) >= 2:
            bar_size = args[1].strip()
        if len(args) >= 3:
            try:
                port = int(args[2])
            except ValueError:
                print(f"Invalid port '{args[2]}', using default 8050")
                port = 8050
        if len(args) >= 4:
            alt_strategy_file = args[3].strip()

        # Validate symbol
        if symbol not in self.data_manager.data:
            print(f"No data for symbol '{symbol}'.")
            return

        # Check for required dependencies
        try:
            from tdr_core.charting import run_dash_app, DASH_AVAILABLE
            if not DASH_AVAILABLE:
                print("Install dash & plotly first (pip install dash plotly).")
                return
        except ImportError:
            print(
                "Charting module not found. Please ensure tdr_core/charting.py is present.")
            return

        # Determine strategy parameters
        short_window = 12
        long_window = 36
        strategy_name = 'MA Strategy'

        # Get parameters from auto_trader if available
        if self.auto_trader and isinstance(self.auto_trader, MACrossoverStrategy):
            short_window = self.auto_trader.short_window
            long_window = self.auto_trader.long_window
        else:
            # Try to read from best_strategy.json
            try:
                with open('best_strategy.json', 'r') as f:
                    best_params = json.load(f)
                if best_params.get('Strategy') == 'MA':
                    short_window = int(best_params['Short_Window'])
                    long_window = int(best_params['Long_Window'])
                    strategy_name = f"MA({short_window}, {long_window})"
            except:
                print("Could not read 'best_strategy.json' for windows. Using defaults.")

        # Update shared data dictionary for the chart
        self.data_manager_dict[symbol] = self.data_manager.get_price_dataframe(
            symbol).to_dict('list')

        # Start background thread to keep shared data updated
        def update_shared_data():
            while not self.stop_event.is_set():
                self.data_manager_dict[symbol] = self.data_manager.get_price_dataframe(
                    symbol).to_dict('list')
                time.sleep(60)

        threading.Thread(target=update_shared_data, daemon=True).start()

        # Store the port for shutdown purposes
        self.chart_port = port

        # Start the chart process
        self.chart_process = Process(
            target=run_dash_app,
            args=(
                self.data_manager_dict,
                symbol,
                bar_size,
                short_window,
                long_window
            ),
            kwargs={
                'host': '0.0.0.0',
                'port': port,
                'strategy_file': 'best_strategy.json',
                'strategy_name': strategy_name,
                'alt_strategy_file': alt_strategy_file
            }
        )
        self.chart_process.start()

        # Print access information
        print(f"Dash app is running at http://127.0.0.1:{port}/")
        print(f"Symbol: {symbol.upper()}, Bar Size: {bar_size}")
        print(f"Strategy: {strategy_name}")
        if alt_strategy_file:
            print(f"Alternate Strategy File: {alt_strategy_file}")
        print("Use the dropdown in the web interface to compare strategies.")
        print(
            f"To stop the chart, use 'quit' or shut down via http://127.0.0.1:{port}/shutdown")

        time.sleep(1)

    def do_strategy_diagnostics(self, arg):
        """
        Show detailed diagnostics of the current adaptive strategy state.
        Usage: strategy_diagnostics [detailed]

        This exposes what your adaptive strategy is currently thinking and why
        it hasn't triggered a position change yet.
        """
        if not self.auto_trader or not isinstance(self.auto_trader, AdaptiveMultiStrategy):
            print("No adaptive auto trader running.")
            return

        detailed = arg.strip().lower() == 'detailed'

        try:
            # Get current data
            status = self.auto_trader.get_status()
            df = self.data_manager.get_price_dataframe('btcusd')

            if df.empty:
                print("No data available")
                return

            df = ensure_datetime_index(df)
            df_resampled = df.resample('1H').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                'volume': 'sum', 'trades': 'sum', 'timestamp': 'last', 'source': 'last'
            }).dropna()

            current_price = df_resampled.iloc[-1]['close']

            # Get regime analysis (using the strategy's own logic)
            regime, confidence, metrics = self.auto_trader.detect_market_regime(
                df_resampled)

            # Get signals from each strategy type
            trending_signal, trending_reason = self.auto_trader.generate_trending_signal(
                df_resampled)
            ranging_signal, ranging_reason = self.auto_trader.generate_ranging_signal(
                df_resampled)
            volatile_signal, volatile_reason = self.auto_trader.generate_volatile_signal(
                df_resampled)

            # Calculate position metrics
            status = self.auto_trader.get_status()
            position_info = status.get('position_info', {})
            current_position_value = abs(
                self.auto_trader.balance_btc * current_price) + self.auto_trader.balance_usd

            # Check various thresholds and constraints
            has_significant_position = current_position_value > 50000
            confidence_threshold = 0.75 if has_significant_position else 0.65

            if regime == "trending":
                required_confidence = 0.80
            elif regime == "ranging":
                required_confidence = 0.60
            else:  # volatile
                required_confidence = 0.70

            can_switch = confidence >= required_confidence
            can_trade_gap = self.auto_trader.check_trade_gap()
            signal_confirmed = len(
                self.auto_trader.signal_history) >= self.auto_trader.signal_confirmation_bars

            diagnostics = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "current_price": float(current_price),
                "position_analysis": {
                    "current_position": "SHORT" if status['position'] == -1 else "LONG" if status['position'] == 1 else "ERROR",
                    "entry_price": float(position_info.get('entry_price', 0)),
                    "unrealized_pnl": float(position_info.get('unrealized_pnl', 0)),
                    "position_value": float(current_position_value),
                    "has_significant_position": bool(has_significant_position)
                },
                "regime_detection": {
                    "detected_regime": regime,
                    "confidence": float(confidence),
                    "active_strategy": self.auto_trader.active_strategy,
                    "whipsaw_ratio": float(metrics.get('whipsaw_ratio', 0)),
                    "trend_strength": float(metrics.get('trend_strength', 0)),
                    "required_confidence": float(required_confidence),
                    "can_switch_strategy": bool(can_switch),
                    "confidence_gap": float(confidence - required_confidence)
                },
                "strategy_signals": {
                    "trending": {
                        "signal": int(trending_signal),
                        "reason": str(trending_reason),
                        "active": bool(self.auto_trader.active_strategy == "trending")
                    },
                    "ranging": {
                        "signal": int(ranging_signal),
                        "reason": str(ranging_reason),
                        "active": bool(self.auto_trader.active_strategy == "ranging")
                    },
                    "volatile": {
                        "signal": int(volatile_signal),
                        "reason": str(volatile_reason),
                        "active": bool(self.auto_trader.active_strategy == "volatile")
                    }
                },
                "trading_constraints": {
                    "can_trade_gap": bool(can_trade_gap),
                    "signal_confirmed": bool(signal_confirmed),
                    "signal_history_length": int(len(self.auto_trader.signal_history)),
                    "required_confirmation_bars": int(self.auto_trader.signal_confirmation_bars),
                    "daily_trades_used": int(self.auto_trader.trade_count_today),
                    "daily_trades_remaining": int(max(0, self.auto_trader.max_trades_per_day - self.auto_trader.trade_count_today))
                },
                "why_no_trade": []
            }

            # Analyze why no trade has been triggered
            active_signal = None
            if self.auto_trader.active_strategy == "trending":
                active_signal = trending_signal
            elif self.auto_trader.active_strategy == "ranging":
                active_signal = ranging_signal
            elif self.auto_trader.active_strategy == "volatile":
                active_signal = volatile_signal

            # Check each constraint
            if active_signal == 0:
                diagnostics["why_no_trade"].append(
                    f"No signal from active {self.auto_trader.active_strategy} strategy")
            elif active_signal == status['position']:
                diagnostics["why_no_trade"].append(
                    f"Signal ({active_signal}) matches current position ({status['position']})")

            if not signal_confirmed:
                diagnostics["why_no_trade"].append(
                    f"Signal not confirmed: {len(self.auto_trader.signal_history)}/{self.auto_trader.signal_confirmation_bars} bars")

            if not can_trade_gap:
                mins_since = (datetime.now() - self.auto_trader.last_trade_time).total_seconds(
                ) / 60 if self.auto_trader.last_trade_time else 999
                diagnostics["why_no_trade"].append(
                    f"Trade gap: {mins_since:.1f}min < {self.auto_trader.min_trade_gap_minutes}min required")

            if diagnostics["trading_constraints"]["daily_trades_remaining"] <= 0:
                diagnostics["why_no_trade"].append("Daily trade limit reached")

            # Output results
            print("\n" + "="*80)
            print("ADAPTIVE STRATEGY DIAGNOSTICS")
            print("="*80)

            if detailed:
                print(json.dumps(diagnostics, indent=2))
            else:
                self._print_diagnostics_summary(diagnostics)

            print("="*80)

        except Exception as e:
            self.logger.error(f"Error in strategy diagnostics: {e}")
            # Don't print error details to avoid confusion

    def do_tune_strategy(self, arg):
        """
        Adjust adaptive strategy parameters in real-time.
        Usage: tune_strategy <parameter> <value>

        Available parameters:
        - regime_threshold <0.1-0.9>  : Confidence required to switch regimes
        - confirmation_bars <1-5>     : Bars required to confirm signal  
        - trade_gap_minutes <5-60>    : Minutes between trades
        - rsi_oversold <20-35>        : RSI oversold threshold
        - rsi_overbought <65-80>      : RSI overbought threshold
        """
        if not self.auto_trader or not isinstance(self.auto_trader, AdaptiveMultiStrategy):
            print("No adaptive auto trader running.")
            return

        args = arg.split()
        if len(args) != 2:
            print("Usage: tune_strategy <parameter> <value>")
            print("Example: tune_strategy regime_threshold 0.5")
            return

        param, value_str = args

        try:
            value = float(value_str)

            if param == "regime_threshold":
                if 0.1 <= value <= 0.9:
                    self.auto_trader.regime_switch_threshold = value
                    print(f"✅ Updated regime switch threshold to {value:.1%}")
                else:
                    print("❌ Regime threshold must be between 0.1 and 0.9")

            elif param == "confirmation_bars":
                value = int(value)
                if 1 <= value <= 5:
                    self.auto_trader.signal_confirmation_bars = value
                    print(f"✅ Updated signal confirmation bars to {value}")
                else:
                    print("❌ Confirmation bars must be between 1 and 5")

            elif param == "trade_gap_minutes":
                value = int(value)
                if 5 <= value <= 60:
                    self.auto_trader.min_trade_gap_minutes = value
                    print(f"✅ Updated minimum trade gap to {value} minutes")
                else:
                    print("❌ Trade gap must be between 5 and 60 minutes")

            elif param == "rsi_oversold":
                if 20 <= value <= 35:
                    self.auto_trader.rsi_oversold = value
                    print(f"✅ Updated RSI oversold threshold to {value}")
                else:
                    print("❌ RSI oversold must be between 20 and 35")

            elif param == "rsi_overbought":
                if 65 <= value <= 80:
                    self.auto_trader.rsi_overbought = value
                    print(f"✅ Updated RSI overbought threshold to {value}")
                else:
                    print("❌ RSI overbought must be between 65 and 80")

            else:
                print(f"❌ Unknown parameter: {param}")
                print(
                    "Available: regime_threshold, confirmation_bars, trade_gap_minutes, rsi_oversold, rsi_overbought")

        except ValueError:
            print(f"❌ Invalid value: {value_str}")

    def do_force_regime(self, arg):
        """
        Temporarily override regime detection for testing.
        Usage: force_regime <trending|ranging|volatile|auto>

        'auto' returns to automatic regime detection.
        """
        if not self.auto_trader or not isinstance(self.auto_trader, AdaptiveMultiStrategy):
            print("No adaptive auto trader running.")
            return

        regime = arg.strip().lower()

        if regime in ['trending', 'ranging', 'volatile']:
            # Add override mechanism to the strategy
            if not hasattr(self.auto_trader, 'regime_override'):
                self.auto_trader.regime_override = None

            self.auto_trader.regime_override = regime
            self.auto_trader.active_strategy = regime
            print(f"🔧 FORCING regime to {regime.upper()}")
            print(
                f"⚠️  Strategy will use {regime} logic until you run 'force_regime auto'")

        elif regime == 'auto':
            if hasattr(self.auto_trader, 'regime_override'):
                self.auto_trader.regime_override = None
            print("🔄 Returned to automatic regime detection")

        else:
            print("Usage: force_regime <trending|ranging|volatile|auto>")

    def do_force_signal_log(self, arg):
        """
        Force the next signal evaluation to be logged regardless of deduplication.
        Usage: force_signal_log
        
        This resets the signal deduplication counter to ensure the next evaluation is logged.
        """
        if not self.auto_trader or not hasattr(self.auto_trader, 'diagnostic_logger'):
            print("No auto trader with diagnostic logging running.")
            return
            
        # Reset the deduplication state
        self.auto_trader.diagnostic_logger.last_signal_eval = None
        self.auto_trader.diagnostic_logger.signal_eval_count = 0
        print("✅ Signal evaluation deduplication reset")
        print("   Next signal evaluation will be logged to diagnostics")
        print("   Check diagnostics in ~1 minute with: show_diagnostics SIGNAL_EVAL 1")
    
    def do_reset_position(self, arg):
        """
        Reset position tracking to fix calculation errors.
        Usage: reset_position [entry_price] [btc_amount]

        If no entry_price provided, uses current price.
        If no btc_amount provided, calculates from current balances.
        """
        if not self.auto_trader or not self.auto_trader.running:
            print("No auto trader running.")
            return
        
        current_price = self.data_manager.get_current_price('btcusd')
        if not current_price:
            print("Cannot get current price.")
            return
        
        args = arg.split()
        entry_price = current_price
        btc_amount = None
        
        if len(args) >= 1:
             try:
                entry_price = float(args[0])
             except ValueError:
                print("Invalid price. Usage: reset_position [entry_price] [btc_amount]")
                return
                
        if len(args) >= 2:
            try:
                btc_amount = float(args[1])
            except ValueError:
                print("Invalid BTC amount. Usage: reset_position [entry_price] [btc_amount]")
                return

        # Reset position tracking
        if self.auto_trader.position == 1:
            # Long position
            self.auto_trader.position_size = self.auto_trader.balance_btc
            self.auto_trader.position_cost_basis = self.auto_trader.position_size * entry_price
            print(f"Reset LONG position: {self.auto_trader.position_size:.8f} BTC @ ${entry_price:.2f}")
            print(f"New cost basis: ${self.auto_trader.position_cost_basis:.2f}")
        elif self.auto_trader.position == -1:
            # For short positions
            if btc_amount is None:
                # Calculate BTC amount from USD balance and entry price
                btc_amount = self.auto_trader.balance_usd / entry_price
                
            # Set position_size as negative for shorts

        if self.auto_trader.position == -1:
            # For short positions
            if btc_amount is None:
                # Calculate BTC amount from USD balance and entry price
                btc_amount = self.auto_trader.balance_usd / entry_price

            self.auto_trader.position_size = -btc_amount
            self.auto_trader.position_cost_basis = btc_amount * entry_price
            self.auto_trader.last_trade_price = entry_price  # Also set last trade price
            
            print(f"Reset SHORT position: {btc_amount:.8f} BTC equivalent @ ${entry_price:.2f}")
            print(f"Position size: {self.auto_trader.position_size:.8f} (negative = short)")
            print(f"Cost basis: ${self.auto_trader.position_cost_basis:.2f}")
            
            # Calculate and show what P&L should be
            expected_pnl = (entry_price - current_price) * btc_amount
            print(f"Expected P&L at current price ${current_price:.2f}: ${expected_pnl:.2f}")

            btc_equivalent = btc_amount  # Fix undefined variable
            # Log the reset
            if hasattr(self.auto_trader, 'diagnostic_logger'):
                self.auto_trader.diagnostic_logger.log_event("POSITION_RESET", {
                    "type": "manual_reset",
                    "position": "SHORT",
                    "entry_price": entry_price,
                    "btc_amount": btc_amount,
                    "current_price": current_price,
                    "expected_pnl": expected_pnl
                })
        else:
            print("No position to reset.")


    def do_show_diagnostics(self, arg):
        """
        Show recent diagnostic events from the current session.
        Usage: show_diagnostics [event_type] [count]
        
        Event types: ALL, SIGNAL_EVAL, TRADE, REGIME_CHANGE, ERROR, POSITION_ANOMALY, SNAPSHOT
        
        Examples:
          show_diagnostics                    # Show last 10 events
          show_diagnostics SIGNAL_EVAL 20    # Show last 20 signal evaluations
          show_diagnostics ERROR             # Show all errors
        """
        if not self.auto_trader or not hasattr(self.auto_trader, 'diagnostic_logger'):
            print("No diagnostic logger available.")
            return
            
        args = arg.split()
        event_type = args[0].upper() if args else "ALL"
        count = int(args[1]) if len(args) > 1 else 10
        
        try:
            # Read the diagnostic file
            with open(self.auto_trader.diagnostic_logger.filename, 'r') as f:
                data = json.load(f)
                
            print(f"\nDiagnostic Log: {self.auto_trader.diagnostic_logger.filename}")
            print(f"Session Start: {data['session_start']}")
            print(f"Total Events: {data['total_events']}")
            print("\nEvent Type Summary:")
            for etype, ecount in data['event_types'].items():
                print(f"  {etype}: {ecount}")
                
            print(f"\nShowing last {count} {event_type} events:")
            print("="*80)
            
            events = data['events']
            if event_type != "ALL":
                events = [e for e in events if e['type'] == event_type]
                
            for event in events[-count:]:
                print(f"\n[{event['timestamp']}] {event['type']}")
                if event['type'] == 'SIGNAL_EVAL':
                    d = event['data']
                    print(f"  Signal: {d['signal_type']} = {d['signal_value']}")
                    print(f"  Reason: {d['reason']}")
                    print(f"  Will Trade: {'YES' if d['will_trade'] else 'NO'}")
                    if d.get('why_not'):
                        print(f"  Why Not: {', '.join(d['why_not'])}")
                elif event['type'] == 'TRADE':
                    d = event['data']
                    print(f"  Type: {d['trade_type']} @ ${d['price']:.2f}")
                    print(f"  Amount: {d['amount']:.8f} BTC")
                    print(f"  P&L: ${d['pnl']:.2f}")
                elif event['type'] == 'REGIME_CHANGE':
                    d = event['data']
                    print(f"  Change: {d['old_regime']} → {d['new_regime']}")
                    print(f"  Confidence: {d['confidence']:.1%}")
                elif event['type'] == 'ERROR':
                    print(f"  Error: {event['data']['error']}")
                elif event['type'] == 'POSITION_ANOMALY':
                    d = event['data']
                    print(f"  Anomaly: {d['description']}")
                    print(f"  Details: {json.dumps(d['details'], indent=4)}")
                elif event['type'] == 'SNAPSHOT':
                    d = event['data']
                    print(f"  Price: ${d['market']['current_price']:.2f}")
                    print(f"  Position: {d['position']['direction']}")
                    print(f"  MTM: ${d['position']['mtm_usd']:.2f}")
                    print(f"  Unrealized P&L: ${d['position']['unrealized_pnl']:.2f}")
                    
        except FileNotFoundError:
            print(f"Diagnostic file not found: {self.auto_trader.diagnostic_logger.filename}")
        except Exception as e:
            print(f"Error reading diagnostics: {e}")
            
    def do_diagnostics_file(self, arg):
        """
        Show the path to the current diagnostics file for sharing.
        """
        if self.auto_trader and hasattr(self.auto_trader, 'diagnostic_logger'):
            print(f"Current diagnostics file: {os.path.abspath(self.auto_trader.diagnostic_logger.filename)}")
            print("You can share this file to show what's been happening.")
        else:
            print("No diagnostic logger available.")

    def do_summary_diagnostics(self, arg):
        """
        Export a condensed summary of diagnostics suitable for sharing.
        Usage: summary_diagnostics [filename]

        If no filename provided, prints to console.
        """
        if not self.auto_trader or not hasattr(self.auto_trader, 'diagnostic_logger'):
            print("No diagnostic logger available.")
            return

        try:
            summary = self.auto_trader.diagnostic_logger.export_summary()

            filename = arg.strip() if arg.strip() else None

            if filename:
                # Save to file
                with open(filename, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"Summary exported to: {os.path.abspath(filename)}")
                print(f"File size: {os.path.getsize(filename) / 1024:.1f} KB")
            else:
                # Print to console
                print(json.dumps(summary, indent=2))
        except Exception as e:
            print(f"Error creating summary: {e}")

    def do_list_diagnostics(self, arg):
        """
        List all diagnostic files in the current directory.
        Usage: list_diagnostics [days]
        
        Examples:
          list_diagnostics      # Show all diagnostic files
          list_diagnostics 7    # Show files from last 7 days
        """
        days = int(arg) if arg else None
        
        # Find all diagnostic files
        files = glob.glob("diagnostics_*.json")
        
        if not files:
            print("No diagnostic files found.")
            return
            
        # Get file info
        file_info = []
        for f in files:
            try:
                stat = os.stat(f)
                mtime = datetime.fromtimestamp(stat.st_mtime)
                
                # Filter by days if specified
                if days:
                    age = (datetime.now() - mtime).days
                    if age > days:
                        continue
                        
                # Try to read summary info
                try:
                    with open(f, 'r') as file:
                        data = json.load(file)
                        events = data.get('total_events', '?')
                        session_start = data.get('session_start', '?')
                except:
                    events = '?'
                    session_start = '?'
                    
                file_info.append({
                    'name': f,
                    'size_kb': stat.st_size / 1024,
                    'modified': mtime,
                    'events': events,
                    'session': session_start
                })
            except:
                continue
                
        # Sort by modified time, newest first
        file_info.sort(key=lambda x: x['modified'], reverse=True)
        
        print(f"\nDiagnostic Files ({len(file_info)} found):")
        print("="*80)
        print(f"{'Filename':<50} {'Size':<10} {'Events':<10} {'Session Start'}")
        print("-"*80)
        
        for info in file_info:
            print(f"{info['name']:<50} {info['size_kb']:<10.1f} {str(info['events']):<10} {info['session']}")
            
        if self.auto_trader and hasattr(self.auto_trader, 'diagnostic_logger'):
            print(f"\n* Current session: {self.auto_trader.diagnostic_logger.filename}")

    def _print_diagnostics_summary(self, diagnostics):
        """Print a concise summary of strategy diagnostics."""
        status = self.auto_trader.get_status()

        # FIX: Ensure pos_str is properly defined
        pos_str = {1: 'LONG', -1: 'SHORT'}.get(status['position'], 'UNKNOWN')
        pos = diagnostics["position_analysis"]
        regime = diagnostics["regime_detection"]
        signals = diagnostics["strategy_signals"]
        constraints = diagnostics["trading_constraints"]

        print(f"\n🎯 CURRENT POSITION:")
        print(f"   Direction: {pos['current_position']}")
        print(
            f"   Entry: ${pos['entry_price']:.2f} → Current: ${diagnostics['current_price']:.2f}")
        print(f"   P&L: ${pos['unrealized_pnl']:.2f}")

        print(f"\n📊 REGIME DETECTION:")
        print(f"   Detected: {regime['detected_regime'].upper()}")
        print(
            f"   Confidence: {regime['confidence']:.1%} (need {regime['required_confidence']:.1%})")
        print(f"   Active Strategy: {regime['active_strategy'].upper()}")
        print(
            f"   Can Switch: {'✅ YES' if regime['can_switch_strategy'] else '❌ NO'}")
        if not regime['can_switch_strategy']:
            print(
                f"   Gap: {regime['confidence_gap']:.1%} short of required confidence")

        print(f"\n🎪 STRATEGY SIGNALS:")
        for strategy_name, signal_info in signals.items():
            status_icon = "🔵" if signal_info['active'] else "⚪"
            signal_text = "LONG" if signal_info['signal'] == 1 else "SHORT" if signal_info['signal'] == -1 else "NEUTRAL"
            print(f"   {status_icon} {strategy_name.upper()}: {signal_text}")
            if signal_info['active']:
                print(f"      → {signal_info['reason']}")

        print(f"\n⚙️  TRADING CONSTRAINTS:")
        print(
            f"   Signal Confirmed: {'✅' if constraints['signal_confirmed'] else '❌'} ({constraints['signal_history_length']}/{constraints['required_confirmation_bars']} bars)")
        print(
            f"   Trade Gap OK: {'✅' if constraints['can_trade_gap'] else '❌'}")
        print(
            f"   Daily Trades: {constraints['daily_trades_used']}/{constraints['daily_trades_used'] + constraints['daily_trades_remaining']}")

        if diagnostics["why_no_trade"]:
            print(f"\n🚫 WHY NO TRADE YET:")
            for reason in diagnostics["why_no_trade"]:
                print(f"   • {reason}")
        else:
            print(f"\n✅ All constraints satisfied - trade should trigger on next signal!")


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

        # Redefine pos_str here since it's used later in the method  
        status = self.auto_trader.get_status()
        pos_str = {1: 'LONG', -1: 'SHORT'}.get(status['position'], 'UNKNOWN')

        print(f"  • Direction:  {pos_str}")
        print(f"  • Current Price:  ${pos_info.get('current_price', 0.0):.2f}")
        print(f"  • Entry Price:    ${pos_info.get('entry_price', 0.0):.2f}")
        if status['position'] == 1:
            print(
                f"  • Position Size (BTC): {pos_info.get('position_size_btc', 0.0):.8f}")
            print(
                f"  • Position Value (USD): ${pos_info.get('position_size_usd', 0.0):.2f}")
        elif status['position'] == -1:
            print(
                f"  • Short Position (holding USD): ${pos_info.get('position_size_usd', 0.0):.2f}")
            # BUG FIX: Show proper BTC equivalent for short positions
            if pos_info.get('entry_price', 0) > 0:
                btc_equivalent = pos_info.get('position_size_usd', 0.0) / pos_info.get('entry_price', 1)
                print(
                    f"  • BTC Equivalent: {btc_equivalent:.8f} BTC")
        else:
            # This should never happen
            print("  • ERROR: System in undefined state")
        print(
            f"  • Unrealized PnL:  ${pos_info.get('unrealized_pnl', 0.0):.2f}")

        print("\nTrading Statistics:")
        print(f"  • Total Trades: {status['trades_executed']}")
        print(f"  • Profitable Trades: {status['profitable_trades']}")
        print(f"  • Win Rate: {status['win_rate']:.1f}%")

        if status['trades_executed'] > 0:
            print(
                f"  • Avg Profit/Trade: ${status['average_profit_per_trade']:.2f}")
            print(
                f"  • Avg Fee/Trade: ${status.get('average_fee_per_trade', 0.0):.2f}")
            print(
                f"  • Risk/Reward Ratio: {status.get('risk_reward_ratio', 0.0):.2f}")

        if status['last_trade']:
            print("\nLast Trade Info:")
            print(f"  • Reason: {status['last_trade']}")
            print(f"  • Data Source: {status['last_trade_data_source']}")
            print(f"  • Signal Time: {status['last_trade_signal_timestamp']}")

        print("\nTechnical Analysis:")
        if status['next_trigger']:
            print(f"  • {status['next_trigger']}")
        if status['current_trends']:
            print("  • Current Trends:")
            for k, v in status['current_trends'].items():
                print(f"    ◦ {k}: {v}")
        if status['ma_difference'] is not None:
            print(f"  • MA Difference: {status['ma_difference']:.4f}")
        if status['ma_slope_difference'] is not None:
            print(
                f"  • MA Slope Difference: {status['ma_slope_difference']:.4f}")
        if 'short_ma_momentum' in status:
            print(f"  • Short MA Momentum: {status['short_ma_momentum']}")
        if 'long_ma_momentum' in status:
            print(f"  • Long MA Momentum: {status['long_ma_momentum']}")
        if 'momentum_alignment' in status:
            print(f"  • Momentum Alignment: {status['momentum_alignment']}")

        prox = status.get('ma_signal_proximity')
        if prox is not None:
            print(f"  • MA Signal Proximity: {prox*100:.2f}%")
            print("    (Closer to 0% => near a crossover)")

        if status['trades_executed'] == 0:
            print("\nNo trades yet, stats are limited.")
        elif status['win_rate'] < 40:
            print("Warning: Win rate is below 40%. Consider reviewing parameters.")
        if status['current_balance'] < status['initial_balance']*0.9:
            print("Warning: Balance is over 10% below initial.")
        if status['remaining_trades_today'] <= 1:
            print("Warning: Approaching daily trade limit!")

        # Show comprehensive adaptive strategy info in long view
        if hasattr(self.auto_trader, 'current_regime'):
            print("\nAdaptive Strategy Details:")
            print("━"*30)
            print(
                f"  • Current Market Regime: {self.auto_trader.current_regime.upper()}")
            print(
                f"  • Regime Confidence: {self.auto_trader.regime_confidence:.1%}")
            print(
                f"  • Active Trading Strategy: {self.auto_trader.active_strategy.upper()}")
            print(
                f"  • Strategy Switches Today: {self.auto_trader.strategy_switches_today}")
            print(
                f"  • Signal Confirmation: {len(getattr(self.auto_trader, 'signal_history', []))}/{getattr(self.auto_trader, 'signal_confirmation_bars', 2)} bars")
            print(
                f"  • Min Trade Gap: {getattr(self.auto_trader, 'min_trade_gap_minutes', 30)} minutes")

            # Show performance by strategy
            print(f"\n  Strategy Performance Breakdown:")
            for strategy_name, perf in self.auto_trader.strategy_performance.items():
                trades = perf.get('trades', 0)
                profit = perf.get('profit', 0.0)
                status_icon = "🔵" if strategy_name == self.auto_trader.active_strategy else "⚪"
                print(
                    f"    {status_icon} {strategy_name.upper()}: {trades} trades, ${profit:.2f} profit")

            # Explain why current strategy was chosen
            if self.auto_trader.current_regime == "ranging":
                print(f"\n  📊 RANGING MARKET DETECTED:")
                print(f"     • High whipsaw ratio detected (MA crossovers failing)")
                print(f"     • Switched to MEAN REVERSION strategy")
                print(f"     • Will buy oversold conditions, sell overbought")
            elif self.auto_trader.current_regime == "trending":
                print(f"\n  📈 TRENDING MARKET DETECTED:")
                print(f"     • Clear directional movement")
                print(f"     • Using MA CROSSOVER strategy")
            elif self.auto_trader.current_regime == "volatile":
                print(f"\n  ⚡ VOLATILE MARKET DETECTED:")
                print(f"     • High volatility with volume spikes")
                print(f"     • Using BREAKOUT strategy")

        session_duration = datetime.now() - self.auto_trader.strategy_start_time
        hours = session_duration.total_seconds() / 3600
        print(f"\nSession Duration: {hours:.1f} hours\n")
        print("━"*50)
