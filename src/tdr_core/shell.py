###############################################################################
# src/tdr_core/shell.py
###############################################################################
# Full File Path: src/tdr_core/shell.py
#
# CHANGES (EXPLANATION):
#   1) Replace the term "forcibly" with simpler language like "initiate trades" or "flip."
#   2) Always set a theoretical cost basis (entry price, position size) 
#      when no active trade is needed, based on the user’s auto_trade input 
#      (the stance + BTC or USD). 
#   3) Retain all original logic, docstrings, and structure, except minimal 
#      changes in do_auto_trade for RSI and (optionally) MA so that every 
#      scenario sets the cost basis properly if no immediate trade is done.
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
    of DF. If RSI < oversold => position=1, if RSI > overbought => position=-1, 
    else 0.
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
        Convert 'long'|'short'|'neutral' to +1|-1|0 for the user's actual holdings.
        'long' => user actually holds BTC
        'short' => user actually holds USD
        'neutral' => user has neither? (optional)
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

        The user command "auto_trade <amount><btc|usd> <long|short>" 
        means: 
        - If <long>, user is telling the system they currently hold <amount> BTC
        - If <short>, user is telling the system they currently hold <amount> USD
        The system checks the final signal from best_strategy.json. If the system 
        wants the opposite stance, it initiates trades (either 3 partial buys or 
        1 sell). If the system wants the same stance, no immediate trades 
        but we set a theoretical cost basis so status shows entry price, position, etc.
        """
        if self.auto_trader and self.auto_trader.running:
            print("Auto-trading is already running. Stop it first.")
            return

        args_list = arg.split()
        if len(args_list) != 2:
            print("Usage: auto_trade <amount><btc|usd> <long|short|neutral>")
            return

        balance_str = args_list[0].lower()
        user_stance_str = args_list[1].lower()

        user_stance = self.parse_position_str(user_stance_str)
        if user_stance is None:
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

        file_path = os.path.abspath('best_strategy.json')
        if not os.path.exists(file_path):
            print(f"Error: '{file_path}' not found.")
            return

        with open(file_path, 'r') as f:
            best_strategy_params = json.load(f)

        strategy_name = best_strategy_params.get('Strategy')
        do_live = best_strategy_params.get('do_live_trades', False)
        max_trades_day = best_strategy_params.get('max_trades_per_day', 5)

        if strategy_name == 'RSI':
            from tdr_core.shell import determine_rsi_position
            rsi_window = int(best_strategy_params.get('RSI_Window', 14))
            overbought = float(best_strategy_params.get('Overbought', 70))
            oversold   = float(best_strategy_params.get('Oversold', 30))

            df = self.data_manager.get_price_dataframe('btcusd').copy()
            if 'close' not in df.columns and 'price' in df.columns:
                df.rename(columns={'price': 'close'}, inplace=True)

            # The system's stance from historical RSI:
            if df.empty:
                system_stance = 0
            else:
                system_stance = determine_rsi_position(df, rsi_window, overbought, oversold)

            initial_balance_btc = 0.0
            initial_balance_usd = 0.0

            # interpret user stance
            if user_stance == 1 and amount_unit == 'btc':
                initial_balance_btc = amount_num
            elif user_stance == -1 and amount_unit == 'usd':
                initial_balance_usd = amount_num
            elif user_stance == 0:
                print("User stance neutral => no BTC or USD? We'll set them both to 0.")
            else:
                # fallback
                print("Error: mismatch between stance and unit.")
                return

            # create RSI strategy
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
                initial_position=user_stance,
                initial_balance_btc=initial_balance_btc,
                initial_balance_usd=initial_balance_usd
            )

            current_price = self.data_manager.get_current_price('btcusd') or 0.0

            # CHANGED: If system_stance != user_stance, we do partial buy or single sell
            # If system_stance == user_stance, we skip trades and set theoretical cost basis
            if system_stance == 1:  # system wants long
                if user_stance == -1 and initial_balance_usd > 1e-8:
                    # We hold USD => system wants long => do partial buy
                    self.logger.info("RSI says long, user has USD => flipping to long with partial buys.")
                    self.auto_trader.rsi_buy_in_three_parts(
                        current_price,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        datetime.now()  
                    )
                else:
                    # user is already long => no active trades
                    # set theoretical cost basis
                    self.logger.info("RSI says long, user is already long => no trade, set theoretical basis.")
                    if user_stance == 1 and initial_balance_btc > 1e-8 and current_price > 0:
                        if self.auto_trader.position_size < 1e-8:
                            self.auto_trader.position_size = initial_balance_btc
                            self.auto_trader.position_cost_basis = initial_balance_btc * current_price
                            self.auto_trader.theoretical_trade = {
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'direction': 'long',
                                'amount': initial_balance_btc,
                                'theoretical': True
                            }
            elif system_stance == -1:  # system wants short
                if user_stance == 1 and initial_balance_btc > 1e-8:
                    # We hold BTC => system wants short => single sell
                    self.logger.info("RSI says short, user has BTC => flipping to short by selling.")
                    trade_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.auto_trader.execute_trade(
                        "sell",
                        current_price,
                        trade_ts,
                        datetime.now(),
                        initial_balance_btc
                    )
                else:
                    # user is already short => set theoretical basis
                    self.logger.info("RSI says short, user is already short => no trade, set theoretical basis.")
                    if user_stance == -1 and initial_balance_usd > 1e-8 and current_price > 0:
                        short_btc = initial_balance_usd / current_price
                        if self.auto_trader.position_size > -1e-8 and short_btc > 0:
                            self.auto_trader.position_size = - short_btc
                            self.auto_trader.position_cost_basis = short_btc * current_price
                            self.auto_trader.theoretical_trade = {
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'direction': 'short',
                                'amount': initial_balance_usd,
                                'theoretical': True
                            }
            else:
                # system_stance = 0 => RSI says neutral
                self.logger.info("RSI says neutral => no trades, set theoretical basis if user wants to keep it.")
                # up to you if you want to do something special for neutral

            self.auto_trader.start()
            print(f"Auto-trading started with {balance_str}, stance={user_stance_str}, "
                  f"RSI strategy (Window={rsi_window}, Overbought={overbought}, Oversold={oversold}), do_live_trades={do_live}")

        else:
            print(f"Best strategy is not 'RSI'; it's {strategy_name}.")
            print("Currently only showing RSI logic changes for minimal example.")
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
        if 'short_ma_momentum' in status:
            print(f"  • Short MA Momentum: {status['short_ma_momentum']}")
        if 'long_ma_momentum' in status:
            print(f"  • Long MA Momentum: {status['long_ma_momentum']}")
        if 'momentum_alignment' in status:
            print(f"  • Momentum Alignment: {status['momentum_alignment']}")

        # RSI debug info:
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

    def do_stop_auto_trade(self, arg):
        """
        Stop auto-trading if running.
        """
        if self.auto_trader and self.auto_trader.running:
            self.auto_trader.stop()
            print("Auto-trading stopped.")
        else:
            print("No auto-trading is running.")

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
        """
        Alias for 'quit'.
        """
        return self.do_quit(arg)

    def stop_dash_app(self):
        """
        If a Dash app is running in a separate process, attempt to shut it down.
        """
        if self.chart_process and self.chart_process.is_alive():
            try:
                requests.get('http://127.0.0.1:8050/shutdown')
                self.chart_process.join()
                print("Dash app shut down.")
            except Exception as e:
                print("Failed to shut down Dash app:", e)

    def do_chart(self, arg):
        """
        Show a Dash-based chart: chart [symbol] [bar_size].
        E.g., chart btcusd 1H
        """
        args = arg.split()
        symbol = 'btcusd'
        bar_size = '1H'
        if len(args) >= 1:
            symbol = args[0].strip().lower()
        if len(args) >= 2:
            bar_size = args[1].strip()
        if symbol not in self.data_manager.data:
            print(f"No data for symbol '{symbol}'.")
            return

        try:
            import dash
            from dash import dcc, html
            from dash.dependencies import Output, Input
            import plotly.graph_objs as go
            from flask import Flask, request
            from multiprocessing import Process
        except ImportError:
            print("Install dash & plotly first (pip install dash plotly).")
            return

        short_window = 12
        long_window = 36
        if self.auto_trader and isinstance(self.auto_trader, MACrossoverStrategy):
            short_window = self.auto_trader.short_window
            long_window = self.auto_trader.long_window
        else:
            try:
                with open('best_strategy.json','r') as f:
                    best_params = json.load(f)
                if best_params.get('Strategy') == 'MA':
                    short_window = int(best_params['Short_Window'])
                    long_window = int(best_params['Long_Window'])
            except:
                print("Could not read 'best_strategy.json' for windows. Using defaults.")

        self.data_manager_dict[symbol] = self.data_manager.get_price_dataframe(symbol).to_dict('list')

        def update_shared_data():
            while not self.stop_event.is_set():
                self.data_manager_dict[symbol] = self.data_manager.get_price_dataframe(symbol).to_dict('list')
                time.sleep(60)

        threading.Thread(target=update_shared_data, daemon=True).start()

        from tdr import run_dash_app  # minimal local import
        self.chart_process = Process(
            target=run_dash_app,
            args=(self.data_manager_dict, symbol, bar_size, short_window, long_window)
        )
        self.chart_process.start()
        print("Dash app is running at http://127.0.0.1:8050/")
        time.sleep(1)
