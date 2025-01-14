# src/tdr_core/shell.py
# ----------------------------------------------------------------------------
# FULL FILE PATH: src/tdr_core/shell.py
# ----------------------------------------------------------------------------
# CHANGES MADE:
#   1) Added an optional argument to `do_status()` so that:
#        - `status` or `status <anything_not_full>` => shows a short version (default).
#        - `status full` => shows the original long version (the entire block).
#   2) In both versions, we added a "Direction" line to Position Details.
#   3) We have preserved all existing comments, logic, and formatting 
#      except where explicitly needed to accommodate the short/full toggle.

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
from tdr_core.strategies import MACrossoverStrategy
from tdr_core.data_manager import CryptoDataManager
from tdr_core.trade import Trade

# Original references from tdr.py
from utils.analysis import analyze_data, run_trading_system
from data.loader import create_metadata_file, parse_log_file

###############################################################################
def run_dash_app(data_manager_dict, symbol, bar_size, short_window, long_window):
    """
    Dash-based real-time candlestick chart with MA signals.
    """
    import dash
    from dash import dcc, html
    from dash.dependencies import Output, Input
    import plotly.graph_objs as go
    import threading
    import pandas as pd
    import numpy as np
    from flask import Flask, request

    server = Flask(__name__)
    app = dash.Dash(__name__, server=server)

    @server.route('/shutdown')
    def shutdown():
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            return 'Not running with the Werkzeug Server'
        func()
        return 'Server shutting down...'

    app.layout = html.Div(children=[
        html.H1(children='{} Real-time Candlestick Chart'.format(symbol.upper())),
        dcc.Graph(id='live-graph', style={'width': '100%', 'height': '80vh'}),
        dcc.Interval(id='graph-update', interval=60*1000, n_intervals=0)
    ])

    @app.callback(
        Output('live-graph', 'figure'),
        [Input('graph-update', 'n_intervals'),
         Input('live-graph', 'relayoutData')]
    )
    def update_graph_live(n, relayout_data):
        df = pd.DataFrame.from_dict(data_manager_dict[symbol])
        if df.empty:
            return {}

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)

        try:
            df_resampled = df.resample(bar_size).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'trades': 'sum',
                'timestamp': 'last',
                'source': 'last'
            }).dropna()
        except ValueError:
            return {}

        if len(df_resampled) < long_window:
            return {}

        from indicators.technical_indicators import add_moving_averages, generate_ma_signals
        df_ma = add_moving_averages(df_resampled.copy(), short_window, long_window, price_col='close')
        df_ma = generate_ma_signals(df_ma)
        df_ma['Signal_Change'] = df_ma['MA_Signal'].diff()
        df_ma['Buy_Signal_Price'] = np.where(df_ma['Signal_Change'] == 2, df_ma['close'], np.nan)
        df_ma['Sell_Signal_Price'] = np.where(df_ma['Signal_Change'] == -2, df_ma['close'], np.nan)

        if relayout_data and 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            x_start = pd.to_datetime(relayout_data['xaxis.range[0]'])
            x_end = pd.to_datetime(relayout_data['xaxis.range[1]'])
        else:
            x_end = df_ma.index.max()
            x_start = x_end - pd.Timedelta(days=7)

        df_visible = df_ma[(df_ma.index >= x_start) & (df_ma.index <= x_end)]
        if df_visible.empty:
            df_visible = df_ma

        y_min = df_visible[['low','Short_MA','Long_MA','Buy_Signal_Price','Sell_Signal_Price']].min().min()
        y_max = df_visible[['high','Short_MA','Long_MA','Buy_Signal_Price','Sell_Signal_Price']].max().max()
        y_padding = (y_max - y_min) * 0.05
        y_min -= y_padding
        y_max += y_padding

        candlestick = go.Candlestick(
            x=df_visible.index,
            open=df_visible['open'],
            high=df_visible['high'],
            low=df_visible['low'],
            close=df_visible['close'],
            name='Candlestick'
        )
        short_ma_line = go.Scatter(
            x=df_visible.index,
            y=df_visible['Short_MA'],
            line=dict(color='blue', width=1),
            name=f'Short MA ({short_window})'
        )
        long_ma_line = go.Scatter(
            x=df_visible.index,
            y=df_visible['Long_MA'],
            line=dict(color='red', width=1),
            name=f'Long MA ({long_window})'
        )
        buy_signals = go.Scatter(
            x=df_visible.index,
            y=df_visible['Buy_Signal_Price'],
            mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=12),
            name='Buy Signal'
        )
        sell_signals = go.Scatter(
            x=df_visible.index,
            y=df_visible['Sell_Signal_Price'],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=12),
            name='Sell Signal'
        )

        data = [candlestick, short_ma_line, long_ma_line, buy_signals, sell_signals]
        layout = go.Layout(
            xaxis=dict(title='Time', range=[x_start, x_end]),
            yaxis=dict(title='Price ($)', range=[y_min, y_max]),
            title='{} Candlestick Chart with MAs'.format(symbol.upper()),
            height=800
        )

        return {'data': data, 'layout': layout}

    app.run_server(debug=False, use_reloader=False)


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
            'status': 'status [full]',
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
        if strategy_name != 'MA':
            print(f"Best strategy is not 'MA'; it's {strategy_name}.")
            return

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

        # If the user starts "short" and hist_position is also short => theoretical
        if desired_position == -1 and hist_position == -1 and current_market_price > 0:
            short_btc = amount_num / current_market_price
            if self.auto_trader.position_size > -1e-8 and short_btc > 0:
                self.auto_trader.position_size = - short_btc
                self.auto_trader.position_cost_basis = short_btc * current_market_price
                self.logger.info(
                    f"(auto_trade) Setting cost basis to {self.auto_trader.position_cost_basis:.2f} "
                    f"for an initial SHORT of {short_btc:.6f} BTC (=-{short_btc:.6f}) at ${current_market_price:.2f}."
                )
                self.auto_trader.theoretical_trade = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'direction': 'short',
                    'amount': amount_num,
                    'theoretical': True
                }

        # If user request != hist_position => do immediate real trade
        if desired_position == 1 and hist_position != 1 and current_market_price > 0:
            buy_btc = amount_num / current_market_price
            trade_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.logger.info(f"(auto_trade) Forcing immediate BUY for {buy_btc:.6f} BTC at ${current_market_price:.2f}.")
            self.auto_trader.execute_trade(
                "buy",
                current_market_price,
                trade_ts,
                datetime.now(),  # signal_timestamp
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
        Show status of auto-trading. Usage: status [full]
        
        By default (status, no argument), we show only a short version:
          - Position Details with direction 
        If user types "status full", we show the entire block.
        """
        sub_arg = arg.strip().lower()
        show_full = (sub_arg == 'full')

        if not self.auto_trader or not self.auto_trader.running:
            print("Auto-trading is not running.")
            return

        status = self.auto_trader.get_status()
        pos_str = {1:'Long', -1:'Short', 0:'Neutral'}.get(status['position'], 'Unknown')

        # If user asked for the short version (default):
        if not show_full:
            print("\nPosition Details (Short View):")
            print("━"*50)
            # This line is newly added: "Direction"
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

            print(f"  • Unrealized PnL:  ${pos_info.get('unrealized_pnl', 0.0):.2f}\n")
            return

        # Otherwise, show the full (original) status:
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
        # Additional line showing direction
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
        if status['current_trends']:
            print("  • Current Trends:")
            for k, v in status['current_trends'].items():
                print(f"    ◦ {k}: {v}")
        if status['ma_difference'] is not None:
            print(f"  • MA Difference: {status['ma_difference']:.4f}")
        if status['ma_slope_difference'] is not None:
            print(f"  • MA Slope Difference: {status['ma_slope_difference']:.4f}")
        if 'short_ma_momentum' in status:
            print(f"  • Short MA Momentum: {status['short_ma_momentum']}")
        if 'long_ma_momentum' in status:
            print(f"  • Long MA Momentum: {status['long_ma_momentum']}")
        if 'momentum_alignment' in status:
            print(f"  • Momentum Alignment: {status['momentum_alignment']}")

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

    def do_exit(self, arg):
        """
        Alias for 'quit'.
        """
        return self.do_quit(arg)

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

        self.chart_process = Process(
            target=run_dash_app,
            args=(self.data_manager_dict, symbol, bar_size, short_window, long_window)
        )
        self.chart_process.start()
        print("Dash app is running at http://127.0.0.1:8050/")
        time.sleep(1)
