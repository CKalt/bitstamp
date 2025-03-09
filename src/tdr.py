###############################################################################
# src/tdr.py
###############################################################################
# Full File Path: src/tdr.py
#
# CHANGES (for REST server & new commands):
#   1) We add a Flask-based REST interface in a new code block at the bottom.
#   2) We define start_rest_server() and stop_rest_server() controlling
#      a background thread that runs the Flask app.
#   3) We keep ALL original code and comments intact and unremoved.
#   4) We also expose endpoints for candle data and indicator data (e.g. MAs).
###############################################################################

#!/usr/bin/env python
# src/tdr.py

import sys
import os
import pandas as pd
import numpy as np
import asyncio
import websockets
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from multiprocessing import Process, Manager, set_start_method

# For the Plotly and Dash implementation
try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Output, Input
    import plotly.graph_objs as go
except ImportError:
    pass  # We will handle the ImportError in the do_chart method

# Adjust sys.path to import modules from 'src' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

# Import parse_log_file from data.loader
from data.loader import parse_log_file

# Import indicators/technical_indicators
from indicators.technical_indicators import (
    ensure_datetime_index,
    add_moving_averages,
    generate_ma_signals,
    calculate_rsi,
    generate_rsi_signals,
    calculate_bollinger_bands,
    generate_bollinger_band_signals,
    calculate_macd,
    generate_macd_signals,
)

# ------------------------------------------------------------------------
# NEW IMPORTS for refactored modules (preserving original classes/functions)
# ------------------------------------------------------------------------
from tdr_core.data_manager import CryptoDataManager
from tdr_core.trade import Trade
from tdr_core.websocket_client import subscribe_to_websocket
from tdr_core.order_placer import OrderPlacer
from tdr_core.strategies import MACrossoverStrategy, RSITradingStrategy
from tdr_core.shell import CryptoShell

HIGH_FREQUENCY = '1H'  # Default bar size (we will override if best_strategy.json says otherwise)
STALE_FEED_SECONDS = 120  # If more than 2 minutes pass with no trades, attempt reconnect.


def run_websocket(url, symbols, data_manager, stop_event):
    """
    Launch a separate event loop to handle multiple subscribe tasks,
    including staleness detection.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [subscribe_to_websocket(url, symbol, data_manager, stop_event) for symbol in symbols]

    async def main():
        await asyncio.gather(*tasks)

    try:
        loop.run_until_complete(main())
    except Exception as e:
        data_manager.logger.error(f"WebSocket encountered error: {e}")
    finally:
        loop.close()


def setup_logging(verbose):
    logger = logging.getLogger("CryptoShellLogger")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler('crypto_shell.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def main():
    """
    Main entry point: reads best_strategy.json for config,
    parses historical log if present, then launches the CryptoShell.

    ### ADDED: We remove old trade log files to start fresh. ###
    """
    # --- NEW CODE BLOCK: remove old logs if they exist ---
    files_to_remove = ["trades.json", "non-live-trades.json"]
    for filename in files_to_remove:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"Removed old {filename} to start fresh.")
            except Exception as e:
                print(f"Unable to remove {filename}: {e}")
    # -------------------------------------------------------

    config_file = os.path.abspath("best_strategy.json")
    if not os.path.exists(config_file):
        print(f"No '{config_file}' found. Using default settings.")
        config = {}
    else:
        with open(config_file, 'r') as f:
            config = json.load(f)

    start_back = config.get('start_window_days_back', 30)
    end_back   = config.get('end_window_days_back', 0)
    do_live    = config.get('do_live_trades', False)
    max_trades = config.get('max_trades_per_day', 5)

    # NEW: We retrieve bar size from best_strategy.json
    # If not found, we default to '1H'
    bar_size = config.get("Bar_Size", "1H")
    print(f"[TDR] Using bar size: {bar_size}")

    now = datetime.now()
    start_date = now - timedelta(days=start_back) if start_back else None
    end_date   = now - timedelta(days=end_back) if end_back else None

    if start_date and end_date and start_date >= end_date:
        print("Invalid date range from best_strategy.json; ignoring end_date.")
        end_date = None

    logger = setup_logging(verbose=False)
    if do_live:
        logger.info("Running in LIVE trading mode.")
    else:
        logger.info("Running in DRY RUN mode.")

    log_file_path = os.path.abspath("btcusd.log")
    if not os.path.exists(log_file_path):
        print(f"No local log file '{log_file_path}'. Relying on real-time data only.")
        df = pd.DataFrame()
    else:
        df = parse_log_file(log_file_path, start_date, end_date)

    if not df.empty:
        df.rename(columns={'price': 'close'}, inplace=True)
        df['open'] = df['close']
        df['high'] = df['close']
        df['low'] = df['close']
        df['trades'] = 1
        if 'volume' not in df.columns:
            df['volume'] = df.get('amount', 0.0)

    data_manager = CryptoDataManager(["btcusd"], logger=logger)
    if not df.empty:
        data_manager.load_historical_data({'btcusd': df})

    order_placer = OrderPlacer()
    data_manager.order_placer = order_placer

    stop_event = threading.Event()
    shell = CryptoShell(
        data_manager=data_manager,
        order_placer=order_placer,
        logger=logger,
        verbose=False,
        live_trading=do_live,
        stop_event=stop_event,
        max_trades_per_day=max_trades
    )

    url = 'wss://ws.bitstamp.net'
    websocket_thread = threading.Thread(
        target=run_websocket, args=(url, ["btcusd"], data_manager, stop_event), daemon=True)
    websocket_thread.start()
    logger.debug("WebSocket thread started.")

    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting gracefully.")
        shell.do_quit(None)
    finally:
        stop_event.set()
        if websocket_thread.is_alive():
            websocket_thread.join()
        if shell.auto_trader and shell.auto_trader.running:
            shell.auto_trader.stop()
        if shell.chart_process and shell.chart_process.is_alive():
            shell.stop_dash_app()


###############################################################################
# BEGIN: NEW CODE FOR FLASK-BASED REST SERVER
###############################################################################
### NEW CODE ###

from flask import Flask, request, jsonify
_rest_app = Flask("tdr_rest_server")

# Global references so our server can read data from them:
GLOBAL_DATA_MANAGER = None  # We will set this from shell on start_server
GLOBAL_ACTIVE_STRATEGY = None  # We'll store "MA", "RSI", or None
GLOBAL_ACTIVE_STRATEGY_DATA = {}  # For storing e.g. short/long windows, etc.

_rest_server_thread = None
_rest_server_stop_event = threading.Event()
_rest_server_running = False

def _resample_candles(symbol, timeframe):
    """
    Helper to collect the DataFrame from data_manager, resample to timeframe,
    and return OHLC + volume as a list of dictionaries for JSON.
    """
    if (GLOBAL_DATA_MANAGER is None) or (symbol not in GLOBAL_DATA_MANAGER.data):
        return []

    df = GLOBAL_DATA_MANAGER.get_price_dataframe(symbol).copy()
    if df.empty:
        return []

    # Ensure datetime index
    df = ensure_datetime_index(df)
    # Resample
    rule_map = {
        '15m': '15T',  # 15 minutes
        '30m': '30T',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D',
        '1w': '1W'
    }
    # fallback to 1H if not recognized
    rule = rule_map.get(timeframe.lower(), '1H')

    df_resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()

    df_resampled.reset_index(inplace=True)
    candles = []
    for _, row in df_resampled.iterrows():
        candles.append({
            'timestamp': int(row['datetime'].timestamp()),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']),
        })
    return candles


@_rest_app.route('/api/strategy', methods=['GET'])
def get_strategy():
    """
    Returns the currently active strategy name (e.g. "MA") and relevant parameters.
    """
    if GLOBAL_ACTIVE_STRATEGY is None:
        return jsonify({
            'strategy': None,
            'parameters': {}
        })
    return jsonify({
        'strategy': GLOBAL_ACTIVE_STRATEGY,
        'parameters': GLOBAL_ACTIVE_STRATEGY_DATA
    })


@_rest_app.route('/api/candles', methods=['GET'])
def get_candles():
    """
    GET /api/candles?symbol=btcusd&timeframe=1h
    Returns candlestick data in JSON form
    """
    symbol = request.args.get('symbol', 'btcusd')
    timeframe = request.args.get('timeframe', '1h')
    data = _resample_candles(symbol, timeframe)
    return jsonify(data)


@_rest_app.route('/api/indicators/ma', methods=['GET'])
def get_ma_indicators():
    """
    Returns the short and long MA time-series if 'MA' strategy is active,
    otherwise returns empty or partial data
    """
    symbol = request.args.get('symbol', 'btcusd')
    timeframe = request.args.get('timeframe', '1h')

    if (GLOBAL_ACTIVE_STRATEGY != 'MA'):
        return jsonify([])  # not an MA strategy at this time

    short_window = GLOBAL_ACTIVE_STRATEGY_DATA.get('Short_Window', 12)
    long_window = GLOBAL_ACTIVE_STRATEGY_DATA.get('Long_Window', 36)

    if (GLOBAL_DATA_MANAGER is None) or (symbol not in GLOBAL_DATA_MANAGER.data):
        return jsonify([])

    df = GLOBAL_DATA_MANAGER.get_price_dataframe(symbol).copy()
    if df.empty:
        return jsonify([])

    df = ensure_datetime_index(df)
    # Resample similarly
    rule_map = {
        '15m': '15T',
        '30m': '30T',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D',
        '1w': '1W'
    }
    rule = rule_map.get(timeframe.lower(), '1H')
    df_resampled = df.resample(rule).agg({'close': 'last'}).dropna()

    # Add MAs
    df_resampled = add_moving_averages(df_resampled, short_window, long_window, price_col='close')
    df_resampled.reset_index(inplace=True)

    # Return short/long as separate lists of (timestamp, value)
    short_list = []
    long_list = []
    for _, row in df_resampled.iterrows():
        t = int(row['datetime'].timestamp())
        short_list.append({'timestamp': t, 'Short_MA': float(row['Short_MA']) if not pd.isna(row['Short_MA']) else None})
        long_list.append({'timestamp': t, 'Long_MA': float(row['Long_MA']) if not pd.isna(row['Long_MA']) else None})

    return jsonify({
        'short_ma': short_list,
        'long_ma': long_list
    })


def start_rest_server(data_manager, active_strategy_name=None, active_strategy_params=None, port=5000):
    """
    Starts the REST server in a background thread. If already running, does nothing.
    """
    global GLOBAL_DATA_MANAGER, GLOBAL_ACTIVE_STRATEGY, GLOBAL_ACTIVE_STRATEGY_DATA
    global _rest_server_running, _rest_server_stop_event, _rest_server_thread

    if _rest_server_running:
        print("REST server is already running.")
        return

    GLOBAL_DATA_MANAGER = data_manager
    GLOBAL_ACTIVE_STRATEGY = active_strategy_name
    GLOBAL_ACTIVE_STRATEGY_DATA = active_strategy_params if active_strategy_params else {}

    _rest_server_stop_event.clear()

    def _run_app():
        print("REST server started on port", port)
        _rest_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

    _rest_server_running = True
    _rest_server_thread = threading.Thread(target=_run_app, daemon=True)
    _rest_server_thread.start()


def stop_rest_server():
    """
    Signals the REST server to stop. This is a bit tricky with Flask; we do
    a workaround by sending a shutdown request to the server internally.
    """
    global _rest_server_running, _rest_server_thread

    if not _rest_server_running:
        print("REST server is not running.")
        return

    # The recommended approach is to define a shutdown route and call it:
    import requests
    try:
        requests.get('http://127.0.0.1:5000/shutdown')
    except Exception:
        pass

    if _rest_server_thread and _rest_server_thread.is_alive():
        _rest_server_thread.join(timeout=5.0)

    _rest_server_running = False
    _rest_server_thread = None
    print("REST server stopped.")


@_rest_app.route('/shutdown', methods=['GET'])
def shutdown_server():
    """
    This endpoint is used by stop_rest_server() to gracefully shut down the Flask server.
    """
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()
    return "Server shutting down..."

### END NEW CODE ###
###############################################################################
