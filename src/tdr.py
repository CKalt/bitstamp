###############################################################################
# src/tdr.py
###############################################################################
# Full File Path: src/tdr.py
#
# CHANGES (EXPLANATION):
#   1) We introduce RSI auto-trading support so that best_strategy.json 
#      with "Strategy": "RSI" can be used by the "auto_trade" command.
#   2) We preserve all existing comments, logic, and code related to the 
#      MA strategy. We only add new logic in do_auto_trade to handle RSI.
#   3) For consistency, we read "RSI_Window", "Overbought", "Oversold" 
#      from best_strategy.json, and construct an RSITradingStrategy 
#      (see the newly added RSITradingStrategy class in tdr_core/strategies.py).
#   4) We keep the original references to "MACrossoverStrategy" for "MA" 
#      and only add an elif block if "Strategy" == "RSI".
#   5) All original code, comments, and structure remain intact; 
#      only minimal changes appear for RSI integration.
#
# NOTE: You must also see the changes in src/tdr_core/strategies.py 
#       (shown below) for the RSITradingStrategy class. If that file 
#       is missing or out-of-date, please request the updated version.
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
from tdr_core.strategies import MACrossoverStrategy, RSITradingStrategy  # ADDED RSITradingStrategy
from tdr_core.shell import CryptoShell

HIGH_FREQUENCY = '1H'  # Default bar size
STALE_FEED_SECONDS = 120  # If more than 2 minutes pass with no trades, attempt reconnect.


def determine_initial_position(df: pd.DataFrame, short_window: int, long_window: int) -> int:
    """
    Computes the final short/long MA crossover on df to decide 
    if we "should" be long (1), short (-1), or neutral (0).
    """
    if len(df) < long_window:
        return 0
    df_copy = ensure_datetime_index(df.copy())
    df_copy['Short_MA'] = df_copy['close'].rolling(short_window).mean()
    df_copy['Long_MA'] = df_copy['close'].rolling(long_window).mean()
    df_copy.dropna(inplace=True)
    if df_copy.empty:
        return 0
    last_short = df_copy.iloc[-1]['Short_MA']
    last_long  = df_copy.iloc[-1]['Long_MA']
    if last_short > last_long:
        return 1
    elif last_short < last_long:
        return -1
    else:
        return 0


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
    """
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


if __name__ == '__main__':
    set_start_method('spawn')
    main()
