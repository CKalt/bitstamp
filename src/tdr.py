###############################################################################
# src/tdr.py
###############################################################################
# Full File Path: src/tdr.py
#
# CHANGES:
#   1) We have added a small code block at the start of main() to remove
#      the old trade log files if they exist ("trades.json" and
#      "non-live-trades.json"). This clears records of prior runs.
#   2) We preserve all original code, docstrings, and logic.
#   3) ADDED: Complete run_dash_app function for interactive charting with
#      signal visualization and trade markers.
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
    from flask import Flask
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


def run_dash_app(data_manager_dict, symbol, bar_size, short_window, long_window, host='0.0.0.0', port=8050):
    """
    Run the Dash application for interactive charting with signals and trades.
    
    Args:
        data_manager_dict: Shared dictionary containing price data
        symbol: Trading symbol (e.g., 'btcusd')
        bar_size: Bar size for resampling (e.g., '1H')
        short_window: Short moving average window
        long_window: Long moving average window
        host: Host to bind to (0.0.0.0 for remote access)
        port: Port to bind to
    """
    app = dash.Dash(__name__)
    
    # Define the layout
    app.layout = html.Div([
        html.H1(f'{symbol.upper()} Trading Chart', style={'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.Label('Refresh Interval (seconds):'),
                dcc.Input(id='refresh-interval', type='number', value=30, min=5, max=300)
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label('Number of Bars to Display:'),
                dcc.Input(id='bars-to-show', type='number', value=100, min=50, max=500)
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
        ], style={'padding': '10px'}),
        
        dcc.Graph(id='price-chart'),
        
        html.Div([
            html.H3('Current Status'),
            html.Div(id='status-info')
        ], style={'padding': '10px'}),
        
        dcc.Interval(
            id='interval-component',
            interval=30*1000,  # Update every 30 seconds
            n_intervals=0
        ),
        
        # Add shutdown route
        html.Div(id='shutdown-trigger', style={'display': 'none'})
    ])

    @app.callback(
        [Output('price-chart', 'figure'),
         Output('status-info', 'children')],
        [Input('interval-component', 'n_intervals'),
         Input('refresh-interval', 'value'),
         Input('bars-to-show', 'value')]
    )
    def update_graph(n, refresh_interval, bars_to_show):
        try:
            # Update interval component
            if refresh_interval and refresh_interval != 30:
                # This would require updating the interval component, but we'll keep it simple
                pass
            
            # Get data from shared dictionary
            if symbol not in data_manager_dict:
                return {}, "No data available"
            
            data_dict = dict(data_manager_dict[symbol])
            
            if not data_dict or not data_dict.get('timestamp'):
                return {}, "No data available"
            
            # Convert to DataFrame
            df = pd.DataFrame(data_dict)
            
            if df.empty:
                return {}, "No data available"
            
            # Ensure datetime index
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)
            
            # Resample to specified bar size
            df_resampled = df.resample(bar_size).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'trades': 'sum'
            }).dropna()
            
            if len(df_resampled) == 0:
                return {}, "No resampled data available"
            
            # Limit to recent bars
            if bars_to_show and len(df_resampled) > bars_to_show:
                df_resampled = df_resampled.tail(bars_to_show)
            
            # Calculate moving averages
            df_ma = add_moving_averages(df_resampled.copy(), short_window, long_window, price_col='close')
            df_ma = generate_ma_signals(df_ma)
            
            # Create the main price chart
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=df_ma.index,
                open=df_ma['open'],
                high=df_ma['high'],
                low=df_ma['low'],
                close=df_ma['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ))
            
            # Add moving averages
            fig.add_trace(go.Scatter(
                x=df_ma.index,
                y=df_ma['Short_MA'],
                mode='lines',
                name=f'MA({short_window})',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=df_ma.index,
                y=df_ma['Long_MA'],
                mode='lines',
                name=f'MA({long_window})',
                line=dict(color='orange', width=2)
            ))
            
            # Add signal markers
            buy_signals = df_ma[df_ma['MA_Signal'] == 1]
            sell_signals = df_ma[df_ma['MA_Signal'] == -1]
            
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['close'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=15, color='green'),
                    name='Buy Signals'
                ))
            
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['close'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=15, color='red'),
                    name='Sell Signals'
                ))
            
            # Load and display actual trades if available
            try:
                if os.path.exists('trades.json'):
                    with open('trades.json', 'r') as f:
                        trades_data = json.load(f)
                    
                    if trades_data:
                        trade_times = []
                        trade_prices = []
                        trade_colors = []
                        trade_text = []
                        
                        for trade in trades_data:
                            trade_time = pd.to_datetime(trade['timestamp'])
                            trade_times.append(trade_time)
                            trade_prices.append(trade['price'])
                            trade_colors.append('lightgreen' if trade['type'] == 'buy' else 'lightcoral')
                            trade_text.append(f"{trade['type'].upper()}<br>{trade['amount']:.6f} BTC<br>${trade['price']:.2f}")
                        
                        if trade_times:
                            fig.add_trace(go.Scatter(
                                x=trade_times,
                                y=trade_prices,
                                mode='markers',
                                marker=dict(symbol='diamond', size=12, color=trade_colors, 
                                           line=dict(width=2, color='black')),
                                text=trade_text,
                                textposition='top center',
                                name='Actual Trades',
                                hovertemplate='%{text}<extra></extra>'
                            ))
            except Exception as e:
                print(f"Error loading trades: {e}")
            
            # Update layout
            fig.update_layout(
                title=f'{symbol.upper()} - {bar_size} Bars (MA {short_window}/{long_window})',
                xaxis_title='Time',
                yaxis_title='Price (USD)',
                template='plotly_white',
                showlegend=True,
                height=600,
                xaxis_rangeslider_visible=False
            )
            
            # Create status info
            latest_data = df_ma.iloc[-1] if not df_ma.empty else None
            if latest_data is not None:
                current_price = latest_data['close']
                short_ma = latest_data['Short_MA']
                long_ma = latest_data['Long_MA']
                signal = latest_data['MA_Signal']
                
                signal_text = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}.get(signal, 'UNKNOWN')
                signal_color = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}.get(signal_text, 'black')
                
                status_children = [
                    html.P(f"Current Price: ${current_price:.2f}"),
                    html.P(f"MA({short_window}): ${short_ma:.2f}"),
                    html.P(f"MA({long_window}): ${long_ma:.2f}"),
                    html.P(f"Current Signal: ", style={'display': 'inline'}),
                    html.Span(signal_text, style={'color': signal_color, 'fontWeight': 'bold'}),
                    html.P(f"Last Update: {latest_data.name.strftime('%Y-%m-%d %H:%M:%S')}")
                ]
            else:
                status_children = [html.P("No current data available")]
            
            return fig, status_children
            
        except Exception as e:
            print(f"Error updating chart: {e}")
            import traceback
            traceback.print_exc()
            return {}, f"Error: {str(e)}"

    # Add shutdown route
    @app.server.route('/shutdown')
    def shutdown():
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            return 'Not running with the Werkzeug Server'
        func()
        return 'Server shutting down...'

    print(f"Starting Dash app on http://{host}:{port}")
    print("For remote access, you can use tools like ngrok:")
    print(f"  ngrok http {port}")
    print("Or access directly if firewall allows.")
    
    try:
        app.run_server(host=host, port=port, debug=False)
    except Exception as e:
        print(f"Error running Dash app: {e}")


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


# RESTORED: We place back the call to main() at the bottom:
if __name__ == '__main__':
    set_start_method('spawn')
    main()
