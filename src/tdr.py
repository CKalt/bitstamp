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
#   4) FIXED: DataFrame timestamp column handling and data processing issues.
#   5) ADDED: Show recent 48 hours of data by default for better visibility.
#   6) FIXED: Timestamp column preservation in resampling and bypassed
#      ensure_datetime_index issue by calculating moving averages directly.
#   7) ADDED: Strategy file parameter and dropdown for viewing alternate
#      strategies without trading them.
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


def run_dash_app(data_manager_dict, symbol, bar_size, short_window, long_window, host='0.0.0.0', port=8050, strategy_file='best_strategy.json', strategy_name='MA Strategy'):
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
        strategy_file: Strategy configuration file to use
        strategy_name: Display name for the strategy
    """
    app = dash.Dash(__name__)
    
    # Load available strategy files
    available_strategies = []
    for filename in ['best_strategy.json', 'alt_strategy-1.json']:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                if config.get('Strategy') == 'MA':
                    short_w = config.get('Short_Window', 12)
                    long_w = config.get('Long_Window', 36)
                    last_action = config.get('Last_Signal_Action', 'Unknown')
                    total_return = config.get('Total_Return', 0)
                    available_strategies.append({
                        'label': f"{filename}: MA({short_w},{long_w}) - {last_action} - {total_return:.1f}% return",
                        'value': filename
                    })
            except:
                pass
    
    # Define the layout
    app.layout = html.Div([
        html.H1(f'{symbol.upper()} Trading Chart', style={'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.Label('Strategy:'),
                dcc.Dropdown(
                    id='strategy-dropdown',
                    options=available_strategies,
                    value=strategy_file,
                    style={'width': '100%'}
                )
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label('Hours to Display:'),
                dcc.Input(id='hours-to-show', type='number', value=48, min=12, max=168)
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
         Input('strategy-dropdown', 'value'),
         Input('hours-to-show', 'value')]
    )
    def update_graph(n, selected_strategy, hours_to_show):
        try:
            # Load strategy configuration
            current_strategy_file = selected_strategy or strategy_file
            current_short_window = short_window
            current_long_window = long_window
            current_strategy_name = strategy_name
            
            try:
                with open(current_strategy_file, 'r') as f:
                    config = json.load(f)
                if config.get('Strategy') == 'MA':
                    current_short_window = int(config.get('Short_Window', short_window))
                    current_long_window = int(config.get('Long_Window', long_window))
                    last_action = config.get('Last_Signal_Action', 'Unknown')
                    current_strategy_name = f"MA({current_short_window}, {current_long_window}) - {last_action}"
            except Exception as e:
                print(f"Error loading strategy {current_strategy_file}: {e}")
            
            # Get data from shared dictionary
            if symbol not in data_manager_dict:
                return {}, "No data available"
            
            data_dict = dict(data_manager_dict[symbol])
            
            if not data_dict or not data_dict.get('timestamp'):
                return {}, "No data available"
            
            # Convert to DataFrame - handle both list and dict formats
            df = pd.DataFrame(data_dict)
            
            if df.empty:
                return {}, "No data available"
            
            # Ensure we have required columns
            required_columns = ['timestamp', 'close']
            for col in required_columns:
                if col not in df.columns:
                    return {}, f"Missing required column: {col}"
            
            # Handle different column names that might exist
            if 'price' in df.columns and 'close' not in df.columns:
                df['close'] = df['price']
            
            # Ensure we have OHLC data
            if 'open' not in df.columns:
                df['open'] = df['close']
            if 'high' not in df.columns:
                df['high'] = df['close']
            if 'low' not in df.columns:
                df['low'] = df['close']
            if 'volume' not in df.columns:
                df['volume'] = df.get('amount', 0.0)
            
            # Create datetime column and set as index
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            # Resample to specified bar size
            df_resampled = df.resample(bar_size).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'timestamp': 'last'  # Keep the timestamp column
            }).dropna()
            
            if len(df_resampled) == 0:
                return {}, "No resampled data available"
            
            # Limit to recent hours
            if hours_to_show and len(df_resampled) > 0:
                cutoff_time = df_resampled.index[-1] - pd.Timedelta(hours=hours_to_show)
                df_resampled = df_resampled[df_resampled.index >= cutoff_time]
            
            # Calculate moving averages - bypass the ensure_datetime_index issue
            if len(df_resampled) >= max(current_short_window, current_long_window):
                # Create a copy and manually add moving averages without calling ensure_datetime_index
                df_ma = df_resampled.copy()
                df_ma['Short_MA'] = df_ma['close'].rolling(window=current_short_window).mean()
                df_ma['Long_MA'] = df_ma['close'].rolling(window=current_long_window).mean()
                
                # Generate signals manually
                df_ma['MA_Signal'] = 0
                df_ma.loc[df_ma['Short_MA'] > df_ma['Long_MA'], 'MA_Signal'] = 1
                df_ma.loc[df_ma['Short_MA'] < df_ma['Long_MA'], 'MA_Signal'] = -1
            else:
                df_ma = df_resampled.copy()
                df_ma['Short_MA'] = np.nan
                df_ma['Long_MA'] = np.nan
                df_ma['MA_Signal'] = 0
            
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
            
            # Add moving averages if we have enough data
            if len(df_ma) >= max(current_short_window, current_long_window):
                # Only plot MAs where we have valid data
                valid_short = df_ma['Short_MA'].dropna()
                valid_long = df_ma['Long_MA'].dropna()
                
                if not valid_short.empty:
                    fig.add_trace(go.Scatter(
                        x=valid_short.index,
                        y=valid_short,
                        mode='lines',
                        name=f'MA({current_short_window})',
                        line=dict(color='blue', width=2)
                    ))
                
                if not valid_long.empty:
                    fig.add_trace(go.Scatter(
                        x=valid_long.index,
                        y=valid_long,
                        mode='lines',
                        name=f'MA({current_long_window})',
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
            
            # Load and display actual trades if available (only for best_strategy.json)
            try:
                if current_strategy_file == 'best_strategy.json' and os.path.exists('trades.json'):
                    with open('trades.json', 'r') as f:
                        trades_data = json.load(f)
                    
                    if trades_data:
                        trade_times = []
                        trade_prices = []
                        trade_colors = []
                        trade_text = []
                        
                        for trade in trades_data:
                            trade_time = pd.to_datetime(trade['timestamp'])
                            # Only show trades within our time window
                            if hours_to_show:
                                cutoff_time = df_ma.index[-1] - pd.Timedelta(hours=hours_to_show)
                                if trade_time < cutoff_time:
                                    continue
                                    
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
                title=f'{symbol.upper()} - {bar_size} Bars - {current_strategy_name} - Last {hours_to_show}h',
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
                short_ma = latest_data.get('Short_MA', np.nan)
                long_ma = latest_data.get('Long_MA', np.nan)
                signal = latest_data.get('MA_Signal', 0)
                
                signal_text = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}.get(signal, 'UNKNOWN')
                signal_color = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}.get(signal_text, 'black')
                
                status_children = [
                    html.P(f"Strategy: {current_strategy_name}"),
                    html.P(f"Current Price: ${current_price:.2f}"),
                    html.P(f"Data Points: {len(df_ma)} bars"),
                    html.P(f"Time Range: {df_ma.index[0].strftime('%m-%d %H:%M')} to {df_ma.index[-1].strftime('%m-%d %H:%M')}")
                ]
                
                if not pd.isna(short_ma) and not pd.isna(long_ma):
                    status_children.extend([
                        html.P(f"MA({current_short_window}): ${short_ma:.2f}"),
                        html.P(f"MA({current_long_window}): ${long_ma:.2f}"),
                        html.P(f"Current Signal: ", style={'display': 'inline'}),
                        html.Span(signal_text, style={'color': signal_color, 'fontWeight': 'bold'})
                    ])
                else:
                    status_children.append(html.P("Insufficient data for moving averages"))
                
                status_children.append(html.P(f"Last Update: {latest_data.name.strftime('%Y-%m-%d %H:%M:%S')}"))
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
