# src/tdr_core/charting.py
#!/usr/bin/env python3
"""
charting.py - Standalone Trading Chart Module

This module provides interactive charting functionality for cryptocurrency trading strategies
using Dash and Plotly. It supports:
- Multiple strategy comparison
- Real-time data updates
- Candlestick charts with technical indicators
- Trade signal visualization
- Actual trade markers

Usage:
    from tdr_core.charting import run_dash_app
    
    # Start the chart server
    run_dash_app(
        data_manager_dict=shared_data,
        symbol='btcusd',
        bar_size='1H',
        short_window=12,
        long_window=36,
        host='0.0.0.0',
        port=8051,
        strategy_file='best_strategy.json',
        alt_strategy_file='alt_strategy-1.json'
    )
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Output, Input
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Warning: Dash and Plotly not available. Install with: pip install dash plotly")


# Helper functions moved outside of the callback to fix scoping issues
def calculate_rsi_for_chart(df, window=14):
    """Calculate RSI for charting - returns just the RSI series."""
    delta = df['close'].diff()
    gain = (delta.clip(lower=0)).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands_for_chart(df, window=20, num_std=2):
    """Calculate Bollinger Bands for charting."""
    bb_ma = df['close'].rolling(window=window).mean()
    bb_std = df['close'].rolling(window=window).std()
    bb_upper = bb_ma + (bb_std * num_std)
    bb_lower = bb_ma - (bb_std * num_std)
    return bb_ma, bb_upper, bb_lower


def run_dash_app(data_manager_dict, symbol, bar_size, short_window, long_window,
                 host='0.0.0.0', port=8050, strategy_file='best_strategy.json',
                 strategy_name='MA Strategy', alt_strategy_file=None):
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
        strategy_file: Primary strategy configuration file to use
        strategy_name: Display name for the strategy
        alt_strategy_file: Optional alternate strategy file for comparison
    """
    if not DASH_AVAILABLE:
        print("Error: Dash and Plotly are required for charting. Install with: pip install dash plotly")
        return

    app = dash.Dash(__name__)

    # Load available strategy files - dynamic discovery
    available_strategies = []

    # Always include the primary strategy file
    strategy_files_to_check = ['best_strategy.json']

    # Add alternate strategy file if provided
    if alt_strategy_file and alt_strategy_file != 'best_strategy.json':
        strategy_files_to_check.append(alt_strategy_file)

    # Also check for common alternate files if no specific alternate was provided
    if not alt_strategy_file:
        strategy_files_to_check.extend(
            ['alt_strategy-1.json', 'alt_strategy-2.json', 'alt_strategy-3.json'])

    for filename in strategy_files_to_check:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                if config.get('Strategy') == 'MA':
                    short_w = config.get('Short_Window', 12)
                    long_w = config.get('Long_Window', 36)
                    last_action = config.get('Last_Signal_Action', 'Unknown')
                    total_return = config.get('Total_Return', 0)

                    # Create descriptive label
                    file_label = "PRIMARY" if filename == 'best_strategy.json' else "ALT"
                    available_strategies.append({
                        'label': f"{file_label}: {filename} - MA({short_w},{long_w}) - {last_action} - {total_return:.1f}% return",
                        'value': filename
                    })
                elif config.get('Strategy') == 'RSI':
                    rsi_window = config.get('RSI_Window', 14)
                    overbought = config.get('Overbought', 70)
                    oversold = config.get('Oversold', 30)
                    last_action = config.get('Last_Signal_Action', 'Unknown')
                    total_return = config.get('Total_Return', 0)

                    file_label = "PRIMARY" if filename == 'best_strategy.json' else "ALT"
                    available_strategies.append({
                        'label': f"{file_label}: {filename} - RSI({rsi_window},{oversold},{overbought}) - {last_action} - {total_return:.1f}% return",
                        'value': filename
                    })
                else:
                    # Handle other strategy types
                    strategy_type = config.get('Strategy', 'Unknown')
                    total_return = config.get('Total_Return', 0)
                    last_action = config.get('Last_Signal_Action', 'Unknown')

                    file_label = "PRIMARY" if filename == 'best_strategy.json' else "ALT"
                    available_strategies.append({
                        'label': f"{file_label}: {filename} - {strategy_type} - {last_action} - {total_return:.1f}% return",
                        'value': filename
                    })
            except Exception as e:
                print(f"Error loading strategy file {filename}: {e}")

    # If no strategies were loaded, add a default
    if not available_strategies:
        available_strategies.append({
            'label': 'best_strategy.json (default)',
            'value': 'best_strategy.json'
        })

    # Define the layout
    app.layout = html.Div([
        html.H1(f'{symbol.upper()} Trading Chart',
                style={'textAlign': 'center'}),

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
                dcc.Input(id='hours-to-show', type='number',
                          value=48, min=12, max=168)
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
            current_strategy_type = 'MA'  # default

            # Initialize RSI parameters with defaults
            rsi_window = 14
            rsi_overbought = 70
            rsi_oversold = 30

            try:
                with open(current_strategy_file, 'r') as f:
                    config = json.load(f)

                current_strategy_type = config.get('Strategy', 'MA')

                if current_strategy_type == 'MA':
                    current_short_window = int(
                        config.get('Short_Window', short_window))
                    current_long_window = int(
                        config.get('Long_Window', long_window))
                    last_action = config.get('Last_Signal_Action', 'Unknown')
                    current_strategy_name = f"MA({current_short_window}, {current_long_window}) - {last_action}"
                elif current_strategy_type == 'RSI':
                    rsi_window = int(config.get('RSI_Window', 14))
                    rsi_overbought = int(config.get('Overbought', 70))
                    rsi_oversold = int(config.get('Oversold', 30))
                    last_action = config.get('Last_Signal_Action', 'Unknown')
                    current_strategy_name = f"RSI({rsi_window}, {rsi_oversold}, {rsi_overbought}) - {last_action}"
                else:
                    last_action = config.get('Last_Signal_Action', 'Unknown')
                    current_strategy_name = f"{current_strategy_type} - {last_action}"

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
                cutoff_time = df_resampled.index[-1] - \
                    pd.Timedelta(hours=hours_to_show)
                df_resampled = df_resampled[df_resampled.index >= cutoff_time]

            # Calculate indicators based on strategy type
            df_ma = df_resampled.copy()

            if current_strategy_type == 'MA':
                # Calculate moving averages - bypass the ensure_datetime_index issue
                if len(df_resampled) >= max(current_short_window, current_long_window):
                    df_ma['Short_MA'] = df_ma['close'].rolling(
                        window=current_short_window).mean()
                    df_ma['Long_MA'] = df_ma['close'].rolling(
                        window=current_long_window).mean()

                    # Generate signals manually
                    df_ma['MA_Signal'] = 0
                    df_ma.loc[df_ma['Short_MA'] >
                              df_ma['Long_MA'], 'MA_Signal'] = 1
                    df_ma.loc[df_ma['Short_MA'] <
                              df_ma['Long_MA'], 'MA_Signal'] = -1
                else:
                    df_ma['Short_MA'] = np.nan
                    df_ma['Long_MA'] = np.nan
                    df_ma['MA_Signal'] = 0
            elif current_strategy_type == 'RSI':
                # Calculate RSI
                if len(df_resampled) >= rsi_window:
                    delta = df_ma['close'].diff()
                    gain = (delta.clip(lower=0)).rolling(
                        window=rsi_window).mean()
                    loss = (-delta.clip(upper=0)
                            ).rolling(window=rsi_window).mean()
                    rs = gain / loss
                    df_ma['RSI'] = 100 - (100 / (1 + rs))

                    # Generate RSI signals
                    df_ma['RSI_Signal'] = 0
                    df_ma.loc[df_ma['RSI'] < rsi_oversold, 'RSI_Signal'] = 1
                    df_ma.loc[df_ma['RSI'] > rsi_overbought, 'RSI_Signal'] = -1
                else:
                    df_ma['RSI'] = np.nan
                    df_ma['RSI_Signal'] = 0

            # Create subplots: main chart + RSI
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Indicators', 'RSI'),
                row_heights=[0.7, 0.3]
            )

            # Add candlestick chart to main subplot
            fig.add_trace(go.Candlestick(
                x=df_ma.index,
                open=df_ma['open'],
                high=df_ma['high'],
                low=df_ma['low'],
                close=df_ma['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ), row=1, col=1)

            # Calculate and add Bollinger Bands
            bb_ma, bb_upper, bb_lower = calculate_bollinger_bands_for_chart(df_ma)
            
            # Only add BB traces where we have valid data
            valid_bb_upper = bb_upper.dropna()
            valid_bb_lower = bb_lower.dropna()
            valid_bb_ma = bb_ma.dropna()
            
            if not valid_bb_upper.empty:
                fig.add_trace(go.Scatter(
                    x=valid_bb_upper.index,
                    y=valid_bb_upper,
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='rgba(255,0,0,0.3)', width=1, dash='dash'),
                    showlegend=True
                ), row=1, col=1)
            
            if not valid_bb_lower.empty:
                fig.add_trace(go.Scatter(
                    x=valid_bb_lower.index,
                    y=valid_bb_lower,
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='rgba(0,255,0,0.3)', width=1, dash='dash'),
                    showlegend=True,
                    fill='tonexty' if not valid_bb_upper.empty else None,
                    fillcolor='rgba(128,128,128,0.1)'
                ), row=1, col=1)
            
            if not valid_bb_ma.empty:
                fig.add_trace(go.Scatter(
                    x=valid_bb_ma.index,
                    y=valid_bb_ma,
                    mode='lines',
                    name='BB Middle (SMA20)',
                    line=dict(color='orange', width=1),
                    showlegend=True
                ), row=1, col=1)
            
            # Calculate and add RSI
            rsi_values = calculate_rsi_for_chart(df_ma)
            valid_rsi = rsi_values.dropna()
            
            if not valid_rsi.empty:
                fig.add_trace(go.Scatter(
                    x=valid_rsi.index,
                    y=valid_rsi,
                    mode='lines',
                    name='RSI (14)',
                    line=dict(color='purple', width=2),
                    showlegend=False
                ), row=2, col=1)
                
                # Add RSI reference lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", 
                             annotation_text="Overbought (70)", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", 
                             annotation_text="Oversold (30)", row=2, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", 
                             annotation_text="Neutral (50)", row=2, col=1)

            # Add indicators based on strategy type
            if current_strategy_type == 'MA' and len(df_ma) >= max(current_short_window, current_long_window):
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
                    ), row=1, col=1)

                if not valid_long.empty:
                    fig.add_trace(go.Scatter(
                        x=valid_long.index,
                        y=valid_long,
                        mode='lines',
                        name=f'MA({current_long_window})',
                        line=dict(color='cyan', width=2)
                    ), row=1, col=1)

                # Add signal markers
                buy_signals = df_ma[df_ma['MA_Signal'] == 1]
                sell_signals = df_ma[df_ma['MA_Signal'] == -1]

                if not buy_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['close'],
                        mode='markers',
                        marker=dict(symbol='triangle-up',
                                    size=15, color='green'),
                        name='Buy Signals'
                    ), row=1, col=1)

                if not sell_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['close'],
                        mode='markers',
                        marker=dict(symbol='triangle-down',
                                    size=15, color='red'),
                        name='Sell Signals'
                    ), row=1, col=1)

            elif current_strategy_type == 'RSI' and len(df_ma) >= rsi_window:
                # Add RSI signals
                buy_signals = df_ma[df_ma['RSI_Signal'] == 1]
                sell_signals = df_ma[df_ma['RSI_Signal'] == -1]

                if not buy_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['close'],
                        mode='markers',
                        marker=dict(symbol='triangle-up',
                                    size=15, color='green'),
                        name='RSI Buy Signals'
                    ), row=1, col=1)

                if not sell_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['close'],
                        mode='markers',
                        marker=dict(symbol='triangle-down',
                                    size=15, color='red'),
                        name='RSI Sell Signals'
                    ), row=1, col=1)

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
                                cutoff_time = df_ma.index[-1] - \
                                    pd.Timedelta(hours=hours_to_show)
                                if trade_time < cutoff_time:
                                    continue

                            trade_times.append(trade_time)
                            trade_prices.append(trade['price'])
                            trade_colors.append(
                                'lightgreen' if trade['type'] == 'buy' else 'lightcoral')
                            trade_text.append(
                                f"{trade['type'].upper()}<br>{trade['amount']:.6f} BTC<br>${trade['price']:.2f}")

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
                            ), row=1, col=1)
            except Exception as e:
                print(f"Error loading trades: {e}")

            # Update layout
            strategy_display = f"({current_strategy_file})" if current_strategy_file != 'best_strategy.json' else ""
            fig.update_layout(
                title=f'{symbol.upper()} - {bar_size} Bars - {current_strategy_name} {strategy_display} - Last {hours_to_show}h',
                template='plotly_white',
                showlegend=True,
                height=800,
                xaxis_rangeslider_visible=False
            )

            # Update y-axis labels
            fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
            fig.update_xaxes(title_text="Time", row=2, col=1)

            # Create status info
            latest_data = df_ma.iloc[-1] if not df_ma.empty else None
            if latest_data is not None:
                current_price = latest_data['close']

                status_children = [
                    html.P(f"Strategy: {current_strategy_name}"),
                    html.P(f"File: {current_strategy_file}"),
                    html.P(f"Current Price: ${current_price:.2f}"),
                    html.P(f"Data Points: {len(df_ma)} bars"),
                    html.P(
                        f"Time Range: {df_ma.index[0].strftime('%m-%d %H:%M')} to {df_ma.index[-1].strftime('%m-%d %H:%M')}")
                ]

                if current_strategy_type == 'MA':
                    short_ma = latest_data.get('Short_MA', np.nan)
                    long_ma = latest_data.get('Long_MA', np.nan)
                    signal = latest_data.get('MA_Signal', 0)

                    if not pd.isna(short_ma) and not pd.isna(long_ma):
                        signal_text = {1: 'BUY', -1: 'SELL',
                                       0: 'HOLD'}.get(signal, 'UNKNOWN')
                        signal_color = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}.get(
                            signal_text, 'black')

                        status_children.extend([
                            html.P(
                                f"MA({current_short_window}): ${short_ma:.2f}"),
                            html.P(
                                f"MA({current_long_window}): ${long_ma:.2f}"),
                            html.P(f"Current Signal: ", style={
                                   'display': 'inline'}),
                            html.Span(signal_text, style={
                                      'color': signal_color, 'fontWeight': 'bold'})
                        ])
                    else:
                        status_children.append(
                            html.P("Insufficient data for moving averages"))

                elif current_strategy_type == 'RSI':
                    rsi_val = latest_data.get('RSI', np.nan)
                    signal = latest_data.get('RSI_Signal', 0)

                    if not pd.isna(rsi_val):
                        signal_text = {1: 'BUY', -1: 'SELL',
                                       0: 'HOLD'}.get(signal, 'UNKNOWN')
                        signal_color = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}.get(
                            signal_text, 'black')

                        status_children.extend([
                            html.P(f"RSI({rsi_window}): {rsi_val:.2f}"),
                            html.P(
                                f"Overbought: {rsi_overbought}, Oversold: {rsi_oversold}"),
                            html.P(f"Current Signal: ", style={
                                   'display': 'inline'}),
                            html.Span(signal_text, style={
                                      'color': signal_color, 'fontWeight': 'bold'})
                        ])
                    else:
                        status_children.append(
                            html.P("Insufficient data for RSI"))

                status_children.append(
                    html.P(f"Last Update: {latest_data.name.strftime('%Y-%m-%d %H:%M:%S')}"))
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

    if alt_strategy_file:
        print(f"\nLoaded alternate strategy file: {alt_strategy_file}")
        print("Use the dropdown in the web interface to compare strategies.")

    try:
        app.run_server(host=host, port=port, debug=False)
    except Exception as e:
        print(f"Error running Dash app: {e}")


def create_sample_data_manager_dict(symbol='btcusd'):
    """
    Create a sample data manager dictionary for testing purposes.
    In real usage, this would be populated by your data manager.
    """
    import random
    from datetime import datetime, timedelta

    # Generate sample OHLC data
    base_price = 50000
    timestamps = []
    prices = []

    start_time = datetime.now() - timedelta(hours=100)
    for i in range(100):
        timestamp = start_time + timedelta(hours=i)
        price = base_price + random.randint(-2000, 2000)
        timestamps.append(int(timestamp.timestamp()))
        prices.append(price)

    data_dict = {
        symbol: {
            'timestamp': timestamps,
            'close': prices,
            'open': prices,
            'high': [p + random.randint(0, 500) for p in prices],
            'low': [p - random.randint(0, 500) for p in prices],
            'volume': [random.randint(100, 10000) for _ in prices],
            'amount': [random.randint(1, 100) for _ in prices]
        }
    }

    return data_dict


if __name__ == "__main__":
    # Example usage for testing
    if DASH_AVAILABLE:
        print("Running charting module in standalone mode...")

        # Create sample data
        sample_data = create_sample_data_manager_dict()

        # Run the chart
        run_dash_app(
            data_manager_dict=sample_data,
            symbol='btcusd',
            bar_size='1H',
            short_window=12,
            long_window=36,
            host='127.0.0.1',
            port=8050,
            strategy_file='best_strategy.json',
            alt_strategy_file=None
        )
    else:
        print("Dash and Plotly are required to run the charting module.")
        print("Install with: pip install dash plotly")
