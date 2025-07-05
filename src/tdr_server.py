#!/usr/bin/env python
# src/tdr_server.py
# Remote TDR server - pure execution engine with no local configuration

import sys
import os
import json
import logging
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    print("Error: Flask is not installed. Please run: pip install flask flask-cors")
    sys.exit(1)
import argparse

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

# Import core TDR components
from data.loader import parse_log_file
from tdr_core.data_manager import CryptoDataManager
from tdr_core.trade import Trade
from tdr_core.websocket_client import subscribe_to_websocket
from tdr_core.order_placer import OrderPlacer
from tdr_core.strategies import MACrossoverStrategy, AdaptiveMultiStrategy
from tdr_core.shell import CryptoShell

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for server state
data_manager = None
order_placer = None
shell = None
auto_trader = None
websocket_thread = None
stop_event = threading.Event()
logger = None
server_config = {}
initialization_complete = False

def setup_logging(verbose=False):
    """Configure server logging"""
    global logger
    logger = logging.getLogger("TDRServer")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    
    # Enhanced formatter with more context
    formatter = logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
    logger.handlers.clear()
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    file_handler = logging.FileHandler('logs/tdr_server.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize server with configuration from client"""
    global data_manager, order_placer, shell, websocket_thread, server_config, initialization_complete
    
    try:
        config = request.json
        if not config:
            return jsonify({'error': 'No configuration provided'}), 400
        
        logger.info("Initializing server with client configuration...")
        server_config = config
        
        # Extract key configuration
        best_strategy = config.get('best_strategy', {})
        do_live_trades = best_strategy.get('do_live_trades', False)
        verbose = config.get('verbose', False)
        
        # Create data manager with required arguments
        data_manager = CryptoDataManager(["btcusd"], logger=logger, verbose=verbose)
        
        # Mark that historical data loading is pending
        server_config['history_loaded'] = False
        server_config['history_loading'] = False
        
        # Create order placer
        order_placer = OrderPlacer(
            do_live_trades=do_live_trades,
            verbose=verbose,
            data_manager=data_manager
        )
        
        # Initialize position from config if provided
        if 'initial_position' in config:
            pos = config['initial_position']
            data_manager.balance_btc = pos.get('btc_balance', 0)
            data_manager.balance_usd = pos.get('usd_balance', 10000)
            data_manager.position = pos.get('position', 0)
            data_manager.position_size = pos.get('position_size', 0)
            data_manager.position_cost_basis = pos.get('position_cost_basis', 0)
            logger.info(f"Initialized position: BTC={data_manager.balance_btc}, USD={data_manager.balance_usd}")
        
        # Create shell instance
        shell = CryptoShell(
            data_manager=data_manager,
            order_placer=order_placer,
            do_chart=False  # Never run chart on server
        )
        
        # Apply strategy configuration
        shell.config = best_strategy
        
        # Start WebSocket if enabled
        if config.get('enable_websocket', True):
            websocket_url = "wss://ws.bitstamp.net"
            symbols = ["btcusd"]
            
            def run_websocket():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                tasks = [subscribe_to_websocket(websocket_url, symbol, data_manager, stop_event) 
                        for symbol in symbols]
                
                async def main():
                    await asyncio.gather(*tasks)
                
                try:
                    loop.run_until_complete(main())
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                finally:
                    loop.close()
            
            websocket_thread = threading.Thread(target=run_websocket, daemon=True)
            websocket_thread.start()
            logger.info("WebSocket connection started")
        
        initialization_complete = True
        logger.info("Server initialization complete")
        
        return jsonify({
            'success': True,
            'message': 'Server initialized successfully',
            'config_summary': {
                'do_live_trades': do_live_trades,
                'strategy': best_strategy.get('Strategy', 'Unknown'),
                'websocket': config.get('enable_websocket', True),
                'historical_data_loaded': not df.empty if 'df' in locals() else False
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current system status"""
    try:
        if not initialization_complete:
            return jsonify({'error': 'Server not initialized'}), 503
            
        status = {
            'server': 'running',
            'initialized': initialization_complete,
            'timestamp': datetime.now().isoformat(),
            'websocket': 'connected' if websocket_thread and websocket_thread.is_alive() else 'disconnected',
            'data_manager': 'initialized' if data_manager else 'not initialized',
            'order_placer': 'initialized' if order_placer else 'not initialized',
            'live_trading': order_placer.do_live_trades if order_placer else False
        }
        
        if data_manager:
            status['position'] = {
                'btc_balance': data_manager.balance_btc,
                'usd_balance': data_manager.balance_usd,
                'position': data_manager.position,
                'position_size': data_manager.position_size,
                'entry_price': data_manager.position_cost_basis / data_manager.position_size 
                              if data_manager.position_size > 0 else 0
            }
            status['last_price'] = data_manager.get_last_price('btcusd')
        
        if shell and shell.auto_trader:
            status['auto_trader'] = {
                'active': shell.auto_trader.active,
                'strategy': type(shell.auto_trader.strategy).__name__,
                'trades_today': shell.auto_trader.trades_today
            }
        
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Error in get_status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/command', methods=['POST'])
def execute_command():
    """Execute a shell command"""
    try:
        if not initialization_complete:
            return jsonify({'error': 'Server not initialized. Call /api/initialize first'}), 503
            
        if not shell:
            return jsonify({'error': 'Shell not initialized'}), 503
        
        data = request.json
        command = data.get('command')
        source = data.get('source', 'unknown')
        client_ip = request.remote_addr
        
        if not command:
            return jsonify({'error': 'No command provided'}), 400
        
        # Enhanced logging with source and client info
        logger.info(f"[CMD-EXEC] Source: {source} | Client: {client_ip} | Command: {command}")
        
        # Capture shell output
        import io
        from contextlib import redirect_stdout
        
        output_buffer = io.StringIO()
        
        try:
            with redirect_stdout(output_buffer):
                shell.onecmd(command)
            
            output = output_buffer.getvalue()
            
            result = {
                'command': command,
                'output': output,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            # Add state updates for relevant commands
            if any(cmd in command for cmd in ['buy', 'sell', 'status', 'position', 'auto_trade']):
                result['position'] = {
                    'btc_balance': data_manager.balance_btc,
                    'usd_balance': data_manager.balance_usd,
                    'position': data_manager.position,
                    'entry_price': data_manager.position_cost_basis / data_manager.position_size 
                                  if data_manager.position_size > 0 else 0
                }
            
            # Update global auto_trader reference if changed
            if shell.auto_trader:
                global auto_trader
                auto_trader = shell.auto_trader
                result['auto_trader'] = {
                    'active': auto_trader.active,
                    'strategy': type(auto_trader.strategy).__name__
                }
            
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return jsonify({
                'command': command,
                'error': str(e),
                'success': False
            }), 400
            
    except Exception as e:
        logger.error(f"Error in execute_command: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/price/<symbol>', methods=['GET'])
def get_price(symbol):
    """Get current price for a symbol"""
    try:
        if not data_manager:
            return jsonify({'error': 'Server not initialized'}), 503
        
        price = data_manager.get_last_price(symbol)
        if price:
            return jsonify({
                'symbol': symbol,
                'price': price,
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({'error': f'No price data for {symbol}'}), 404
    except Exception as e:
        logger.error(f"Error in get_price: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/<symbol>', methods=['GET'])
def get_data(symbol):
    """Get recent data for a symbol"""
    try:
        if not data_manager:
            return jsonify({'error': 'Server not initialized'}), 503
        
        limit = request.args.get('limit', 100, type=int)
        frequency = request.args.get('frequency', '1H')
        
        df = data_manager.get_data_as_dataframe(symbol, frequency)
        if df is not None and not df.empty:
            data = df.tail(limit).reset_index().to_dict('records')
            return jsonify({
                'symbol': symbol,
                'frequency': frequency,
                'data': data
            }), 200
        else:
            return jsonify({'error': f'No data for {symbol}'}), 404
    except Exception as e:
        logger.error(f"Error in get_data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ping', methods=['GET'])
def ping():
    """Simple ping endpoint for health checks"""
    return jsonify({
        'status': 'pong',
        'timestamp': datetime.now().isoformat(),
        'initialized': initialization_complete
    }), 200

@app.route('/api/load_history', methods=['POST'])
def load_history():
    """Load historical data in the background"""
    global server_config
    
    if not initialization_complete:
        return jsonify({'error': 'Server not initialized'}), 503
        
    if server_config.get('history_loading', False):
        return jsonify({'status': 'already_loading', 'message': 'History is already being loaded'}), 200
        
    if server_config.get('history_loaded', False):
        return jsonify({'status': 'already_loaded', 'message': 'History has already been loaded'}), 200
    
    # Start loading in background
    def load_history_background():
        try:
            server_config['history_loading'] = True
            logger.info("Starting historical data load...")
            
            log_file = 'btcusd.log'
            if os.path.exists(log_file):
                # Get config parameters
                best_strategy = server_config.get('best_strategy', {})
                start_back = best_strategy.get('start_window_days_back', 30)
                end_back = best_strategy.get('end_window_days_back', 0)
                now = datetime.now()
                start_date = now - timedelta(days=start_back) if start_back else None
                end_date = now - timedelta(days=end_back) if end_back else None
                
                logger.info(f"Loading historical data from {log_file}")
                df = parse_log_file(log_file, start_date=start_date, end_date=end_date)
                
                if not df.empty:
                    # Need to process the dataframe like in main tdr.py
                    df.rename(columns={'price': 'close'}, inplace=True)
                    df['open'] = df['close']
                    df['high'] = df['close']
                    df['low'] = df['close']
                    df['trades'] = 1
                    if 'volume' not in df.columns:
                        df['volume'] = df.get('amount', 0.0)
                    
                    data_manager.load_historical_data({'btcusd': df})
                    logger.info(f"Loaded {len(df)} historical records")
                    server_config['history_record_count'] = len(df)
            
            server_config['history_loaded'] = True
            server_config['history_loading'] = False
            logger.info("Historical data load complete")
            
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            server_config['history_loading'] = False
            server_config['history_error'] = str(e)
    
    # Start in background thread
    history_thread = threading.Thread(target=load_history_background, daemon=True)
    history_thread.start()
    
    return jsonify({
        'status': 'loading_started',
        'message': 'Historical data loading started in background'
    }), 200

@app.route('/api/history_status', methods=['GET'])
def history_status():
    """Check historical data loading status"""
    if not initialization_complete:
        return jsonify({'error': 'Server not initialized'}), 503
        
    return jsonify({
        'history_loaded': server_config.get('history_loaded', False),
        'history_loading': server_config.get('history_loading', False),
        'history_error': server_config.get('history_error', None),
        'record_count': server_config.get('history_record_count', 0)
    }), 200

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get active server configuration for verification"""
    if not initialization_complete:
        return jsonify({'error': 'Server not initialized'}), 503
    
    config_data = {
        'server_config': server_config,
        'live_trading': order_placer.do_live_trades if order_placer else False,
        'best_strategy': server_config.get('best_strategy', {}) if server_config else {},
        'position': {
            'btc_balance': data_manager.balance_btc if data_manager else 0,
            'usd_balance': data_manager.balance_usd if data_manager else 0,
            'position': data_manager.position if data_manager else 0,
            'entry_price': data_manager.position_cost_basis / data_manager.position_size 
                          if data_manager and data_manager.position_size > 0 else 0
        } if data_manager else {}
    }
    
    # Extract key parameters for easy verification
    bs = config_data['best_strategy']
    config_data['key_parameters'] = {
        'strategy': bs.get('Strategy', 'Unknown'),
        'short_window': bs.get('Short_Window', 0),
        'long_window': bs.get('Long_Window', 0),
        'do_live_trades': bs.get('do_live_trades', False),
        'regime_switch_threshold': bs.get('regime_switch_threshold', 0),
        'signal_confirmation_bars': bs.get('signal_confirmation_bars', 0),
        'min_trade_gap_minutes': bs.get('min_trade_gap_minutes', 0)
    }
    
    return jsonify(config_data), 200

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get server logs"""
    try:
        # Get query parameters
        lines = request.args.get('lines', 100, type=int)
        level = request.args.get('level', 'ALL')  # ALL, ERROR, WARNING, INFO, DEBUG
        search = request.args.get('search', '')
        log_type = request.args.get('type', 'server')  # server, trading, diagnostic
        
        # Determine which log file to read
        if log_type == 'server':
            log_file = 'tdr_server.log'
        elif log_type == 'trading':
            log_file = 'crypto_shell.log'
        elif log_type == 'diagnostic':
            # Find most recent diagnostic log
            import glob
            diagnostic_files = sorted(glob.glob('diagnostics_*.json'), reverse=True)
            if diagnostic_files:
                # Return diagnostic data as JSON
                with open(diagnostic_files[0], 'r') as f:
                    diagnostic_data = json.load(f)
                return jsonify({
                    'type': 'diagnostic',
                    'file': diagnostic_files[0],
                    'data': diagnostic_data[-lines:] if isinstance(diagnostic_data, list) else diagnostic_data
                }), 200
            else:
                return jsonify({'error': 'No diagnostic logs found'}), 404
        else:
            return jsonify({'error': f'Unknown log type: {log_type}'}), 400
        
        # Read log file
        if not os.path.exists(log_file):
            return jsonify({'error': f'Log file not found: {log_file}'}), 404
        
        # Read last N lines
        import subprocess
        cmd = ['tail', '-n', str(lines), log_file]
        if search:
            # Add grep for search
            cmd = ['sh', '-c', f'tail -n {lines} {log_file} | grep -i "{search}"']
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        log_lines = result.stdout.split('\n') if result.stdout else []
        
        # Filter by level if specified
        if level != 'ALL':
            log_lines = [line for line in log_lines if level in line]
        
        return jsonify({
            'type': log_type,
            'file': log_file,
            'lines': log_lines,
            'total_lines': len(log_lines),
            'filter': {
                'level': level,
                'search': search,
                'requested_lines': lines
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/diagnostics', methods=['GET'])
def get_diagnostics():
    """Get diagnostic events"""
    try:
        # Get parameters
        event_type = request.args.get('event_type', 'ALL')
        count = request.args.get('count', 50, type=int)
        
        # Find diagnostic files
        import glob
        diagnostic_files = sorted(glob.glob('diagnostic_events_*.json'), reverse=True)
        
        if not diagnostic_files:
            return jsonify({'error': 'No diagnostic files found'}), 404
        
        all_events = []
        for file in diagnostic_files[:5]:  # Check last 5 files max
            try:
                with open(file, 'r') as f:
                    for line in f:
                        if line.strip():
                            event = json.loads(line)
                            if event_type == 'ALL' or event.get('event_type') == event_type:
                                all_events.append(event)
                if len(all_events) >= count:
                    break
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
        
        # Sort by timestamp and get most recent
        all_events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            'events': all_events[:count],
            'total_events': len(all_events),
            'event_type_filter': event_type,
            'files_checked': len(diagnostic_files)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting diagnostics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades', methods=['GET'])
def get_trades():
    """Get trade history"""
    try:
        limit = request.args.get('limit', 50, type=int)
        
        # Check if trades.json exists
        if not os.path.exists('trades.json'):
            return jsonify({'trades': [], 'message': 'No trades file found'}), 200
        
        with open('trades.json', 'r') as f:
            trades = json.load(f)
        
        # Sort by timestamp (most recent first)
        if isinstance(trades, list):
            trades.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            trades = trades[:limit]
        
        return jsonify({
            'trades': trades,
            'total_trades': len(trades),
            'limit': limit
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    """Gracefully shutdown the server"""
    try:
        logger.info("Shutdown requested")
        
        # Stop components
        stop_event.set()
        if websocket_thread:
            websocket_thread.join(timeout=5)
        
        # Shutdown Flask
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            return jsonify({'error': 'Not running with the Werkzeug Server'}), 500
        func()
        
        return jsonify({'message': 'Server shutting down...'}), 200
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
        return jsonify({'error': str(e)}), 500

def main():
    """Main server entry point"""
    parser = argparse.ArgumentParser(description='TDR Trading Server - Configuration from Client')
    parser.add_argument('--port', type=int, default=4000, help='Server port (default: 4000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger.info(f"Starting TDR Server on {args.host}:{args.port}")
    logger.info("Waiting for client to send configuration...")
    
    # Start Flask server
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    finally:
        stop_event.set()
        if websocket_thread:
            websocket_thread.join(timeout=5)
        logger.info("Server shutdown complete")

if __name__ == '__main__':
    main()