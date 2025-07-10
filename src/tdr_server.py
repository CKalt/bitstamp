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
history_loading_lock = threading.Lock()

def auto_load_history():
    """Automatically load history after initialization"""
    global server_config
    
    # Use lock to prevent duplicate loading
    with history_loading_lock:
        # Check if already loading or loaded
        if server_config.get('history_loading', False):
            logger.info("History already loading, skipping duplicate load")
            return
        if server_config.get('history_loaded', False):
            logger.info("History already loaded, skipping duplicate load")
            return
        
        # Set loading flag inside the lock to prevent race conditions
        server_config['history_loading'] = True
    
    try:
        server_config['history_progress'] = 0
        server_config['file_progress'] = 0
        server_config['history_status'] = 'Starting...'
        server_config['current_phase'] = 'starting'
        logger.info("Auto-loading historical data...")
        
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
            
            # Create a custom progress monitoring approach
            import sys
            from io import StringIO
            import threading
            
            # Set initial progress
            server_config['history_progress'] = 0
            server_config['file_progress'] = 0
            server_config['history_status'] = 'Phase 1: Reading file...'
            server_config['current_phase'] = 'reading_file'
            
            # Create a thread-safe stdout capture
            class ProgressCapture:
                def __init__(self, original_stdout):
                    self.original = original_stdout
                    self.buffer = []
                    self.last_progress = -1
                    self.reached_100 = False
                    
                def write(self, text):
                    self.original.write(text)  # Still write to console
                    self.buffer.append(text)
                    
                    # Check for progress updates in real-time
                    if 'Progress:' in text:
                        try:
                            percent = float(text.split(':')[1].split('%')[0].strip())
                            # Cap progress at 100% - percentages over 100 are confusing
                            percent = min(percent, 100.0)
                            
                            # Update file reading progress
                            server_config['file_progress'] = percent
                            
                            # Overall progress: file reading is 0-80% of total
                            overall_progress = percent * 0.8
                            server_config['history_progress'] = overall_progress
                            
                            # Update status with phase info
                            date_part = text.split('-', 1)[-1].strip() if '-' in text else ''
                            server_config['history_status'] = f"Phase 1: Reading file ({percent:.0f}%) - {date_part}"
                            
                            # Only log if progress changed by at least 0.1%
                            if abs(percent - self.last_progress) >= 0.1:
                                logger.info(f"File reading: {percent:.1f}% (Overall: {overall_progress:.0f}%)")
                                self.last_progress = percent
                            
                            # Track when file reading reaches 100%
                            if percent >= 100.0:
                                self.reached_100 = True
                                server_config['current_phase'] = 'creating_dataframe'
                                server_config['history_status'] = "Phase 2: Creating DataFrame..."
                                server_config['history_progress'] = 80  # File reading complete = 80% overall
                        except:
                            pass
                
                def flush(self):
                    self.original.flush()
            
            # Replace stdout with our progress capture
            old_stdout = sys.stdout
            sys.stdout = ProgressCapture(old_stdout)
            
            try:
                logger.info("Starting parse_log_file...")
                # Start a thread to provide periodic updates if parse_log_file is slow
                stop_updates = threading.Event()
                
                def progress_updater():
                    elapsed = 0
                    while not stop_updates.is_set():
                        if server_config.get('history_progress', 0) == 0:
                            # If still at 0%, show elapsed time
                            server_config['history_status'] = f'Loading historical data... ({elapsed}s elapsed)'
                        stop_updates.wait(1)  # Update every second
                        elapsed += 1
                
                update_thread = threading.Thread(target=progress_updater, daemon=True)
                update_thread.start()
                
                df = parse_log_file(log_file, start_date=start_date, end_date=end_date)
                
                stop_updates.set()
                update_thread.join(timeout=0.1)
                logger.info(f"parse_log_file completed with {len(df) if not df.empty else 0} records")
            finally:
                sys.stdout = old_stdout
            
            if not df.empty:
                # Phase 2: DataFrame processing (80-90%)
                server_config['current_phase'] = 'processing_dataframe'
                server_config['history_status'] = "Phase 2: Processing DataFrame..."
                server_config['history_progress'] = 85
                logger.info("Processing DataFrame...")
                
                # Need to process the dataframe like in main tdr.py
                df.rename(columns={'price': 'close'}, inplace=True)
                df['open'] = df['close']
                df['high'] = df['close']
                df['low'] = df['close']
                df['trades'] = 1
                if 'volume' not in df.columns:
                    df['volume'] = df.get('amount', 0.0)
                
                # Phase 3: Loading into data manager (90-100%)
                server_config['current_phase'] = 'loading_data'
                server_config['history_status'] = "Phase 3: Loading into data manager..."
                server_config['history_progress'] = 90
                logger.info("Loading data into manager...")
                
                data_manager.load_historical_data({'btcusd': df})
                logger.info(f"Loaded {len(df)} historical records")
                server_config['history_record_count'] = len(df)
        
        # Final completion
        server_config['history_loaded'] = True
        server_config['history_loading'] = False
        server_config['history_progress'] = 100.0
        server_config['file_progress'] = 100.0
        server_config['current_phase'] = 'complete'
        server_config['history_status'] = f"Complete - {server_config.get('history_record_count', 0)} records loaded"
        logger.info("Historical data loaded successfully")
        
        # Check if auto_resume is enabled
        if best_strategy.get('auto_resume', False):
            logger.info("Auto-resume is enabled, checking for saved position...")
            try:
                # Check if auto-trader is already running
                if shell and shell.auto_trader:
                    logger.info("Auto-trader is already running, skipping auto-resume")
                else:
                    # Load last saved position
                    resume_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resume-auto-trade.json')
                    if os.path.exists(resume_file):
                        with open(resume_file, 'r') as f:
                            resume_data = json.load(f)
                        
                        logger.info(f"Found saved position: {resume_data['position']} {resume_data['amount']} {resume_data['unit']} @ ${resume_data['entry_price']}")
                        
                        # Extract command arguments
                        parts = resume_data['command'].split()
                        if len(parts) >= 4 and parts[0] == 'resume_auto_trade':
                            resume_args = ' '.join(parts[1:])
                            logger.info(f"Executing auto-resume: {resume_args}")
                            
                            # Execute resume command
                            if shell:
                                shell.do_resume_auto_trade(resume_args)
                                logger.info("Auto-resume completed successfully")
                            else:
                                logger.error("Shell not available for auto-resume")
                        else:
                            logger.error(f"Invalid resume command format: {resume_data.get('command')}")
                    else:
                        logger.info("No saved position found for auto-resume")
            except Exception as e:
                logger.error(f"Error during auto-resume: {e}")
                import traceback
                traceback.print_exc()
            
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        server_config['history_loading'] = False
        server_config['history_loaded'] = False
        server_config['history_error'] = str(e)
        server_config['history_status'] = f"Error: {str(e)}"

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
        order_placer = OrderPlacer()
        
        # Initialize position from config if provided
        if 'initial_position' in config:
            pos = config['initial_position']
            data_manager.balance_btc = pos.get('btc_balance', 0)
            data_manager.balance_usd = pos.get('usd_balance', 10000)
            data_manager.position = pos.get('position', 0)
            data_manager.position_size = pos.get('position_size', 0)
            data_manager.position_cost_basis = pos.get('position_cost_basis', 0)
            
            # Set last_trade_price from position tracking
            if data_manager.position_size != 0:
                data_manager.last_trade_price = data_manager.position_cost_basis / abs(data_manager.position_size)
            else:
                # Fallback to Last_Trade_Price from best_strategy if available
                data_manager.last_trade_price = best_strategy.get('Last_Trade_Price', 0)
            
            # Validate against trades.json if available
            try:
                trades_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trades.json')
                if os.path.exists(trades_file):
                    with open(trades_file, 'r') as f:
                        trades = json.load(f)
                    
                    if trades:
                        # Find all trades for current position
                        position_trades = []
                        current_position = None
                        
                        for trade in reversed(trades):
                            if not current_position:
                                current_position = 'LONG' if trade['type'] == 'buy' else 'SHORT'
                                position_trades.append(trade)
                            elif (current_position == 'LONG' and trade['type'] == 'buy') or \
                                 (current_position == 'SHORT' and trade['type'] == 'sell'):
                                position_trades.append(trade)
                            else:
                                break
                        
                        if position_trades:
                            position_trades.reverse()
                            # Calculate actual entry price from trades
                            total_btc = sum(float(t['amount']) for t in position_trades)
                            total_cost = sum(float(t['amount']) * float(t['price']) for t in position_trades)
                            actual_entry_price = total_cost / total_btc if total_btc > 0 else 0
                            
                            if actual_entry_price > 0:
                                # Update position tracking with correct values
                                if current_position == 'LONG' and data_manager.position == 1:
                                    data_manager.position_size = total_btc
                                    data_manager.position_cost_basis = total_cost
                                    data_manager.last_trade_price = actual_entry_price
                                    logger.info(f"Validated LONG position from trades.json: {total_btc:.8f} BTC @ ${actual_entry_price:.2f}")
                                elif current_position == 'SHORT' and data_manager.position == -1:
                                    data_manager.position_size = -total_btc
                                    data_manager.position_cost_basis = total_cost
                                    data_manager.last_trade_price = actual_entry_price
                                    logger.info(f"Validated SHORT position from trades.json: {total_btc:.8f} BTC sold @ ${actual_entry_price:.2f}")
            except Exception as e:
                logger.warning(f"Could not validate position from trades.json: {e}")
            
            logger.info(f"Initialized position: BTC={data_manager.balance_btc}, USD={data_manager.balance_usd}, Entry=${data_manager.last_trade_price:.2f}")
        
        # Create shell instance
        shell = CryptoShell(
            data_manager=data_manager,
            order_placer=order_placer,
            logger=logger,
            verbose=verbose,
            live_trading=do_live_trades,
            stop_event=stop_event,
            max_trades_per_day=best_strategy.get('max_trades_per_day', 5)
        )
        
        # CRITICAL: Disable interactive mode for server
        shell.use_rawinput = False
        logger.info("Shell created in non-interactive mode (use_rawinput=False)")
        
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
        
        # Store best_strategy in server config
        server_config['best_strategy'] = best_strategy
        
        # Automatically start loading history after initialization
        threading.Thread(target=auto_load_history, daemon=True).start()
        
        return jsonify({
            'success': True,
            'message': 'Server initialized successfully',
            'config_summary': {
                'do_live_trades': do_live_trades,
                'strategy': best_strategy.get('Strategy', 'Unknown'),
                'websocket': config.get('enable_websocket', True),
                'historical_data_loaded': False,
                'history_loading': True,
                'auto_resume': best_strategy.get('auto_resume', False),
                'message': 'History loading will start automatically' + (' with auto-resume' if best_strategy.get('auto_resume', False) else '')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ping', methods=['GET'])
def ping():
    """Simple ping endpoint for connection testing"""
    return jsonify({'status': 'pong', 'timestamp': datetime.now().isoformat()}), 200

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
            'live_trading': shell.live_trading if shell else False,
            'history_loaded': server_config.get('history_loaded', False),
            'history_loading': server_config.get('history_loading', False),
            'history_progress': server_config.get('history_progress', 0),
            'file_progress': server_config.get('file_progress', 0),
            'current_phase': server_config.get('current_phase', 'not_started'),
            'history_status': server_config.get('history_status', ''),
            'auto_resume': server_config.get('best_strategy', {}).get('auto_resume', False)
        }
        
        if data_manager:
            # Get proper entry price from strategy if available
            if shell and shell.auto_trader and hasattr(shell.auto_trader, 'get_status'):
                strategy_status = shell.auto_trader.get_status()
                position_info = strategy_status.get('position_info', {})
                entry_price = position_info.get('entry_price', 0)
            else:
                # Fallback calculation
                entry_price = data_manager.position_cost_basis / abs(data_manager.position_size) if data_manager.position_size != 0 else 0
            
            status['position'] = {
                'btc_balance': data_manager.balance_btc,
                'usd_balance': data_manager.balance_usd,
                'position': data_manager.position,
                'position_size': data_manager.position_size,
                'entry_price': entry_price
            }
            status['last_price'] = data_manager.last_price.get('btcusd', 0) if isinstance(data_manager.last_price, dict) else data_manager.last_price
        
        if shell and shell.auto_trader:
            status['auto_trader'] = {
                'active': shell.auto_trader.running if hasattr(shell.auto_trader, 'running') else False,
                'strategy': type(shell.auto_trader).__name__,
                'trades_today': shell.auto_trader.trade_count_today if hasattr(shell.auto_trader, 'trade_count_today') else 0
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
            # Check if command requires history and block if not loaded
            history_required_commands = ['auto_trade', 'resume_auto_trade', 'buy', 'sell', 
                                       'limit_buy', 'limit_sell', 'strategy_diagnostics']
            
            if any(cmd in command for cmd in history_required_commands):
                if not server_config.get('history_loaded', False):
                    if server_config.get('history_loading', False):
                        # Special handling for resume_auto_trade - enable auto_resume instead of blocking
                        if 'resume_auto_trade' in command and server_config.get('best_strategy'):
                            if not server_config['best_strategy'].get('auto_resume', False):
                                server_config['best_strategy']['auto_resume'] = True
                                logger.info("Enabled auto_resume - will resume automatically when history loads")
                                return jsonify({
                                    'command': command,
                                    'success': True,
                                    'message': 'Auto-resume enabled. Trading will start automatically when history finishes loading.',
                                    'auto_resume': True,
                                    'history_loading': True,
                                    'history_progress': server_config.get('history_progress', 0)
                                }), 200
                            else:
                                return jsonify({
                                    'command': command,
                                    'success': True,
                                    'message': 'Auto-resume already enabled. Trading will start automatically when history finishes loading.',
                                    'auto_resume': True,
                                    'history_loading': True,
                                    'history_progress': server_config.get('history_progress', 0)
                                }), 200
                        
                        # For other commands, still block
                        return jsonify({
                            'command': command,
                            'success': False,
                            'error': 'Historical data is still loading. Please wait and check history_status.',
                            'history_loading': True
                        }), 400
                    else:
                        return jsonify({
                            'command': command,
                            'success': False,
                            'error': 'Historical data not loaded. Please run load_history first.',
                            'history_loaded': False
                        }), 400
            
            # Special debug logging for resume_auto_trade
            if 'resume_auto_trade' in command:
                logger.info(f"[RESUME_DEBUG] About to execute resume_auto_trade via shell.onecmd")
                logger.info(f"[RESUME_DEBUG] Shell object exists: {shell is not None}")
                logger.info(f"[RESUME_DEBUG] Command: '{command}'")
            
            with redirect_stdout(output_buffer):
                logger.info(f"[RESUME_DEBUG] Calling shell.onecmd('{command}')")
                shell.onecmd(command)
                logger.info(f"[RESUME_DEBUG] shell.onecmd completed")
            
            output = output_buffer.getvalue()
            
            if 'resume_auto_trade' in command:
                logger.info(f"[RESUME_DEBUG] Output captured: {len(output)} chars")
                logger.info(f"[RESUME_DEBUG] Output preview: {output[:200]}...")
            
            result = {
                'command': command,
                'output': output,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            # Add state updates for relevant commands
            if any(cmd in command for cmd in ['buy', 'sell', 'status', 'position', 'auto_trade']):
                # Get proper entry price from strategy if available
                if shell and shell.auto_trader and hasattr(shell.auto_trader, 'get_status'):
                    strategy_status = shell.auto_trader.get_status()
                    position_info = strategy_status.get('position_info', {})
                    entry_price = position_info.get('entry_price', 0)
                else:
                    # Fallback calculation
                    entry_price = data_manager.position_cost_basis / abs(data_manager.position_size) if data_manager.position_size != 0 else 0
                
                result['position'] = {
                    'btc_balance': data_manager.balance_btc,
                    'usd_balance': data_manager.balance_usd,
                    'position': data_manager.position,
                    'entry_price': entry_price
                }
            
            # Update global auto_trader reference if changed
            if shell.auto_trader:
                global auto_trader
                auto_trader = shell.auto_trader
                result['auto_trader'] = {
                    'active': shell.auto_trader.running if hasattr(shell.auto_trader, 'running') else False,
                    'strategy': type(shell.auto_trader).__name__
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
        'history_progress': server_config.get('history_progress', 0),
        'file_progress': server_config.get('file_progress', 0),
        'current_phase': server_config.get('current_phase', 'not_started'),
        'history_status': server_config.get('history_status', ''),
        'history_error': server_config.get('history_error', None),
        'record_count': server_config.get('history_record_count', 0)
    }), 200

@app.route('/api/fix_entry_price', methods=['POST'])
def fix_entry_price():
    """Recalculate and fix entry price from trades.json"""
    if not initialization_complete:
        return jsonify({'error': 'Server not initialized'}), 503
    
    try:
        # Load trades.json
        trades_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trades.json')
        if not os.path.exists(trades_file):
            return jsonify({'error': 'trades.json not found'}), 404
            
        with open(trades_file, 'r') as f:
            trades = json.load(f)
        
        if not trades:
            return jsonify({'error': 'No trades found'}), 404
        
        # Find all trades for current position
        position_trades = []
        current_position = None
        
        for trade in reversed(trades):
            if not current_position:
                current_position = 'LONG' if trade['type'] == 'buy' else 'SHORT'
                position_trades.append(trade)
            elif (current_position == 'LONG' and trade['type'] == 'buy') or \
                 (current_position == 'SHORT' and trade['type'] == 'sell'):
                position_trades.append(trade)
            else:
                break
        
        position_trades.reverse()
        
        # Calculate correct average entry price
        total_btc = sum(float(t['amount']) for t in position_trades)
        total_cost = sum(float(t['amount']) * float(t.get('order_result', t).get('price', t['price'])) for t in position_trades)
        avg_entry_price = total_cost / total_btc if total_btc > 0 else 0
        
        logger.info(f"Calculated correct entry price from {len(position_trades)} trades:")
        logger.info(f"  Total BTC: {total_btc:.8f}")
        logger.info(f"  Total Cost: ${total_cost:.2f}")
        logger.info(f"  Average Entry Price: ${avg_entry_price:.2f}")
        
        # Update data_manager position tracking
        if data_manager:
            if current_position == 'LONG':
                data_manager.position = 1
                data_manager.position_size = total_btc
                data_manager.position_cost_basis = total_cost
                data_manager.last_trade_price = avg_entry_price
                logger.info("Updated data_manager with correct LONG position")
            elif current_position == 'SHORT':
                data_manager.position = -1
                data_manager.position_size = -total_btc
                data_manager.position_cost_basis = total_cost
                data_manager.last_trade_price = avg_entry_price
                logger.info("Updated data_manager with correct SHORT position")
        
        # Update auto_trader if running
        if shell and shell.auto_trader:
            shell.auto_trader.position = data_manager.position
            shell.auto_trader.position_size = data_manager.position_size
            shell.auto_trader.position_cost_basis = data_manager.position_cost_basis
            shell.auto_trader.last_trade_price = avg_entry_price
            
            # Clear any theoretical trade
            if hasattr(shell.auto_trader, 'theoretical_trade'):
                shell.auto_trader.theoretical_trade = None
            
            logger.info("Updated auto_trader with correct position")
            
            # Force save resume state with correct values
            shell.auto_trader.save_resume_state()
        
        # Update best_strategy.json with correct Last_Trade_Price
        best_strategy_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_strategy.json')
        if os.path.exists(best_strategy_file):
            with open(best_strategy_file, 'r') as f:
                best_strategy = json.load(f)
            
            # Get the last actual trade price (not average)
            last_trade = position_trades[-1] if position_trades else None
            if last_trade:
                last_price = float(last_trade.get('order_result', last_trade).get('price', last_trade['price']))
                best_strategy['Last_Trade_Price'] = last_price
                best_strategy['Last_Trade_Timestamp'] = int(datetime.strptime(last_trade['timestamp'], '%Y-%m-%d %H:%M:%S').timestamp())
                
                with open(best_strategy_file, 'w') as f:
                    json.dump(best_strategy, f, indent=2)
                
                logger.info(f"Updated best_strategy.json with Last_Trade_Price: ${last_price:.2f}")
        
        # Update server config
        if server_config and 'best_strategy' in server_config:
            server_config['best_strategy']['Last_Trade_Price'] = avg_entry_price
        
        return jsonify({
            'success': True,
            'current_position': current_position,
            'total_btc': total_btc,
            'average_entry_price': avg_entry_price,
            'trades_count': len(position_trades),
            'message': f'Fixed entry price to ${avg_entry_price:.2f} from {len(position_trades)} trades'
        }), 200
        
    except Exception as e:
        logger.error(f"Error fixing entry price: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/best_strategy', methods=['GET', 'POST'])
def handle_best_strategy():
    """Get or update the server's best_strategy.json"""
    if not initialization_complete:
        return jsonify({'error': 'Server not initialized'}), 503
    
    if request.method == 'GET':
        return get_best_strategy()
    else:  # POST
        return update_best_strategy()

def get_best_strategy():
    """Get the server's best_strategy.json"""
    try:
        best_strategy_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_strategy.json')
        if os.path.exists(best_strategy_file):
            with open(best_strategy_file, 'r') as f:
                strategy = json.load(f)
            return jsonify({
                'success': True,
                'best_strategy': strategy,
                'source': 'server'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'best_strategy.json not found on server'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def update_best_strategy():
    """Update the server's best_strategy.json from client"""
    try:
        new_strategy = request.json
        if not new_strategy:
            return jsonify({'error': 'No strategy data provided'}), 400
        
        best_strategy_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_strategy.json')
        
        # Load current server strategy to preserve server-only fields
        current_strategy = {}
        if os.path.exists(best_strategy_file):
            with open(best_strategy_file, 'r') as f:
                current_strategy = json.load(f)
        
        # Preserve server-managed fields
        server_fields = ['Last_Trade_Price', 'Last_Trade_Timestamp', 'auto_resume', 'max_trades_per_day']
        for field in server_fields:
            if field in current_strategy:
                new_strategy[field] = current_strategy[field]
        
        # Save updated strategy
        with open(best_strategy_file, 'w') as f:
            json.dump(new_strategy, f, indent=2)
        
        # Update server config
        if server_config:
            server_config['best_strategy'] = new_strategy
        
        logger.info("Updated best_strategy.json from client")
        
        return jsonify({
            'success': True,
            'message': 'Strategy updated successfully',
            'preserved_fields': {field: new_strategy.get(field) for field in server_fields if field in new_strategy}
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating strategy: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get active server configuration for verification"""
    if not initialization_complete:
        return jsonify({'error': 'Server not initialized'}), 503
    
    # Get proper entry price from strategy if available
    if data_manager and shell and shell.auto_trader and hasattr(shell.auto_trader, 'get_status'):
        strategy_status = shell.auto_trader.get_status()
        position_info = strategy_status.get('position_info', {})
        entry_price = position_info.get('entry_price', 0)
    elif data_manager and data_manager.position_size != 0:
        # Fallback calculation
        entry_price = data_manager.position_cost_basis / abs(data_manager.position_size)
    else:
        entry_price = 0
    
    config_data = {
        'server_config': server_config,
        'live_trading': order_placer.do_live_trades if order_placer else False,
        'best_strategy': server_config.get('best_strategy', {}) if server_config else {},
        'position': {
            'btc_balance': data_manager.balance_btc if data_manager else 0,
            'usd_balance': data_manager.balance_usd if data_manager else 0,
            'position': data_manager.position if data_manager else 0,
            'entry_price': entry_price
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

@app.route('/api/position_history', methods=['GET'])
def get_position_history():
    """Get position history and last resume state"""
    try:
        if not initialization_complete:
            return jsonify({'error': 'Server not initialized'}), 503
            
        response = {
            'timestamp': datetime.now().isoformat(),
            'current_position': None,
            'last_position': None,
            'history': []
        }
        
        # Get current position if auto-trader is running
        if shell and shell.auto_trader:
            status = shell.auto_trader.get_status()
            position_info = status.get('position_info', {})
            response['current_position'] = {
                'active': True,
                'position': position_info.get('position', 'unknown'),
                'amount': position_info.get('amount', 0),
                'entry_price': position_info.get('entry_price', 0),
                'current_price': position_info.get('current_price', 0),
                'unrealized_pnl': position_info.get('unrealized_pnl', 0)
            }
        
        # Read last saved position from resume-auto-trade.json
        import os
        resume_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resume-auto-trade.json')
        if os.path.exists(resume_file):
            try:
                with open(resume_file, 'r') as f:
                    response['last_position'] = json.load(f)
            except Exception as e:
                logger.error(f"Error reading resume file: {e}")
        
        # Read position history
        history_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'position-history.json')
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    # Return last 10 entries
                    response['history'] = history[-10:] if len(history) > 10 else history
            except Exception as e:
                logger.error(f"Error reading history file: {e}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error getting position history: {e}")
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