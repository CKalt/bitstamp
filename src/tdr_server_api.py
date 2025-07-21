#!/usr/bin/env python3
"""
Enhanced API endpoints for direct monitoring and control
Allows Claude to use curl for faster, more direct communication
"""
from flask import Blueprint, jsonify, request
import json
from datetime import datetime

# Create API blueprint
api_bp = Blueprint('api', __name__)

# These will be set by the main server
shell = None
data_manager = None
logger = None

def init_api(shell_instance, data_manager_instance, logger_instance):
    """Initialize API with server components"""
    global shell, data_manager, logger
    shell = shell_instance
    data_manager = data_manager_instance
    logger = logger_instance

@api_bp.route('/api/signal_status', methods=['GET'])
def get_signal_status():
    """Get comprehensive signal status for monitoring"""
    try:
        if not shell or not shell.auto_trader:
            return jsonify({'error': 'Auto trader not running'}), 503
            
        from tdr_core.signal_monitor import SignalMonitor
        monitor = SignalMonitor(data_manager, shell.auto_trader)
        status = monitor.get_current_signal_status()
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting signal status: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/api/signal_history', methods=['GET'])
def get_signal_history():
    """Get signal history"""
    try:
        hours = request.args.get('hours', 1, type=float)
        
        if not shell or not shell.auto_trader:
            return jsonify({'error': 'Auto trader not running'}), 503
            
        from tdr_core.signal_monitor import SignalMonitor
        monitor = SignalMonitor(data_manager, shell.auto_trader)
        history = monitor.get_signal_history(hours=hours)
        
        # Convert datetime objects to strings
        for h in history:
            h['timestamp'] = h['timestamp'].isoformat()
            
        return jsonify({
            'hours': hours,
            'count': len(history),
            'history': history
        })
        
    except Exception as e:
        logger.error(f"Error getting signal history: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/api/quick_status', methods=['GET'])
def get_quick_status():
    """Get quick status - minimal info for frequent polling"""
    try:
        if not shell or not shell.auto_trader:
            return jsonify({'error': 'Auto trader not running'}), 503
            
        current_price = data_manager.get_current_price('btcusd')
        
        # Get MA values
        df = data_manager.get_dataframe('btcusd')
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            short_ma = latest.get(f'SMA_{shell.auto_trader.short_window}', 0)
            long_ma = latest.get(f'SMA_{shell.auto_trader.long_window}', 0)
            ma_diff_pct = ((short_ma - long_ma) / long_ma * 100) if long_ma > 0 else 0
        else:
            short_ma = long_ma = ma_diff_pct = 0
            
        # Calculate P&L
        if shell.auto_trader.position == 1:  # LONG
            pnl = (current_price - shell.auto_trader.last_trade_price) * shell.auto_trader.position_size
        elif shell.auto_trader.position == -1:  # SHORT
            pnl = (shell.auto_trader.last_trade_price - current_price) * abs(shell.auto_trader.position_size)
        else:
            pnl = 0
            
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'price': current_price,
            'position': 'LONG' if shell.auto_trader.position == 1 else 'SHORT' if shell.auto_trader.position == -1 else 'FLAT',
            'entry_price': shell.auto_trader.last_trade_price,
            'pnl': round(pnl, 2),
            'ma_diff_pct': round(ma_diff_pct, 2),
            'signal_matches': (short_ma > long_ma and shell.auto_trader.position == 1) or (short_ma < long_ma and shell.auto_trader.position == -1)
        })
        
    except Exception as e:
        logger.error(f"Error getting quick status: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/api/execute_command', methods=['POST'])
def execute_command():
    """Execute any shell command via HTTP POST"""
    try:
        data = request.json
        command = data.get('command', '')
        source = data.get('source', 'api')
        
        if not command:
            return jsonify({'error': 'No command provided'}), 400
            
        if not shell:
            return jsonify({'error': 'Shell not initialized'}), 503
            
        logger.info(f"[API-CMD] Source: {source} | Command: {command}")
        
        # Capture output
        import io
        from contextlib import redirect_stdout
        
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            shell.onecmd(command)
            
        output = output_buffer.getvalue()
        
        # For some commands, we want to return structured data
        if command.startswith('status'):
            # Also include position data
            status = shell.auto_trader.get_status() if shell.auto_trader else {}
            return jsonify({
                'success': True,
                'output': output,
                'data': status,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': True,
                'output': output,
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/api/missed_signals', methods=['GET'])
def check_missed_signals():
    """Check for potentially missed trading signals"""
    try:
        if not shell or not shell.auto_trader:
            return jsonify({'error': 'Auto trader not running'}), 503
            
        # Simple check: if signal doesn't match position for multiple bars
        df = data_manager.get_dataframe('btcusd')
        if df is None or df.empty:
            return jsonify({'error': 'No data available'}), 503
            
        # Check last 5 bars
        missed_bars = 0
        for i in range(1, min(6, len(df))):
            row = df.iloc[-i]
            short_ma = row.get(f'SMA_{shell.auto_trader.short_window}', 0)
            long_ma = row.get(f'SMA_{shell.auto_trader.long_window}', 0)
            
            signal = 1 if short_ma > long_ma else -1
            if signal != shell.auto_trader.position:
                missed_bars += 1
                
        return jsonify({
            'missed_bars': missed_bars,
            'alert': missed_bars >= 3,
            'message': f"Signal has been opposite to position for {missed_bars} bars" if missed_bars > 0 else "No missed signals"
        })
        
    except Exception as e:
        logger.error(f"Error checking missed signals: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'auto_trader': shell.auto_trader is not None if shell else False,
        'data_available': data_manager is not None
    })