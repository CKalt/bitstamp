"""
Enhanced monitoring API endpoints for TDR Server
"""
import os
import json
from flask import Blueprint, jsonify, request
import logging
from datetime import datetime, timedelta

logger = logging.getLogger('TDRServer')
monitoring_api = Blueprint('monitoring_api', __name__)

@monitoring_api.route('/api/status/detailed', methods=['GET'])
def get_detailed_status():
    """Get comprehensive status including position, P&L, and strategy details"""
    try:
        from tdr_server import shell, data_manager, auto_trader
        
        if not shell:
            return jsonify({'error': 'Server not initialized'}), 503
        
        # Execute status long command
        import io
        from contextlib import redirect_stdout
        
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            shell.onecmd('status long')
        
        status_output = output_buffer.getvalue()
        
        # Get additional details
        result = {
            'timestamp': datetime.now().isoformat(),
            'status_long': status_output,
            'position': {},
            'strategy': {},
            'market': {}
        }
        
        # Position details
        if auto_trader:
            status = auto_trader.get_status()
            position_info = status.get('position_info', {})
            
            result['position'] = {
                'type': 'LONG' if auto_trader.position == 1 else 'SHORT' if auto_trader.position == -1 else 'FLAT',
                'size_btc': auto_trader.position_size,
                'entry_price': position_info.get('entry_price', 0),
                'current_price': data_manager.get_last_price('btcusd') if data_manager else 0,
                'unrealized_pnl': position_info.get('unrealized_pnl', 0),
                'position_value': position_info.get('position_size_usd', 0),
                'cost_basis': auto_trader.position_cost_basis
            }
            
            # Strategy details
            if hasattr(auto_trader, 'current_strategy'):
                result['strategy'] = {
                    'type': auto_trader.current_strategy,
                    'confidence': getattr(auto_trader, 'last_confidence', 0),
                    'regime': getattr(auto_trader, 'market_regime', 'UNKNOWN')
                }
        
        # Market data
        if data_manager:
            result['market'] = {
                'last_price': data_manager.get_last_price('btcusd'),
                'websocket_connected': hasattr(data_manager, 'websocket_connected') and data_manager.websocket_connected
            }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in detailed status: {e}")
        return jsonify({'error': str(e)}), 500

@monitoring_api.route('/api/trades/history', methods=['GET'])
def get_trade_history():
    """Get trade history from trades.json"""
    try:
        trades_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trades.json')
        
        if not os.path.exists(trades_file):
            return jsonify({
                'trades': [],
                'message': 'No trades.json file found'
            }), 200
        
        with open(trades_file, 'r') as f:
            trades_data = json.load(f)
        
        # Handle both array and object formats
        if isinstance(trades_data, dict) and 'trades' in trades_data:
            trades = trades_data['trades']
        elif isinstance(trades_data, list):
            trades = trades_data
        else:
            trades = []
        
        # Get query parameters
        limit = request.args.get('limit', 50, type=int)
        days = request.args.get('days', 0, type=int)
        
        # Filter by days if specified
        if days > 0:
            cutoff = datetime.now() - timedelta(days=days)
            filtered_trades = []
            for trade in trades:
                try:
                    trade_time = datetime.fromisoformat(trade.get('timestamp', '').replace('Z', '+00:00'))
                    if trade_time >= cutoff:
                        filtered_trades.append(trade)
                except:
                    pass
            trades = filtered_trades
        
        # Limit results
        trades = trades[-limit:]
        
        # Calculate summary
        total_buys = sum(1 for t in trades if t.get('type') == 'BUY')
        total_sells = sum(1 for t in trades if t.get('type') == 'SELL')
        
        return jsonify({
            'trades': trades,
            'count': len(trades),
            'summary': {
                'total_buys': total_buys,
                'total_sells': total_sells,
                'last_trade': trades[-1] if trades else None
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        return jsonify({'error': str(e)}), 500

@monitoring_api.route('/api/strategy/config', methods=['GET'])
def get_strategy_config():
    """Get current strategy configuration from best_strategy.json"""
    try:
        strategy_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_strategy.json')
        
        if not os.path.exists(strategy_file):
            return jsonify({
                'error': 'No best_strategy.json file found'
            }), 404
        
        with open(strategy_file, 'r') as f:
            strategy_config = json.load(f)
        
        # Add runtime information if available
        from tdr_server import auto_trader
        if auto_trader:
            strategy_config['runtime'] = {
                'active': True,
                'current_strategy': getattr(auto_trader, 'current_strategy', 'UNKNOWN'),
                'trades_executed': auto_trader.trades_executed
            }
        
        return jsonify(strategy_config), 200
        
    except Exception as e:
        logger.error(f"Error getting strategy config: {e}")
        return jsonify({'error': str(e)}), 500

@monitoring_api.route('/api/strategy/flip_distance', methods=['GET'])
def get_flip_distance():
    """Calculate how close we are to flipping positions"""
    try:
        from tdr_server import auto_trader, data_manager
        
        if not auto_trader or not data_manager:
            return jsonify({'error': 'Auto trader not running'}), 503
        
        current_price = data_manager.get_last_price('btcusd')
        
        result = {
            'current_price': current_price,
            'position': 'LONG' if auto_trader.position == 1 else 'SHORT' if auto_trader.position == -1 else 'FLAT',
            'flip_analysis': {}
        }
        
        # Get MA values if using MA strategy
        if hasattr(auto_trader, 'short_ma') and hasattr(auto_trader, 'long_ma'):
            df = data_manager.get_dataframe('btcusd')
            if df is not None and len(df) > 0:
                current_short_ma = auto_trader.short_ma[-1] if len(auto_trader.short_ma) > 0 else 0
                current_long_ma = auto_trader.long_ma[-1] if len(auto_trader.long_ma) > 0 else 0
                
                if current_short_ma > 0 and current_long_ma > 0:
                    ma_diff = current_short_ma - current_long_ma
                    ma_diff_pct = (ma_diff / current_long_ma) * 100
                    
                    result['flip_analysis']['ma_crossover'] = {
                        'short_ma': current_short_ma,
                        'long_ma': current_long_ma,
                        'difference': ma_diff,
                        'difference_pct': ma_diff_pct,
                        'current_signal': 'LONG' if ma_diff > 0 else 'SHORT'
                    }
                    
                    # Calculate price needed for flip
                    if auto_trader.position == 1:  # LONG position
                        # Need SHORT signal (short MA < long MA)
                        price_for_flip = current_long_ma
                        distance = current_price - price_for_flip
                        distance_pct = (distance / current_price) * 100
                        result['flip_analysis']['ma_crossover']['flip_scenario'] = {
                            'need': 'SHORT signal (short MA < long MA)',
                            'price_for_flip': price_for_flip,
                            'distance': distance,
                            'distance_pct': distance_pct
                        }
                    else:  # SHORT position
                        # Need LONG signal (short MA > long MA)
                        price_for_flip = current_long_ma
                        distance = price_for_flip - current_price
                        distance_pct = (distance / current_price) * 100
                        result['flip_analysis']['ma_crossover']['flip_scenario'] = {
                            'need': 'LONG signal (short MA > long MA)',
                            'price_for_flip': price_for_flip,
                            'distance': distance,
                            'distance_pct': distance_pct
                        }
        
        # Adaptive strategy regime info
        if hasattr(auto_trader, 'market_regime'):
            result['flip_analysis']['adaptive'] = {
                'current_regime': auto_trader.market_regime,
                'confidence': getattr(auto_trader, 'last_confidence', 0),
                'strategy': auto_trader.current_strategy,
                'thresholds': {
                    'switch_to_ranging': '60% confidence',
                    'switch_to_trending': '60% confidence'
                }
            }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error calculating flip distance: {e}")
        return jsonify({'error': str(e)}), 500

@monitoring_api.route('/api/resume/status', methods=['GET'])
def get_resume_status():
    """Get auto-resume status and configuration"""
    try:
        resume_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resume-auto-trade.json')
        
        result = {
            'resume_file_exists': os.path.exists(resume_file),
            'auto_resume_enabled': False,
            'resume_data': None
        }
        
        if os.path.exists(resume_file):
            with open(resume_file, 'r') as f:
                result['resume_data'] = json.load(f)
        
        # Check if auto-resume is enabled in strategy
        strategy_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_strategy.json')
        if os.path.exists(strategy_file):
            with open(strategy_file, 'r') as f:
                strategy_config = json.load(f)
                result['auto_resume_enabled'] = strategy_config.get('auto_resume', False)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error getting resume status: {e}")
        return jsonify({'error': str(e)}), 500

@monitoring_api.route('/api/position/sync', methods=['GET'])
def check_position_sync():
    """Check if position is in sync between different components"""
    try:
        from tdr_server import auto_trader, data_manager, shell
        
        result = {
            'in_sync': True,
            'discrepancies': [],
            'positions': {}
        }
        
        # Get position from auto_trader
        if auto_trader:
            result['positions']['auto_trader'] = {
                'position': auto_trader.position,
                'size': auto_trader.position_size,
                'cost_basis': auto_trader.position_cost_basis
            }
        
        # Get position from data_manager
        if data_manager and hasattr(data_manager, 'position'):
            result['positions']['data_manager'] = {
                'position': data_manager.position,
                'size': getattr(data_manager, 'position_size', 0),
                'cost_basis': getattr(data_manager, 'position_cost_basis', 0)
            }
        
        # Check resume file
        resume_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resume-auto-trade.json')
        if os.path.exists(resume_file):
            with open(resume_file, 'r') as f:
                resume_data = json.load(f)
                result['positions']['resume_file'] = {
                    'position': 1 if resume_data.get('position') == 'LONG' else -1 if resume_data.get('position') == 'SHORT' else 0,
                    'size': resume_data.get('amount', 0),
                    'entry_price': resume_data.get('entry_price', 0)
                }
        
        # Check for discrepancies
        if auto_trader and data_manager:
            if auto_trader.position != data_manager.position:
                result['in_sync'] = False
                result['discrepancies'].append('Position direction mismatch')
            
            if abs(auto_trader.position_size - getattr(data_manager, 'position_size', 0)) > 0.00001:
                result['in_sync'] = False
                result['discrepancies'].append('Position size mismatch')
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error checking position sync: {e}")
        return jsonify({'error': str(e)}), 500