"""
Log API endpoints for TDR Server
"""
import os
import subprocess
from flask import Blueprint, jsonify, request
import logging

logger = logging.getLogger('TDRServer')
log_api = Blueprint('log_api', __name__)

# Get log file path
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'tdr_server.log')

@log_api.route('/api/logs/tail', methods=['GET'])
def tail_logs():
    """Tail the last N lines of the log file"""
    try:
        lines = request.args.get('lines', 50, type=int)
        lines = min(lines, 1000)  # Cap at 1000 lines
        
        if not os.path.exists(LOG_FILE):
            return jsonify({
                'error': 'Log file not found',
                'log_file': LOG_FILE
            }), 404
        
        # Use tail command for efficiency
        result = subprocess.run(
            ['tail', '-n', str(lines), LOG_FILE],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return jsonify({'error': 'Failed to read log file'}), 500
        
        log_lines = result.stdout.strip().split('\n')
        
        return jsonify({
            'lines': log_lines,
            'count': len(log_lines),
            'log_file': LOG_FILE
        }), 200
        
    except Exception as e:
        logger.error(f"Error tailing logs: {e}")
        return jsonify({'error': str(e)}), 500

@log_api.route('/api/logs/grep', methods=['GET'])
def grep_logs():
    """Search logs for a pattern"""
    try:
        pattern = request.args.get('pattern', '')
        lines = request.args.get('lines', 100, type=int)
        lines = min(lines, 1000)  # Cap at 1000 lines
        
        if not pattern:
            return jsonify({'error': 'No search pattern provided'}), 400
            
        if not os.path.exists(LOG_FILE):
            return jsonify({
                'error': 'Log file not found',
                'log_file': LOG_FILE
            }), 404
        
        # Use grep for pattern matching
        result = subprocess.run(
            ['grep', '-i', pattern, LOG_FILE, '|', 'tail', '-n', str(lines)],
            shell=True,
            capture_output=True,
            text=True
        )
        
        log_lines = result.stdout.strip().split('\n') if result.stdout else []
        
        return jsonify({
            'pattern': pattern,
            'lines': log_lines,
            'count': len(log_lines),
            'log_file': LOG_FILE
        }), 200
        
    except Exception as e:
        logger.error(f"Error searching logs: {e}")
        return jsonify({'error': str(e)}), 500

@log_api.route('/api/logs/stream', methods=['GET'])
def stream_logs():
    """Get recent log entries with optional filtering"""
    try:
        minutes = request.args.get('minutes', 5, type=int)
        level = request.args.get('level', 'INFO')  # INFO, WARNING, ERROR
        
        if not os.path.exists(LOG_FILE):
            return jsonify({
                'error': 'Log file not found', 
                'log_file': LOG_FILE
            }), 404
        
        # Read last N minutes of logs
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_logs = []
        with open(LOG_FILE, 'r') as f:
            # Read from end of file
            lines = f.readlines()[-1000:]  # Last 1000 lines max
            
            for line in lines:
                # Parse timestamp from log line
                try:
                    # Format: 2025-07-21 15:46:22,717 - [TDRServer] - INFO - ...
                    if ' - ' in line and len(line) > 23:
                        timestamp_str = line[:23]
                        log_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                        
                        if log_time >= cutoff_time:
                            # Filter by level if specified
                            if level == 'ALL' or f' - {level} - ' in line:
                                recent_logs.append({
                                    'timestamp': timestamp_str,
                                    'level': level if f' - {level} - ' in line else 'UNKNOWN',
                                    'message': line.strip()
                                })
                except:
                    pass  # Skip malformed lines
        
        return jsonify({
            'logs': recent_logs,
            'count': len(recent_logs),
            'minutes': minutes,
            'level': level
        }), 200
        
    except Exception as e:
        logger.error(f"Error streaming logs: {e}")
        return jsonify({'error': str(e)}), 500

@log_api.route('/api/logs/errors', methods=['GET'])
def get_errors():
    """Get recent errors and warnings"""
    try:
        lines = request.args.get('lines', 50, type=int)
        
        if not os.path.exists(LOG_FILE):
            return jsonify({
                'error': 'Log file not found',
                'log_file': LOG_FILE
            }), 404
        
        # Find ERROR and WARNING lines
        result = subprocess.run(
            f'grep -E "ERROR|WARNING" {LOG_FILE} | tail -n {lines}',
            shell=True,
            capture_output=True,
            text=True
        )
        
        error_lines = result.stdout.strip().split('\n') if result.stdout else []
        
        # Parse into structured format
        errors = []
        for line in error_lines:
            if line:
                level = 'ERROR' if 'ERROR' in line else 'WARNING'
                errors.append({
                    'level': level,
                    'message': line.strip()
                })
        
        return jsonify({
            'errors': errors,
            'count': len(errors)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting errors: {e}")
        return jsonify({'error': str(e)}), 500