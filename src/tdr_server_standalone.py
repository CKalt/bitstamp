#!/usr/bin/env python3
"""
Standalone Server Initialization Module
Loads configuration from server-side files instead of waiting for client
"""
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger('TDRServer')

def load_server_config():
    """Load configuration from server-side files"""
    config = {
        'best_strategy': {},
        'server_config': {},
        'auto_resume': True,  # Always enable auto-resume
        'verbose': True
    }
    
    # Load best_strategy.json
    best_strategy_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_strategy.json')
    if os.path.exists(best_strategy_file):
        logger.info(f"Loading best_strategy.json from {best_strategy_file}")
        with open(best_strategy_file, 'r') as f:
            config['best_strategy'] = json.load(f)
        logger.info(f"Loaded strategy: {config['best_strategy'].get('Strategy')} "
                   f"with windows {config['best_strategy'].get('Short_Window')}/{config['best_strategy'].get('Long_Window')}")
    else:
        logger.warning("best_strategy.json not found, using defaults")
        # Default strategy configuration
        config['best_strategy'] = {
            "Strategy": "AdaptiveMultiStrategy",
            "Short_Window": 10,
            "Long_Window": 46,
            "do_live_trades": True,
            "start_window_days_back": 30,
            "end_window_days_back": 0
        }
    
    # Check for server_config.json
    server_config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'server_config.json')
    if os.path.exists(server_config_file):
        logger.info(f"Loading server_config.json from {server_config_file}")
        with open(server_config_file, 'r') as f:
            server_specific = json.load(f)
            config.update(server_specific)
    
    # Ensure required fields
    config['best_strategy']['do_live_trades'] = config['best_strategy'].get('do_live_trades', True)
    config['best_strategy']['start_window_days_back'] = config['best_strategy'].get('start_window_days_back', 30)
    config['best_strategy']['end_window_days_back'] = config['best_strategy'].get('end_window_days_back', 0)
    
    return config

def initialize_server_standalone():
    """Initialize server using local configuration files"""
    logger.info("Starting standalone server initialization...")
    
    # Load configuration from local files
    config = load_server_config()
    
    # Check if resume file exists
    import os
    import json
    resume_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resume-auto-trade.json')
    has_resume_file = os.path.exists(resume_file)
    
    if has_resume_file:
        logger.info(f"Found resume file at {resume_file}")
        with open(resume_file, 'r') as f:
            resume_data = json.load(f)
            logger.info(f"Resume position: {resume_data['position']} {resume_data['amount']} {resume_data.get('unit', 'btc')} @ ${resume_data['entry_price']}")
        
        # Validate against trades.json
        trades_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trades.json')
        if os.path.exists(trades_file):
            logger.info("Validating resume position against trades.json...")
            with open(trades_file, 'r') as f:
                trades_data = json.load(f)
            
            # Validate against most recent trades only (last 1-3 BUYs or last SELL)
            trades = trades_data.get('trades', [])
            if trades:
                last_trade = trades[-1]
                
                if last_trade['type'] == 'SELL':
                    # Validate against last SELL
                    if resume_data['position'] != 'SHORT':
                        logger.error(f"POSITION MISMATCH: Resume shows {resume_data['position']} but last trade was SELL")
                        raise ValueError("Position mismatch between resume file and trades.json")
                
                elif last_trade['type'] == 'BUY':
                    # Find last 1-3 consecutive BUYs
                    consecutive_buys = []
                    for i in range(len(trades) - 1, -1, -1):
                        if trades[i]['type'] == 'BUY' and len(consecutive_buys) < 3:
                            consecutive_buys.insert(0, trades[i])
                        elif trades[i]['type'] != 'BUY':
                            break
                    
                    total_btc = sum(t['amount'] for t in consecutive_buys)
                    
                    if resume_data['position'] != 'LONG':
                        logger.error(f"POSITION MISMATCH: Resume shows {resume_data['position']} but last trades were BUYs")
                        raise ValueError("Position mismatch between resume file and trades.json")
                    
                    # Allow 1% tolerance for multiple BUY trades
                    if abs(total_btc - resume_data['amount']) / resume_data['amount'] > 0.01:
                        logger.error(f"AMOUNT MISMATCH: Resume shows {resume_data['amount']} BTC but last {len(consecutive_buys)} BUYs total {total_btc:.8f} BTC")
                        raise ValueError("Position mismatch between resume file and trades.json")
                    
                    logger.info(f"Position validated: {len(consecutive_buys)} BUY trades totaling {total_btc:.8f} BTC ≈ resume {resume_data['amount']} BTC")
    
    # Create initialization payload
    init_payload = {
        'best_strategy': config['best_strategy'],
        'verbose': config.get('verbose', True),
        'test_mode': False,  # Always use live mode for server
        'auto_resume': True  # Force auto-resume to load position
    }
    
    # Force auto_resume in best_strategy
    init_payload['best_strategy']['auto_resume'] = True
    
    logger.info("Server configuration loaded:")
    logger.info(f"  Strategy: {config['best_strategy'].get('Strategy')}")
    logger.info(f"  Live Trading: {config['best_strategy'].get('do_live_trades')}")
    logger.info(f"  Auto Resume: True (forced)")
    logger.info(f"  Resume File: {'FOUND' if has_resume_file else 'NOT FOUND'}")
    
    return init_payload