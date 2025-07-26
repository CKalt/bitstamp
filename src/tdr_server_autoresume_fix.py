# Auto-resume fix for tdr_server.py
# This patch respects the auto_resume config setting instead of forcing it to True

def apply_autoresume_fix():
    """
    Fix for auto_resume being forced to True.
    
    The bug: When resume_auto_trade command is sent while history is loading,
    the server forces auto_resume = True regardless of config.
    
    The fix: Check and respect the config setting.
    """
    
    # Line 593-594 in tdr_server.py should be changed from:
    # if not server_config['best_strategy'].get('auto_resume', False):
    #     server_config['best_strategy']['auto_resume'] = True
    
    # To:
    # if not server_config['best_strategy'].get('auto_resume', False):
    #     # Don't force auto_resume if it's explicitly set to False in config
    #     logger.info("Auto-resume is disabled in config. Queuing resume command for manual execution after history loads.")
    #     server_config['pending_resume_command'] = command
    #     return jsonify({
    #         'command': command,
    #         'success': True,
    #         'message': 'Resume command queued. Execute manually after history loads.',
    #         'auto_resume': False,
    #         'history_loading': True
    #     }), 200
    
    pass

# Additional fix needed in the auto_resume_trading function (around line 225)
# Should check the config setting:
def auto_resume_trading_fix():
    """
    The auto_resume_trading function should respect the config setting.
    """
    
    # Add this check at the beginning of auto_resume_trading:
    # if not server_config.get('best_strategy', {}).get('auto_resume', False):
    #     logger.info("Auto-resume is disabled in configuration")
    #     return
    
    pass