#!/usr/bin/env python3
"""
Update best_strategy.json to ensure it has the correct strategy type configuration.
This ensures consistency with backtesting by using pure MA strategy.
"""

import json
import os
import shutil
from datetime import datetime

def update_strategy_config():
    """Update best_strategy.json to ensure pure MA strategy is configured"""
    
    config_file = 'best_strategy.json'
    
    # Backup existing file
    if os.path.exists(config_file):
        backup_file = f'best_strategy.json.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        shutil.copy(config_file, backup_file)
        print(f"✅ Backed up existing config to {backup_file}")
    
    # Load current config
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ {config_file} not found!")
        return
    
    print("\n📊 Current Configuration:")
    print(f"   Strategy: {config.get('Strategy', 'Not set')}")
    print(f"   strategy_type: {config.get('strategy_type', 'Not set')}")
    print(f"   enable_adaptive_strategy: {config.get('enable_adaptive_strategy', 'Not set')}")
    
    # Update to ensure pure MA strategy
    updates_made = False
    
    # Ensure strategy_type is set to MA
    if config.get('strategy_type') != 'MA':
        config['strategy_type'] = 'MA'
        updates_made = True
        print("\n✅ Set strategy_type = 'MA'")
    
    # Ensure adaptive strategy is disabled
    if config.get('enable_adaptive_strategy', True) != False:
        config['enable_adaptive_strategy'] = False
        updates_made = True
        print("✅ Set enable_adaptive_strategy = False")
    
    # Ensure other critical parameters for pure MA
    if config.get('Strategy') != 'MA':
        config['Strategy'] = 'MA'
        updates_made = True
        print("✅ Set Strategy = 'MA'")
    
    if updates_made:
        # Save updated config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("\n✅ Updated best_strategy.json successfully!")
        print("\n📊 New Configuration:")
        print(f"   Strategy: {config.get('Strategy')}")
        print(f"   strategy_type: {config.get('strategy_type')}")
        print(f"   enable_adaptive_strategy: {config.get('enable_adaptive_strategy')}")
        print(f"   Short_Window: {config.get('Short_Window')}")
        print(f"   Long_Window: {config.get('Long_Window')}")
        
        print("\n⚠️  IMPORTANT: You need to:")
        print("1. Stop the current auto-trader")
        print("2. Run this script")
        print("3. Restart auto-trader to use pure MA strategy")
    else:
        print("\n✅ Configuration already correct - no changes needed")

if __name__ == "__main__":
    update_strategy_config()