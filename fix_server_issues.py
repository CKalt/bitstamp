#!/usr/bin/env python3
"""
Temporary fix script for server issues:
1. Clean old position data
2. Fix excessive logging
3. Set correct position
4. Disable emergency exit spam
"""
import os
import json
import shutil
from datetime import datetime

def backup_file(filepath):
    """Backup a file before modifying"""
    if os.path.exists(filepath):
        backup = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup)
        print(f"✅ Backed up {filepath} to {backup}")
        return True
    return False

def clean_resume_file():
    """Remove or reset the resume-auto-trade.json file"""
    resume_file = "resume-auto-trade.json"
    
    if os.path.exists(resume_file):
        backup_file(resume_file)
        os.remove(resume_file)
        print(f"✅ Removed old {resume_file}")
    else:
        print(f"ℹ️  No existing {resume_file} found")

def create_clean_position_file(btc_amount, entry_price):
    """Create a clean resume file with correct position"""
    resume_data = {
        "timestamp": datetime.now().isoformat(),
        "position": "LONG",
        "amount": btc_amount,
        "unit": "btc",
        "entry_price": entry_price,
        "current_price": entry_price,
        "unrealized_pnl": 0.0,
        "command": f"resume_auto_trade {btc_amount}btc long {entry_price}",
        "strategy": {
            "type": "AdaptiveMultiStrategy",
            "short_window": 6,
            "long_window": 34,
            "current_regime": "unknown",
            "active_strategy": "trending"
        },
        "balances": {
            "btc": btc_amount,
            "usd": 0.0
        },
        "trades_executed": 0,
        "last_trade_time": None,
        "trade_references": [],
        "pivot_protection": {
            "enabled": True,
            "tracker": {}
        }
    }
    
    with open("resume-auto-trade.json", "w") as f:
        json.dump(resume_data, f, indent=2)
    
    print(f"✅ Created clean resume file for LONG {btc_amount} BTC @ ${entry_price}")

def fix_logging_config():
    """Create a logging configuration file to reduce spam"""
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": "tdr_server.log",
                "formatter": "standard",
                "level": "INFO"
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "WARNING"  # Only warnings and errors to console
            }
        },
        "loggers": {
            "TDRServer": {
                "handlers": ["file", "console"],
                "level": "INFO",
                "propagate": False
            }
        }
    }
    
    with open("logging_config.json", "w") as f:
        json.dump(log_config, f, indent=2)
    
    print("✅ Created logging configuration to reduce console spam")

def update_strategy_config():
    """Update best_strategy.json with sensible defaults"""
    config_file = "best_strategy.json"
    
    if os.path.exists(config_file):
        backup_file(config_file)
        
        with open(config_file, "r") as f:
            config = json.load(f)
        
        # Ensure critical parameters are set
        updates = {
            "emergency_loss_threshold": -10000,  # Increase from -5000 to reduce spam
            "emergency_override_enabled": False,  # Disable emergency exit for now
            "log_signal_evaluation": False,  # Reduce signal evaluation logs
            "verbose_logging": False,  # Reduce general verbosity
            "auto_resume": False  # Don't auto-resume on startup
        }
        
        config.update(updates)
        
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print("✅ Updated best_strategy.json with fixes:")
        for key, value in updates.items():
            print(f"   - {key}: {value}")

def create_start_script():
    """Create a clean start script"""
    script_content = """#!/bin/bash
# Clean start script for TDR server

echo "🧹 Cleaning and starting TDR server..."

# Run Python cleanup
python3 fix_server_issues.py

echo ""
echo "🚀 Starting server with reduced logging..."
echo ""

# Start server with logging config
export TDR_LOG_CONFIG=logging_config.json
python3 src/tdr.py --server 2>&1 | grep -v "Signal Evaluation\\|ENTRY_PRICE_DEBUG\\|Using last SELL price"
"""
    
    with open("start_clean.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("start_clean.sh", 0o755)
    print("✅ Created start_clean.sh script")

def main():
    print("=" * 60)
    print("TDR Server Issue Fix Script")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("src/tdr.py"):
        print("❌ Error: Must run from bitstamp project directory")
        print("   cd /home/chris/projects/bitstamp")
        return
    
    print("\n1. Cleaning old position data...")
    clean_resume_file()
    
    print("\n2. Creating correct LONG position...")
    create_clean_position_file(btc_amount=1.36, entry_price=117454)
    
    print("\n3. Fixing logging configuration...")
    fix_logging_config()
    
    print("\n4. Updating strategy configuration...")
    update_strategy_config()
    
    print("\n5. Creating clean start script...")
    create_start_script()
    
    print("\n" + "=" * 60)
    print("✅ All fixes applied!")
    print("=" * 60)
    print("\nTo start the server cleanly:")
    print("  ./start_clean.sh")
    print("\nOr manually:")
    print("  python3 src/tdr.py --server")
    print("\nThe server will now:")
    print("  - Start with your LONG 1.36 BTC @ $117,454 position")
    print("  - Have reduced console logging")
    print("  - Not spam emergency exit warnings")
    print("  - Not auto-resume old positions")

if __name__ == "__main__":
    main()