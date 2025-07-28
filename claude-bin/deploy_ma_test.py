#!/usr/bin/env python3
"""
Deploy MA 3/22 configuration to test server with enhanced logging
"""
import json
import os
from datetime import datetime


def create_test_config():
    """Create test configuration for MA 3/22"""
    
    config = {
        # MA Parameters from backtest winner
        "Short_Window": 3,
        "Long_Window": 22,
        "Frequency": "1H",
        "Strategy": "MA",
        "Bar_Size": "1H",
        
        # Enable live trading but limit position
        "do_live_trades": True,
        "strategy_type": "MA",
        "enable_adaptive_strategy": False,
        "auto_resume": False,
        
        # TEST SERVER LIMITS
        "max_position_btc": 0.001,
        "max_position_usd": 100,
        "trading_mode": "test_comparison",
        "server_port": 4002,
        "log_prefix": "TEST_MA322",
        
        # Trading constraints
        "ma_separation_threshold": 0.3,
        "max_trades_per_day": 10,  # Allow more for testing
        "max_trades_per_hour": 3,
        "min_time_between_trades_minutes": 20,
        "enable_pivot_protection": False,
        "enable_regime_detection": False,
        
        # Enhanced logging
        "log_signal_evaluation": True,
        "verbose_logging": True,
        "enable_comparison_logging": True,
        
        # Risk management
        "consecutive_loss_limit": 3,
        "daily_loss_limit": -100,  # In USD for test
        "emergency_loss_threshold": -200,
        "emergency_override_enabled": False,
        
        # Backtested performance (for reference)
        "backtest_30day_return": 8.64,
        "backtest_trades": 42,
        "backtest_sharpe": 6.70,
        "backtest_max_drawdown": 4.5,
        
        # Metadata
        "config_created": datetime.now().isoformat(),
        "config_purpose": "Test MA 3/22 with comparison logging",
        "deployment_note": "CAUTION: MA 3 is very aggressive, test carefully"
    }
    
    return config


def main():
    """Deploy configuration"""
    print("="*60)
    print("MA 3/22 TEST DEPLOYMENT")
    print("="*60)
    
    config = create_test_config()
    
    # Save configuration
    with open("best_strategy_test_ma322.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created: best_strategy_test_ma322.json")
    print(f"\nConfiguration:")
    print(f"  - MA periods: {config['Short_Window']}/{config['Long_Window']}")
    print(f"  - Max position: {config['max_position_btc']} BTC (${config['max_position_usd']})")
    print(f"  - Max trades/day: {config['max_trades_per_day']}")
    print(f"  - Comparison logging: {config['enable_comparison_logging']}")
    
    print("\n📋 Deployment steps:")
    print("1. Copy config to test directory:")
    print("   cp best_strategy_test_ma322.json best_strategy.json")
    print("\n2. Commit and push to development branch:")
    print("   git add best_strategy.json")
    print("   git commit -m 'Test MA 3/22 with comparison logging'")
    print("   git push origin development")
    print("\n3. On server (ssh ck):")
    print("   gg tst")
    print("   git pull")
    print("   # Stop current test server if running")
    print("   screen -S server-tst -X quit")
    print("   # Start with enhanced strategy")
    print("   screen -dmS server-tst bash -c 'source source-venv.sh && python src/tdr.py --server --enhanced-ma'")
    print("\n4. Start test trading:")
    print("   curl -X POST http://localhost:4002/api/command \\")
    print("     -H \"Content-Type: application/json\" \\")
    print("     -d '{\"command\": \"auto_trade 0.001btc MA short=3 long=22 do_live_trades=True\"}'")
    print("\n5. Monitor logs:")
    print("   tail -f logs/backtest_comparison/live_trading_*.jsonl")
    print("\n6. Run daily comparison:")
    print("   python claude-bin/compare_live_vs_backtest.py")
    
    print("\n⚠️  WARNINGS:")
    print("  - MA 3 is EXTREMELY sensitive (3-hour moving average)")
    print("  - Expect many trades (1-2 per day based on backtest)")
    print("  - Test position is limited to 0.001 BTC (~$100)")
    print("  - Comparison logs will verify backtest accuracy")
    
    # Create startup script
    startup_script = """#!/bin/bash
# Start test server with MA 3/22 and enhanced logging

cd "$(dirname "$0")"
source source-venv.sh

echo "Starting test server with MA 3/22..."
echo "Configuration: Short=3, Long=22"
echo "Max position: 0.001 BTC"
echo "Comparison logging: ENABLED"

# Start server with enhanced MA strategy
python src/tdr.py --server --port 4002 --strategy enhanced_ma &

echo "Server starting on port 4002..."
echo "Logs: logs/backtest_comparison/"
"""
    
    with open("start_test_ma322.sh", "w") as f:
        f.write(startup_script)
    
    os.chmod("start_test_ma322.sh", 0o755)
    print(f"\n✅ Created: start_test_ma322.sh")


if __name__ == "__main__":
    main()