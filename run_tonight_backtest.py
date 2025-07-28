#!/usr/bin/env python3
"""
Backtest runner for tonight - tests multiple MA combinations
and creates live-compatible best_strategy.json files
"""
import subprocess
import json
import os
from datetime import datetime
import shutil


def create_test_config(ma_short, ma_long):
    """Create a test configuration file"""
    config = {
        "Short_Window": ma_short,
        "Long_Window": ma_long,
        "initial_balance": 10000,
        "fee_rate": 0.0012,
        "slippage_rate": 0.0005,
        "enable_pivot_protection": True,
        "pivot_buffer": 100,
        "enable_trailing_pivots": True,
        "pivot_profit_tiers": [
            {"threshold": 0.05, "protection_ratio": 0.70},
            {"threshold": 0.10, "protection_ratio": 0.80},
            {"threshold": 0.15, "protection_ratio": 0.85}
        ]
    }
    
    filename = f"test_config_ma_{ma_short}_{ma_long}.json"
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    return filename


def run_backtest(config_file, output_file):
    """Run backtest with given config"""
    cmd = [
        "python", "src/bktst.py",
        "--data", "btcusd.log",
        "--config", config_file,
        "--start-date", "2024-10-01",  # Last 3 months
        "--save-results", output_file
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return None
    
    # Parse output for key metrics
    output_lines = result.stdout.split('\n')
    metrics = {}
    for line in output_lines:
        if "Sharpe Ratio:" in line:
            metrics['sharpe'] = float(line.split(':')[1].strip())
        elif "Total Return:" in line:
            metrics['return'] = float(line.split(':')[1].strip().rstrip('%'))
        elif "Win Rate:" in line:
            metrics['win_rate'] = float(line.split(':')[1].strip().rstrip('%'))
    
    return metrics


def main():
    print("=== Backtesting Multiple MA Combinations ===")
    print(f"Start time: {datetime.now()}")
    print(f"Data file: btcusd.log")
    print()
    
    # Test configurations
    ma_configs = [
        (6, 34),    # Current live
        (5, 15),    # Very fast
        (8, 21),    # Fibonacci
        (10, 20),   # Common fast
        (10, 46),   # Old config
        (12, 26),   # MACD-like
        (20, 50),   # Classic
        (50, 200),  # Golden cross
    ]
    
    results = []
    
    # Run backtests
    for ma_short, ma_long in ma_configs:
        print(f"\n--- Testing MA {ma_short}/{ma_long} ---")
        
        # Create config
        config_file = create_test_config(ma_short, ma_long)
        output_file = f"backtest_ma_{ma_short}_{ma_long}.json"
        
        # Run backtest
        metrics = run_backtest(config_file, output_file)
        
        if metrics:
            results.append({
                'ma_short': ma_short,
                'ma_long': ma_long,
                'sharpe': metrics.get('sharpe', 0),
                'return': metrics.get('return', 0),
                'win_rate': metrics.get('win_rate', 0),
                'output_file': output_file
            })
            print(f"✅ Sharpe: {metrics.get('sharpe', 0):.3f}, "
                  f"Return: {metrics.get('return', 0):.1f}%, "
                  f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
        
        # Clean up temp config
        os.remove(config_file)
    
    # Sort by Sharpe ratio
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    
    # Print summary
    print("\n=== SUMMARY (sorted by Sharpe Ratio) ===")
    print(f"{'MA Config':>12} {'Sharpe':>8} {'Return':>8} {'Win Rate':>9}")
    print("-" * 40)
    for r in results:
        print(f"MA {r['ma_short']:>2}/{r['ma_long']:>3}     "
              f"{r['sharpe']:>7.3f}  {r['return']:>7.1f}%  {r['win_rate']:>8.1f}%")
    
    if results:
        best = results[0]
        print(f"\n🏆 Best: MA {best['ma_short']}/{best['ma_long']} "
              f"(Sharpe: {best['sharpe']:.3f})")
        
        # Convert best to live format
        print("\nConverting best result to live format...")
        cmd = [
            "python", "convert_backtest_to_live.py",
            best['output_file'],
            "--ma-short", str(best['ma_short']),
            "--ma-long", str(best['ma_long']),
            "--output", "best_strategy_recommended.json"
        ]
        subprocess.run(cmd)
        
        print("\n✅ Created: best_strategy_recommended.json")
        print("\nTo deploy:")
        print("\n=== ON MAC (where you are now) ===")
        print("1. Review the recommended configuration")
        print("2. cp best_strategy.json best_strategy.json.backup_$(date +%Y%m%d)")
        print("3. cp best_strategy_recommended.json best_strategy.json")
        print("4. git add best_strategy.json")
        print("5. git commit -m 'Update MA parameters from backtest'")
        print("6. git push")
        print("\n=== ON SERVER (ssh ck) ===")
        print("7. ssh ck")
        print("8. gg btc  # or cd /home/chris/projects/bitstamp")
        print("9. git pull")
        print("10. Restart trading server with new config")
    
    print(f"\nCompleted at: {datetime.now()}")


if __name__ == "__main__":
    main()