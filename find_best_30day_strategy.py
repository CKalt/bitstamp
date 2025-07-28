#!/usr/bin/env python3
"""
Find the best MA parameters for the last 30 days
Tests multiple combinations and outputs best_strategy.json format
"""
import subprocess
import json
import os
from datetime import datetime, timedelta


def test_ma_combination(ma_short, ma_long, days_back=30):
    """Test a specific MA combination"""
    
    # Create config for this test
    config = {
        "Short_Window": ma_short,
        "Long_Window": ma_long,
        "initial_balance": 10000,
        "fee_rate": 0.0012,
        "slippage_rate": 0.0005,
        "enable_pivot_protection": False,  # Match current live settings
        "enable_adaptive_strategy": False,
        "strategy_type": "MA"
    }
    
    config_file = f"test_ma_{ma_short}_{ma_long}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Run backtest
    output_file = f"result_ma_{ma_short}_{ma_long}.json"
    cmd = [
        "env/bin/python", "src/bktst.py",
        "--data", "btcusd.log",
        "--config", config_file,
        "--start-date", start_date.strftime("%Y-%m-%d"),
        "--end-date", end_date.strftime("%Y-%m-%d"),
        "--save-results", output_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            # Load results
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            # Clean up
            os.remove(config_file)
            os.remove(output_file)
            
            return {
                'ma_short': ma_short,
                'ma_long': ma_long,
                'return': results.get('total_return', 0) * 100,
                'trades': results.get('num_trades', 0),
                'sharpe': results.get('sharpe_ratio', 0),
                'final_equity': results.get('final_equity', 10000)
            }
        else:
            print(f"Error testing MA {ma_short}/{ma_long}: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print(f"Timeout testing MA {ma_short}/{ma_long}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Clean up on error
    if os.path.exists(config_file):
        os.remove(config_file)
    
    return None


def main():
    print("Finding best MA parameters for last 30 days...")
    print(f"Testing period: {(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}")
    print()
    
    # Test combinations - including current and variations
    test_configs = [
        (6, 34),   # Current
        (5, 15),   # Very fast
        (8, 21),   # Fibonacci
        (10, 20),  # Balanced
        (10, 30),  # Medium
        (12, 26),  # MACD-like
        (15, 30),  # Medium-slow
        (20, 50),  # Classic
        (10, 46),  # Old config
    ]
    
    results = []
    
    print(f"{'MA Config':>12} {'Return':>10} {'Trades':>8} {'Sharpe':>8}")
    print("-" * 40)
    
    for ma_short, ma_long in test_configs:
        print(f"Testing {ma_short:>2}/{ma_long:>2}...", end='', flush=True)
        result = test_ma_combination(ma_short, ma_long)
        
        if result:
            results.append(result)
            print(f"\r{ma_short:>2}/{ma_long:>2}       {result['return']:>9.2f}% {result['trades']:>8} {result['sharpe']:>8.3f}")
        else:
            print(f"\r{ma_short:>2}/{ma_long:>2}       FAILED")
    
    if not results:
        print("\nNo successful results!")
        return
    
    # Sort by return
    results.sort(key=lambda x: x['return'], reverse=True)
    
    # Get best result
    best = results[0]
    
    print(f"\n{'='*50}")
    print(f"BEST STRATEGY: MA {best['ma_short']}/{best['ma_long']}")
    print(f"Return: {best['return']:.2f}%")
    print(f"Trades: {best['trades']}")
    print(f"Sharpe: {best['sharpe']:.3f}")
    print(f"Final Value: ${best['final_equity']:.2f}")
    
    # Create best_strategy.json format
    best_strategy = {
        "Frequency": "1H",
        "Strategy": "MA", 
        "Short_Window": best['ma_short'],
        "Long_Window": best['ma_long'],
        "Bar_Size": "1H",
        "Final_Balance": best['final_equity'],
        "Total_Return": best['return'],
        "Total_Trades": float(best['trades']),
        "Average_Trades_Per_Day": best['trades'] / 30.0,
        "Sharpe_Ratio": best['sharpe'],
        "do_live_trades": True,
        "strategy_type": "MA",
        "enable_adaptive_strategy": False,
        "auto_resume": False,
        
        # Include other parameters from current config
        "consecutive_loss_limit": 3,
        "daily_loss_limit": -2000,
        "emergency_loss_threshold": -10000,
        "emergency_override_enabled": False,
        "ignore_small_crosses": True,
        "log_signal_evaluation": True,
        "ma_separation_threshold": 0.3,
        "max_trades_per_day": 5,
        "max_trades_per_hour": 2,
        "min_breakout_distance_percent": 0.2,
        "min_time_between_trades_minutes": 30,
        "pause_after_limit_hours": 4,
        "enable_pivot_protection": False,
        "enable_regime_detection": False,
        "verbose_logging": True,
        
        # Placeholder fields
        "Last_Signal_Action": "GO LONG",
        "Last_Signal_Timestamp": int(datetime.now().timestamp()),
        "Last_Trade_Timestamp": int(datetime.now().timestamp()),
        "Last_Trade_Price": 118000,
        "Profit_Factor": 1.0
    }
    
    # Save to file
    with open('best_strategy_30day_recommendation.json', 'w') as f:
        json.dump(best_strategy, f, indent=2)
    
    print(f"\n✅ Created: best_strategy_30day_recommendation.json")
    
    # Show comparison
    print(f"\n{'='*50}")
    print("COMPARISON WITH CURRENT (MA 6/34):")
    print(f"{'='*50}")
    current = next((r for r in results if r['ma_short'] == 6 and r['ma_long'] == 34), None)
    if current:
        print(f"Current MA 6/34: {current['return']:.2f}% return, {current['trades']} trades")
        print(f"Best MA {best['ma_short']}/{best['ma_long']}: {best['return']:.2f}% return, {best['trades']} trades")
        print(f"Improvement: {best['return'] - current['return']:+.2f}%")
    
    print("\n📋 All Results (sorted by return):")
    for r in results:
        print(f"   MA {r['ma_short']:>2}/{r['ma_long']:>2}: {r['return']:>6.2f}% ({r['trades']:>2} trades, Sharpe: {r['sharpe']:>5.3f})")


if __name__ == "__main__":
    main()