#!/usr/bin/env python3
"""
Run backtests with verbose output and timing
"""
import subprocess
import json
import os
import time
from datetime import datetime, timedelta


def create_test_config(ma_short, ma_long):
    """Create test configuration"""
    config = {
        "Short_Window": ma_short,
        "Long_Window": ma_long,
        "initial_balance": 10000,
        "fee_rate": 0.0012,
        "slippage_rate": 0.0005,
        "enable_pivot_protection": False,
        "enable_adaptive_strategy": False,
        "strategy_type": "MA",
        "enable_trailing_pivots": False
    }
    
    filename = f"temp_config_ma_{ma_short}_{ma_long}.json"
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    return filename


def run_single_backtest(ma_short, ma_long, days_back=30):
    """Run a single backtest showing all output"""
    
    config_file = create_test_config(ma_short, ma_long)
    output_file = f"result_ma_{ma_short}_{ma_long}.json"
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    cmd = [
        "env/bin/python", "src/bktst.py",
        "--data", "btcusd.log",
        "--config", config_file,
        "--start-date", start_date.strftime("%Y-%m-%d"),
        "--end-date", end_date.strftime("%Y-%m-%d"),
        "--save-results", output_file,
        "--verbose"  # Add verbose flag if supported
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Run and show output in real-time
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.1f} seconds")
        
        if result.returncode == 0 and os.path.exists(output_file):
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
                'final_equity': results.get('final_equity', 10000),
                'time': elapsed
            }
        else:
            print(f"❌ Backtest failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    finally:
        # Always clean up
        if os.path.exists(config_file):
            os.remove(config_file)
        if os.path.exists(output_file):
            os.remove(output_file)
    
    return None


def main():
    print("=" * 60)
    print("VERBOSE BACKTEST RUNNER")
    print("=" * 60)
    print(f"Testing Period: Last 30 days")
    print(f"Start: {(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Test configurations
    test_configs = [
        (6, 34),   # Current
        (5, 15),   # Very fast
        (8, 21),   # Fibonacci
        (10, 20),  # Balanced
        (10, 30),  # Medium
        (12, 26),  # MACD-like
        (15, 30),  # Medium-slow
        (20, 50),  # Classic
    ]
    
    print(f"\nTotal tests: {len(test_configs)}")
    print("=" * 60)
    
    results = []
    total_start = time.time()
    
    # Run each test
    for i, (ma_short, ma_long) in enumerate(test_configs, 1):
        print(f"\n🔷 TEST {i}/{len(test_configs)}: MA {ma_short}/{ma_long}")
        
        result = run_single_backtest(ma_short, ma_long, days_back=30)
        
        if result:
            results.append(result)
            print(f"\n✅ Summary: Return={result['return']:+.2f}%, Trades={result['trades']}, Sharpe={result['sharpe']:.3f}")
        else:
            print(f"\n❌ Test failed")
        
        # Progress update
        elapsed_total = time.time() - total_start
        avg_time = elapsed_total / i
        remaining = avg_time * (len(test_configs) - i)
        print(f"\n⏱  Progress: {i}/{len(test_configs)} | Elapsed: {elapsed_total:.0f}s | ETA: {remaining:.0f}s")
        print("=" * 60)
    
    # Sort by return
    results.sort(key=lambda x: x['return'], reverse=True)
    
    # Display final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS (sorted by return)")
    print("=" * 60)
    print(f"{'Rank':<5} {'MA Config':>12} {'Return':>10} {'Trades':>8} {'Sharpe':>8} {'Time':>8}")
    print("-" * 50)
    
    for i, r in enumerate(results, 1):
        print(f"{i:<5} {r['ma_short']:>5}/{r['ma_long']:<5} {r['return']:>9.2f}% {r['trades']:>8} {r['sharpe']:>8.3f} {r['time']:>7.1f}s")
    
    if results:
        best = results[0]
        print(f"\n🏆 WINNER: MA {best['ma_short']}/{best['ma_long']} with {best['return']:.2f}% return")
        
        # Create best_strategy.json
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
            "ma_separation_threshold": 0.3,
            "max_trades_per_day": 5,
            "max_trades_per_hour": 2,
            "min_time_between_trades_minutes": 30,
            "enable_pivot_protection": False,
            "enable_regime_detection": False,
            "log_signal_evaluation": True,
            "verbose_logging": True
        }
        
        with open('best_strategy_30day_optimized.json', 'w') as f:
            json.dump(best_strategy, f, indent=2)
        
        print(f"\n✅ Created: best_strategy_30day_optimized.json")
    
    total_time = time.time() - total_start
    print(f"\n⏱  Total time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()