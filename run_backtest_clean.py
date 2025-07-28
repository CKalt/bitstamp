#!/usr/bin/env python3
"""
Clean backtest runner with progress tracking
"""
import subprocess
import json
import os
import time
import warnings
from datetime import datetime, timedelta
import sys

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


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


def run_single_backtest(ma_short, ma_long, days_back=30, test_num=1, total_tests=1):
    """Run a single backtest with clean output"""
    
    config_file = create_test_config(ma_short, ma_long)
    output_file = f"result_ma_{ma_short}_{ma_long}.json"
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Python command with warnings suppressed
    cmd = [
        "env/bin/python", "-W", "ignore", "src/bktst.py",
        "--data", "btcusd.log",
        "--config", config_file,
        "--start-date", start_date.strftime("%Y-%m-%d"),
        "--end-date", end_date.strftime("%Y-%m-%d"),
        "--save-results", output_file
    ]
    
    print(f"\n[{test_num}/{total_tests}] Testing MA {ma_short}/{ma_long} (last {days_back} days)")
    print("-" * 50)
    
    start_time = time.time()
    last_update = start_time
    
    try:
        # Run with output capture
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                 text=True, bufsize=1)
        
        # Monitor output
        trade_count = 0
        data_loaded = False
        
        while True:
            line = process.stdout.readline()
            if line == '' and process.poll() is not None:
                break
                
            if line:
                # Parse key events
                if "Loading data from" in line:
                    print("   📂 Loading data...", end='\r')
                elif "Loaded" in line and "data points" in line:
                    data_loaded = True
                    try:
                        points = line.split("Loaded")[1].split("data points")[0].strip()
                        print(f"   ✓ Loaded {points} data points", end='\r')
                    except:
                        pass
                elif "Initial LONG position" in line or "Initial SHORT position" in line:
                    print("   🔄 Running simulation...", end='\r')
                elif "BUY @" in line or "SELL @" in line:
                    trade_count += 1
                    if trade_count % 10 == 0:  # Update every 10 trades
                        elapsed = time.time() - start_time
                        print(f"   🔄 Simulating... {trade_count} trades ({elapsed:.0f}s)", end='\r')
                
                # Print progress every 5 seconds
                current_time = time.time()
                if current_time - last_update > 5:
                    elapsed = current_time - start_time
                    if data_loaded:
                        print(f"   ⏱  Processing... {elapsed:.0f}s ({trade_count} trades so far)", end='\r')
                    else:
                        print(f"   ⏱  Loading data... {elapsed:.0f}s", end='\r')
                    last_update = current_time
        
        # Get any errors
        stderr = process.stderr.read()
        rc = process.poll()
        
        elapsed = time.time() - start_time
        
        if rc == 0 and os.path.exists(output_file):
            # Load results
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            # Clean up
            os.remove(config_file)
            os.remove(output_file)
            
            # Clear line and print summary
            print(" " * 60, end='\r')  # Clear progress line
            print(f"   ✅ Complete: {elapsed:.1f}s, {results.get('num_trades', 0)} trades")
            print(f"   📊 Return: {results.get('total_return', 0) * 100:+.2f}%, Sharpe: {results.get('sharpe_ratio', 0):.3f}")
            
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
            print(f"\n   ❌ Failed (return code: {rc})")
            if stderr:
                print(f"   Error: {stderr[:200]}...")
            
    except Exception as e:
        print(f"\n   ❌ Exception: {str(e)}")
    finally:
        # Always clean up
        if os.path.exists(config_file):
            os.remove(config_file)
        if os.path.exists(output_file):
            os.remove(output_file)
    
    return None


def main():
    print("=" * 60)
    print("CLEAN BACKTEST RUNNER")
    print("=" * 60)
    print(f"Testing Period: Last 30 days")
    print(f"Data: btcusd.log (4.4GB)")
    print(f"Note: First test takes longer due to data loading")
    
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
    
    print(f"\nTests to run: {len(test_configs)}")
    print("=" * 60)
    
    results = []
    total_start = time.time()
    
    # Run each test
    for i, (ma_short, ma_long) in enumerate(test_configs, 1):
        result = run_single_backtest(ma_short, ma_long, days_back=30, 
                                   test_num=i, total_tests=len(test_configs))
        
        if result:
            results.append(result)
        
        # Overall progress
        elapsed_total = time.time() - total_start
        avg_time = elapsed_total / i
        remaining = avg_time * (len(test_configs) - i)
        
        print(f"\n⏱  Overall Progress: {i}/{len(test_configs)} tests")
        print(f"   Elapsed: {elapsed_total/60:.1f}m | ETA: {remaining/60:.1f}m more")
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
        marker = "🏆" if i == 1 else "  "
        print(f"{marker} {i:<3} {r['ma_short']:>5}/{r['ma_long']:<5} {r['return']:>9.2f}% {r['trades']:>8} {r['sharpe']:>8.3f} {r['time']:>7.1f}s")
    
    if results:
        best = results[0]
        
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
        print(f"\n📋 Winner: MA {best['ma_short']}/{best['ma_long']}")
        print(f"   Return: {best['return']:.2f}%")
        print(f"   Sharpe: {best['sharpe']:.3f}")
        print(f"   Trades: {best['trades']} ({best['trades']/30:.1f} per day)")
        
        # Show comparison with current
        current = next((r for r in results if r['ma_short'] == 6 and r['ma_long'] == 34), None)
        if current and current != best:
            print(f"\n📊 Current MA 6/34: {current['return']:.2f}% return")
            print(f"   Improvement: {best['return'] - current['return']:+.2f}%")
    
    total_time = time.time() - total_start
    print(f"\n⏱  Total time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()