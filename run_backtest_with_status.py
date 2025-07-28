#!/usr/bin/env python3
"""
Run backtests with detailed progress tracking and regular status updates
"""
import subprocess
import json
import os
import time
import threading
from datetime import datetime, timedelta
import sys


class ProgressTracker:
    def __init__(self, total_tests):
        self.total_tests = total_tests
        self.completed_tests = 0
        self.start_time = time.time()
        self.test_times = []
        self.current_test_start = None
        self.current_ma = None
        self.current_status = "Starting..."
        self.last_output = ""
        
    def start_test(self, ma_short, ma_long):
        self.current_test_start = time.time()
        self.current_ma = (ma_short, ma_long)
        self.current_status = f"Testing MA {ma_short}/{ma_long}"
        print(f"\n[{self.completed_tests + 1}/{self.total_tests}] Testing MA {ma_short}/{ma_long}...", flush=True)
        
    def update_status(self, status):
        self.current_status = status
        self.last_output = status
        
    def complete_test(self, success=True):
        test_duration = time.time() - self.current_test_start
        self.test_times.append(test_duration)
        self.completed_tests += 1
        
        # Calculate statistics
        avg_time = sum(self.test_times) / len(self.test_times)
        remaining_tests = self.total_tests - self.completed_tests
        estimated_remaining = avg_time * remaining_tests
        total_elapsed = time.time() - self.start_time
        
        # Progress bar
        progress = self.completed_tests / self.total_tests
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r[{bar}] {progress*100:.1f}%", end='', flush=True)
        print(f"\n   ✓ Completed in {test_duration:.1f}s (avg: {avg_time:.1f}s per test)")
        print(f"   ⏱  Elapsed: {self.format_time(total_elapsed)} | Remaining: {self.format_time(estimated_remaining)}")
        print(f"   📊 ETA: {(datetime.now() + timedelta(seconds=estimated_remaining)).strftime('%H:%M:%S')}")
        
    def get_status(self):
        if self.current_test_start:
            elapsed = time.time() - self.current_test_start
            return f"{self.current_status} ({elapsed:.0f}s) - {self.last_output}"
        return self.current_status
        
    def format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


def status_monitor(tracker, stop_event):
    """Background thread to print status every 10 seconds"""
    while not stop_event.is_set():
        time.sleep(10)
        if not stop_event.is_set():
            elapsed = time.time() - tracker.start_time
            print(f"\n📍 STATUS: {tracker.get_status()}", flush=True)
            print(f"   Progress: {tracker.completed_tests}/{tracker.total_tests} tests | Total time: {tracker.format_time(elapsed)}", flush=True)


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


def run_single_backtest(ma_short, ma_long, days_back=30, tracker=None):
    """Run a single backtest with progress monitoring"""
    
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
        "--save-results", output_file
    ]
    
    try:
        # Run with real-time output monitoring
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        
        # Monitor output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output and tracker:
                # Update status based on output
                if "Loading data" in output:
                    tracker.update_status("📂 Loading historical data...")
                elif "Loaded" in output and "data points" in output:
                    # Extract number of data points
                    try:
                        points = output.split("Loaded")[1].split("data points")[0].strip()
                        tracker.update_status(f"📊 Processing {points} data points...")
                    except:
                        tracker.update_status("📊 Processing price data...")
                elif "Resampling" in output:
                    tracker.update_status("🕐 Converting to hourly data...")
                elif "Running backtest" in output:
                    tracker.update_status("🔄 Simulating trades...")
                elif "Trade" in output and "executed" in output:
                    # Count trades
                    tracker.update_status("💹 Executing trades...")
                elif "Calculating" in output:
                    tracker.update_status("📈 Calculating metrics...")
                elif "Sharpe" in output or "Return" in output:
                    tracker.update_status("✅ Finalizing results...")
        
        # Check for errors
        stderr = process.stderr.read()
        rc = process.poll()
        
        if rc == 0:
            # Load results
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            # Clean up
            os.remove(config_file)
            if os.path.exists(output_file):
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
            print(f"\n   ❌ Error (return code {rc}): {stderr}")
            
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
    print("BACKTEST RUNNER WITH PROGRESS TRACKING")
    print("=" * 60)
    print(f"Testing Period: Last 30 days")
    print(f"End Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Start Date: {(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')}")
    
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
    
    print(f"\nTotal tests to run: {len(test_configs)}")
    print("Estimated time: 10-20 minutes (depends on data size)")
    print("\n💡 Status updates will appear every 10 seconds\n")
    
    # Initialize progress tracker
    tracker = ProgressTracker(len(test_configs))
    results = []
    
    # Start status monitor thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=status_monitor, args=(tracker, stop_event))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Run each test
    for ma_short, ma_long in test_configs:
        tracker.start_test(ma_short, ma_long)
        
        result = run_single_backtest(ma_short, ma_long, days_back=30, tracker=tracker)
        
        if result:
            results.append(result)
            print(f"     ✅ Return: {result['return']:+.2f}% | Trades: {result['trades']} | Sharpe: {result['sharpe']:.3f}")
        else:
            print(f"     ❌ Failed")
        
        tracker.complete_test(result is not None)
    
    # Stop monitor thread
    stop_event.set()
    
    # Sort by return
    results.sort(key=lambda x: x['return'], reverse=True)
    
    # Display results
    print("\n" + "=" * 60)
    print("FINAL RESULTS (sorted by return)")
    print("=" * 60)
    print(f"{'Rank':<5} {'MA Config':>12} {'Return':>10} {'Trades':>8} {'Sharpe':>8}")
    print("-" * 45)
    
    for i, r in enumerate(results, 1):
        print(f"{i:<5} {r['ma_short']:>5}/{r['ma_long']:<5} {r['return']:>9.2f}% {r['trades']:>8} {r['sharpe']:>8.3f}")
    
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
            
            # Copy other parameters from current config
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
        print("\nTo deploy:")
        print("1. Review the configuration")
        print("2. cp best_strategy.json best_strategy.json.backup_$(date +%Y%m%d)")
        print("3. cp best_strategy_30day_optimized.json best_strategy.json")
        print("4. git add best_strategy.json && git commit -m 'Update MA parameters' && git push")
        print("5. ssh ck → cd /home/chris/projects/bitstamp → git pull")
    
    total_time = time.time() - tracker.start_time
    print(f"\n⏱  Total time: {tracker.format_time(total_time)}")


if __name__ == "__main__":
    main()