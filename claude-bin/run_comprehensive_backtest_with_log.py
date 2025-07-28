#!/usr/bin/env python3
"""
Comprehensive backtest with MA and RSI strategies on HOURLY data
Saves output to file for analysis
"""
import time
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import json
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from data.loader import parse_log_file

# Create log file
log_file = open('backtest_results_log.txt', 'w')

def log_print(msg):
    """Print to both console and log file"""
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()


class ProgressTracker:
    def __init__(self, total_tests):
        self.total_tests = total_tests
        self.completed_tests = 0
        self.start_time = time.time()
        self.test_times = []
        self.current_test = None
        self.lock = threading.Lock()
        
    def start_test(self, test_name):
        with self.lock:
            self.current_test = test_name
            self.current_test_start = time.time()
            
    def complete_test(self):
        with self.lock:
            if hasattr(self, 'current_test_start'):
                test_time = time.time() - self.current_test_start
                self.test_times.append(test_time)
                self.completed_tests += 1
                
    def get_status(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            
            if self.completed_tests > 0:
                avg_time = sum(self.test_times) / len(self.test_times)
                remaining_tests = self.total_tests - self.completed_tests
                eta_seconds = avg_time * remaining_tests
                eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                
                return {
                    'current': self.current_test,
                    'completed': self.completed_tests,
                    'total': self.total_tests,
                    'elapsed': elapsed,
                    'eta': eta_time.strftime('%H:%M:%S'),
                    'avg_time': avg_time
                }
            else:
                return {
                    'current': self.current_test,
                    'completed': 0,
                    'total': self.total_tests,
                    'elapsed': elapsed,
                    'eta': 'Calculating...',
                    'avg_time': 0
                }


def progress_monitor(tracker, stop_event):
    """Background thread to print progress every 10 seconds"""
    while not stop_event.is_set():
        time.sleep(10)
        if not stop_event.is_set():
            status = tracker.get_status()
            log_print(f"\n📊 PROGRESS UPDATE [{datetime.now().strftime('%H:%M:%S')}]")
            log_print(f"   Current test: {status['current']}")
            log_print(f"   Completed: {status['completed']}/{status['total']} tests")
            log_print(f"   Elapsed: {status['elapsed']/60:.1f} minutes")
            log_print(f"   ETA: {status['eta']}")
            log_print(f"   Avg time per test: {status['avg_time']:.1f}s")
            log_print("-" * 60)


def load_hourly_data(days_back=30):
    """Load and resample data to hourly bars"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    log_print(f"Loading {days_back} days of data...")
    log_print(f"Start date: {start_date}")
    log_print(f"End date: {end_date}")
    
    df = parse_log_file('btcusd.log', start_date=start_date, end_date=end_date)
    
    if df is None or len(df) == 0:
        raise ValueError("No data loaded")
    
    # Ensure datetime index
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
    
    log_print(f"Loaded {len(df):,} ticks")
    log_print(f"First tick: {df.index[0]}")
    log_print(f"Last tick: {df.index[-1]}")
    
    # Resample to hourly bars
    log_print("Resampling to hourly bars (matching live system)...")
    df_hourly = df.resample('1H').agg({
        'price': ['first', 'max', 'min', 'last'],
        'amount': 'sum'
    })
    df_hourly.columns = ['open', 'high', 'low', 'close', 'volume']
    df_hourly = df_hourly.dropna()
    
    log_print(f"Created {len(df_hourly):,} hourly bars")
    log_print(f"First bar: {df_hourly.index[0]}")
    log_print(f"Last bar: {df_hourly.index[-1]}")
    log_print(f"Sample data (last 5 bars):")
    log_print(str(df_hourly.tail()))
    
    return df_hourly


def calculate_rsi(df, period=14):
    """Calculate RSI indicator"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def backtest_ma_strategy(df_hourly, ma_short, ma_long):
    """Backtest MA crossover strategy"""
    df = df_hourly.copy()
    
    # Calculate MAs
    df['MA_short'] = df['close'].rolling(window=ma_short).mean()
    df['MA_long'] = df['close'].rolling(window=ma_long).mean()
    
    # Skip NaN values
    df = df.dropna()
    
    # Generate signals
    df['signal'] = 0
    df.loc[df['MA_short'] > df['MA_long'], 'signal'] = 1
    df.loc[df['MA_short'] < df['MA_long'], 'signal'] = -1
    
    return simulate_trading(df, f"MA_{ma_short}_{ma_long}")


def backtest_rsi_strategy(df_hourly, rsi_period=14, oversold=30, overbought=70):
    """Backtest RSI strategy"""
    df = df_hourly.copy()
    
    # Calculate RSI
    df['RSI'] = calculate_rsi(df, rsi_period)
    
    # Skip NaN values
    df = df.dropna()
    
    # Generate signals
    # Traditional RSI: Buy when oversold, Sell when overbought
    df['signal'] = 0
    df.loc[df['RSI'] < oversold, 'signal'] = 1  # Buy signal
    df.loc[df['RSI'] > overbought, 'signal'] = -1  # Sell signal
    
    # Forward fill signals (maintain position between signals)
    df['signal'] = df['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return simulate_trading(df, f"RSI_{rsi_period}_{oversold}_{overbought}")


def simulate_trading(df, strategy_name):
    """Simulate trading with given signals"""
    position = 0
    balance = 10000
    btc = 0
    trades = []
    fee_rate = 0.0012
    
    for i in range(1, len(df)):
        prev_signal = df.iloc[i-1]['signal']
        curr_signal = df.iloc[i]['signal']
        
        if prev_signal != curr_signal and curr_signal != 0:
            price = df.iloc[i]['close']
            timestamp = df.index[i]
            
            if curr_signal == 1 and position <= 0:
                # BUY
                if balance > 0:
                    fee = balance * fee_rate
                    btc = (balance - fee) / price
                    balance = 0
                    position = 1
                    trades.append({
                        'time': timestamp,
                        'type': 'BUY',
                        'price': price
                    })
            elif curr_signal == -1 and position >= 0:
                # SELL
                if btc > 0:
                    usd = btc * price
                    fee = usd * fee_rate
                    balance = usd - fee
                    btc = 0
                    position = -1
                    trades.append({
                        'time': timestamp,
                        'type': 'SELL',
                        'price': price
                    })
    
    # Final value
    final_value = balance if btc == 0 else btc * df.iloc[-1]['close']
    total_return = ((final_value - 10000) / 10000) * 100
    
    # Calculate max drawdown
    equity_curve = []
    temp_balance = 10000
    temp_btc = 0
    temp_position = 0
    
    for i in range(len(df)):
        if i > 0 and df.iloc[i-1]['signal'] != df.iloc[i]['signal'] and df.iloc[i]['signal'] != 0:
            price = df.iloc[i]['close']
            
            if df.iloc[i]['signal'] == 1 and temp_position <= 0:
                if temp_balance > 0:
                    fee = temp_balance * fee_rate
                    temp_btc = (temp_balance - fee) / price
                    temp_balance = 0
                    temp_position = 1
            elif df.iloc[i]['signal'] == -1 and temp_position >= 0:
                if temp_btc > 0:
                    usd = temp_btc * price
                    fee = usd * fee_rate
                    temp_balance = usd - fee
                    temp_btc = 0
                    temp_position = -1
        
        current_value = temp_balance if temp_btc == 0 else temp_btc * df.iloc[i]['close']
        equity_curve.append(current_value)
    
    max_drawdown = 0
    peak = equity_curve[0]
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    return {
        'strategy': strategy_name,
        'trades': len(trades),
        'return': total_return,
        'final_value': final_value,
        'max_drawdown': max_drawdown * 100,
        'trades_per_day': len(trades) / 30.0
    }


def main():
    log_print("=" * 80)
    log_print("COMPREHENSIVE BACKTEST - MA & RSI STRATEGIES")
    log_print("=" * 80)
    log_print(f"Start time: {datetime.now()}")
    log_print(f"Period: Last 30 days")
    log_print(f"Data: Hourly bars (matching live system)")
    log_print(f"Progress updates: Every 10 seconds")
    log_print("=" * 80)
    
    # Load hourly data once
    df_hourly = load_hourly_data(days_back=30)
    
    # Define test configurations
    ma_configs = [
        # Original set
        (6, 34),   # Current
        (5, 15),   # Very fast
        (8, 21),   # Fibonacci
        (10, 20),  # Balanced
        (10, 30),  # Medium
        (12, 26),  # MACD-like
        (15, 30),  # Medium-slow
        (20, 50),  # Classic
        
        # Additional MA combinations
        (3, 10),   # Ultra fast
        (5, 20),   # Fast
        (7, 14),   # Week-based
        (9, 21),   # Fibonacci variant
        (10, 40),  # Medium-wide
        (13, 26),  # MACD variant
        (15, 45),  # Wide
        (20, 40),  # Balanced wide
        (25, 50),  # Slow
        (30, 60),  # Very slow
        (50, 100), # Long term
        (50, 200), # Golden cross
    ]
    
    rsi_configs = [
        # RSI configurations (period, oversold, overbought)
        (14, 30, 70),  # Standard
        (14, 20, 80),  # Wider bands
        (14, 25, 75),  # Tighter bands
        (9, 30, 70),   # Faster
        (21, 30, 70),  # Slower
        (14, 35, 65),  # Conservative
        (7, 30, 70),   # Very fast
        (28, 30, 70),  # Very slow
        (14, 40, 60),  # Very conservative
        (14, 15, 85),  # Extreme bands
    ]
    
    total_tests = len(ma_configs) + len(rsi_configs)
    
    log_print(f"\nTotal tests to run: {total_tests}")
    log_print(f"  - MA strategies: {len(ma_configs)}")
    log_print(f"  - RSI strategies: {len(rsi_configs)}")
    log_print(f"Estimated time: {total_tests * 0.5:.0f}-{total_tests * 1.5:.0f} seconds")
    log_print("=" * 80)
    
    # Initialize progress tracking
    tracker = ProgressTracker(total_tests)
    results = []
    
    # Start progress monitor thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=progress_monitor, args=(tracker, stop_event))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    log_print("\nTesting MA strategies...")
    log_print("-" * 60)
    
    # Test MA strategies
    for ma_short, ma_long in ma_configs:
        strategy_name = f"MA {ma_short}/{ma_long}"
        tracker.start_test(strategy_name)
        
        try:
            result = backtest_ma_strategy(df_hourly, ma_short, ma_long)
            results.append(result)
            log_print(f"✓ {strategy_name:<15} Return: {result['return']:>6.2f}% | Trades: {result['trades']:>3} | MaxDD: {result['max_drawdown']:>5.1f}%")
        except Exception as e:
            log_print(f"✗ {strategy_name:<15} Error: {str(e)}")
        
        tracker.complete_test()
    
    log_print("\n\nTesting RSI strategies...")
    log_print("-" * 60)
    
    # Test RSI strategies
    for rsi_period, oversold, overbought in rsi_configs:
        strategy_name = f"RSI {rsi_period} ({oversold}/{overbought})"
        tracker.start_test(strategy_name)
        
        try:
            result = backtest_rsi_strategy(df_hourly, rsi_period, oversold, overbought)
            results.append(result)
            log_print(f"✓ {strategy_name:<20} Return: {result['return']:>6.2f}% | Trades: {result['trades']:>3} | MaxDD: {result['max_drawdown']:>5.1f}%")
        except Exception as e:
            log_print(f"✗ {strategy_name:<20} Error: {str(e)}")
        
        tracker.complete_test()
    
    # Stop monitor thread
    stop_event.set()
    
    # Sort results by return
    results.sort(key=lambda x: x['return'], reverse=True)
    
    # Display final results
    log_print("\n" + "=" * 80)
    log_print("FINAL RESULTS (sorted by return)")
    log_print("=" * 80)
    log_print(f"{'Rank':<5} {'Strategy':<25} {'Return':>8} {'Trades':>7} {'T/Day':>6} {'MaxDD':>7}")
    log_print("-" * 60)
    
    for i, r in enumerate(results[:20], 1):  # Top 20
        log_print(f"{i:<5} {r['strategy']:<25} {r['return']:>7.2f}% {r['trades']:>7} {r['trades_per_day']:>6.2f} {r['max_drawdown']:>6.1f}%")
    
    # Show category winners
    log_print("\n" + "=" * 80)
    log_print("CATEGORY WINNERS")
    log_print("=" * 80)
    
    # Best MA strategy
    ma_results = [r for r in results if r['strategy'].startswith('MA_')]
    if ma_results:
        best_ma = ma_results[0]
        log_print(f"🏆 Best MA Strategy: {best_ma['strategy']}")
        log_print(f"   Return: {best_ma['return']:.2f}% | Trades: {best_ma['trades']} | Max DD: {best_ma['max_drawdown']:.1f}%")
    
    # Best RSI strategy
    rsi_results = [r for r in results if r['strategy'].startswith('RSI_')]
    if rsi_results:
        best_rsi = rsi_results[0]
        log_print(f"\n🏆 Best RSI Strategy: {best_rsi['strategy']}")
        log_print(f"   Return: {best_rsi['return']:.2f}% | Trades: {best_rsi['trades']} | Max DD: {best_rsi['max_drawdown']:.1f}%")
    
    # Overall winner
    if results:
        winner = results[0]
        log_print(f"\n🏆 OVERALL WINNER: {winner['strategy']}")
        log_print(f"   Return: {winner['return']:.2f}%")
        log_print(f"   Trades: {winner['trades']} ({winner['trades_per_day']:.2f} per day)")
        log_print(f"   Max Drawdown: {winner['max_drawdown']:.1f}%")
        
        # Save all results to JSON
        with open('comprehensive_backtest_results.json', 'w') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'period_days': 30,
                'data_type': 'hourly_bars',
                'results': results,
                'winner': winner
            }, f, indent=2)
        
        log_print(f"\n✅ Saved all results to: comprehensive_backtest_results.json")
    
    # Summary
    elapsed = time.time() - tracker.start_time
    log_print(f"\n⏱  Total time: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
    log_print(f"📊 Tests completed: {tracker.completed_tests}/{tracker.total_tests}")
    
    log_print("\n" + "=" * 80)
    log_print("NOTE: All results based on HOURLY bars matching the live system")
    log_print("=" * 80)
    
    log_file.close()
    print("\n✅ Full log saved to: backtest_results_log.txt")


if __name__ == "__main__":
    main()