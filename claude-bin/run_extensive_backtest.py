#!/usr/bin/env python3
"""
Extensive backtest with many more MA combinations and hybrid strategies
Designed to run for ~10 minutes to find robust, frequent-trading strategies
"""
import time
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import json
import os
import itertools

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from data.loader import parse_log_file

# Create log file
log_file = open('extensive_backtest_log.txt', 'w')

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
        self.last_update = time.time()
        
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
                
                tests_per_minute = (self.completed_tests / elapsed) * 60 if elapsed > 0 else 0
                
                return {
                    'current': self.current_test,
                    'completed': self.completed_tests,
                    'total': self.total_tests,
                    'elapsed': elapsed,
                    'eta': eta_time.strftime('%H:%M:%S'),
                    'avg_time': avg_time,
                    'tests_per_minute': tests_per_minute,
                    'percent': (self.completed_tests / self.total_tests) * 100
                }
            else:
                return {
                    'current': self.current_test,
                    'completed': 0,
                    'total': self.total_tests,
                    'elapsed': elapsed,
                    'eta': 'Calculating...',
                    'avg_time': 0,
                    'tests_per_minute': 0,
                    'percent': 0
                }
                
    def should_update(self):
        """Check if we should print an update (every 10 seconds)"""
        with self.lock:
            now = time.time()
            if now - self.last_update >= 10:
                self.last_update = now
                return True
            return False


def progress_monitor(tracker, stop_event):
    """Background thread to print progress every 10 seconds"""
    while not stop_event.is_set():
        time.sleep(1)  # Check every second
        if tracker.should_update() and not stop_event.is_set():
            status = tracker.get_status()
            log_print(f"\n📊 PROGRESS UPDATE [{datetime.now().strftime('%H:%M:%S')}]")
            log_print(f"   Current test: {status['current']}")
            log_print(f"   Progress: {status['completed']}/{status['total']} ({status['percent']:.1f}%)")
            log_print(f"   Speed: {status['tests_per_minute']:.1f} tests/minute")
            log_print(f"   Elapsed: {status['elapsed']/60:.1f} minutes")
            log_print(f"   ETA: {status['eta']}")
            log_print("-" * 60)


def load_hourly_data(days_back=30):
    """Load and resample data to hourly bars"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    log_print(f"Loading {days_back} days of data...")
    df = parse_log_file('btcusd.log', start_date=start_date, end_date=end_date)
    
    if df is None or len(df) == 0:
        raise ValueError("No data loaded")
    
    # Ensure datetime index
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
    
    log_print(f"Loaded {len(df):,} ticks")
    
    # Resample to hourly bars
    log_print("Resampling to hourly bars...")
    df_hourly = df.resample('1H').agg({
        'price': ['first', 'max', 'min', 'last'],
        'amount': 'sum'
    })
    df_hourly.columns = ['open', 'high', 'low', 'close', 'volume']
    df_hourly = df_hourly.dropna()
    
    log_print(f"Created {len(df_hourly):,} hourly bars")
    return df_hourly


def calculate_rsi(df, period=14):
    """Calculate RSI indicator"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def backtest_ma_strategy(df_hourly, ma_short, ma_long, min_trades=10):
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
    
    result = simulate_trading(df, f"MA_{ma_short}_{ma_long}")
    
    # Add minimum trade filter
    if result['trades'] < min_trades:
        result['filtered'] = True
        
    return result


def backtest_rsi_strategy(df_hourly, rsi_period=14, oversold=30, overbought=70, min_trades=10):
    """Backtest RSI strategy"""
    df = df_hourly.copy()
    
    # Calculate RSI
    df['RSI'] = calculate_rsi(df, rsi_period)
    
    # Skip NaN values
    df = df.dropna()
    
    # Generate signals
    df['signal'] = 0
    df.loc[df['RSI'] < oversold, 'signal'] = 1
    df.loc[df['RSI'] > overbought, 'signal'] = -1
    
    # Forward fill signals
    df['signal'] = df['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    result = simulate_trading(df, f"RSI_{rsi_period}_{oversold}_{overbought}")
    
    # Add minimum trade filter
    if result['trades'] < min_trades:
        result['filtered'] = True
        
    return result


def backtest_hybrid_strategy(df_hourly, ma_short, ma_long, rsi_period=14, rsi_threshold=30, min_trades=10):
    """Backtest hybrid MA + RSI confirmation strategy"""
    df = df_hourly.copy()
    
    # Calculate indicators
    df['MA_short'] = df['close'].rolling(window=ma_short).mean()
    df['MA_long'] = df['close'].rolling(window=ma_long).mean()
    df['RSI'] = calculate_rsi(df, rsi_period)
    
    # Skip NaN values
    df = df.dropna()
    
    # Generate signals with RSI confirmation
    df['ma_signal'] = 0
    df.loc[df['MA_short'] > df['MA_long'], 'ma_signal'] = 1
    df.loc[df['MA_short'] < df['MA_long'], 'ma_signal'] = -1
    
    # Only take signals when RSI confirms
    df['signal'] = 0
    # Long when MA bullish AND RSI not overbought
    df.loc[(df['ma_signal'] == 1) & (df['RSI'] < (100 - rsi_threshold)), 'signal'] = 1
    # Short when MA bearish AND RSI not oversold
    df.loc[(df['ma_signal'] == -1) & (df['RSI'] > rsi_threshold), 'signal'] = -1
    
    result = simulate_trading(df, f"Hybrid_MA{ma_short}_{ma_long}_RSI{rsi_period}_{rsi_threshold}")
    
    # Add minimum trade filter
    if result['trades'] < min_trades:
        result['filtered'] = True
        
    return result


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
    
    # Calculate Sharpe ratio (simplified)
    if len(trades) > 2:
        returns = []
        for i in range(1, len(trades)):
            if trades[i]['type'] == 'SELL' and trades[i-1]['type'] == 'BUY':
                ret = (trades[i]['price'] - trades[i-1]['price']) / trades[i-1]['price']
                returns.append(ret)
        
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return / std_return) * np.sqrt(365) if std_return > 0 else 0
        else:
            sharpe = 0
    else:
        sharpe = 0
    
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
        'trades_per_day': len(trades) / 30.0,
        'sharpe': sharpe,
        'filtered': False
    }


def generate_ma_combinations():
    """Generate comprehensive MA combinations"""
    # Short MA: 3-50
    short_mas = list(range(3, 51, 1))
    
    # Long MA: must be at least 2x short MA, up to 200
    combinations = []
    
    for short in short_mas:
        # Long MA from 2x short to min(200, 10x short)
        min_long = short * 2
        max_long = min(200, short * 10)
        
        # Sample long MAs with varying density
        if short <= 10:
            # For very short MAs, test more long values
            long_step = 2
        elif short <= 20:
            long_step = 3
        elif short <= 30:
            long_step = 5
        else:
            long_step = 10
            
        for long in range(min_long, max_long + 1, long_step):
            combinations.append((short, long))
    
    return combinations


def main():
    log_print("=" * 80)
    log_print("EXTENSIVE BACKTEST - SEARCHING FOR ROBUST STRATEGIES")
    log_print("=" * 80)
    log_print(f"Start time: {datetime.now()}")
    log_print(f"Target runtime: ~10 minutes")
    log_print(f"Minimum trades filter: 10 trades in 30 days")
    log_print("=" * 80)
    
    # Load hourly data once
    df_hourly = load_hourly_data(days_back=30)
    
    # Generate test configurations
    log_print("\nGenerating test configurations...")
    
    # MA combinations
    ma_configs = generate_ma_combinations()
    log_print(f"MA combinations: {len(ma_configs)}")
    
    # RSI configurations - more variations
    rsi_configs = []
    for period in [7, 9, 14, 21, 28]:
        for oversold in range(20, 45, 5):
            for overbought in range(55, 85, 5):
                if overbought - oversold >= 20:  # Ensure reasonable band
                    rsi_configs.append((period, oversold, overbought))
    
    log_print(f"RSI combinations: {len(rsi_configs)}")
    
    # Hybrid configurations - select subset to keep runtime reasonable
    hybrid_configs = []
    # Test promising MA combinations with RSI confirmation
    promising_mas = [(5, 15), (6, 34), (8, 21), (10, 30), (12, 26), (15, 45)]
    for ma_short, ma_long in promising_mas:
        for rsi_period in [9, 14, 21]:
            for rsi_threshold in [25, 30, 35, 40]:
                hybrid_configs.append((ma_short, ma_long, rsi_period, rsi_threshold))
    
    log_print(f"Hybrid combinations: {len(hybrid_configs)}")
    
    total_tests = len(ma_configs) + len(rsi_configs) + len(hybrid_configs)
    log_print(f"\nTotal tests to run: {total_tests:,}")
    log_print(f"Estimated time at 50 tests/minute: {total_tests/50:.1f} minutes")
    log_print("=" * 80)
    
    # Initialize progress tracking
    tracker = ProgressTracker(total_tests)
    results = []
    
    # Start progress monitor thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=progress_monitor, args=(tracker, stop_event))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    log_print("\nPhase 1: Testing MA strategies...")
    log_print("-" * 60)
    
    # Test MA strategies
    ma_results_summary = {'tested': 0, 'successful': 0, 'filtered': 0}
    
    for ma_short, ma_long in ma_configs:
        strategy_name = f"MA {ma_short}/{ma_long}"
        tracker.start_test(strategy_name)
        
        try:
            result = backtest_ma_strategy(df_hourly, ma_short, ma_long, min_trades=10)
            results.append(result)
            ma_results_summary['tested'] += 1
            
            if result.get('filtered', False):
                ma_results_summary['filtered'] += 1
            else:
                ma_results_summary['successful'] += 1
                # Only log promising results
                if result['return'] > 2 and result['trades'] >= 15:
                    log_print(f"  💎 {strategy_name}: Return={result['return']:.1f}%, Trades={result['trades']}, Sharpe={result['sharpe']:.2f}")
                    
        except Exception as e:
            log_print(f"  ❌ {strategy_name}: Error: {str(e)}")
        
        tracker.complete_test()
    
    log_print(f"\nMA Results: {ma_results_summary['successful']} successful, {ma_results_summary['filtered']} filtered (too few trades)")
    
    log_print("\n\nPhase 2: Testing RSI strategies...")
    log_print("-" * 60)
    
    # Test RSI strategies
    rsi_results_summary = {'tested': 0, 'successful': 0, 'filtered': 0}
    
    for rsi_period, oversold, overbought in rsi_configs:
        strategy_name = f"RSI {rsi_period} ({oversold}/{overbought})"
        tracker.start_test(strategy_name)
        
        try:
            result = backtest_rsi_strategy(df_hourly, rsi_period, oversold, overbought, min_trades=10)
            results.append(result)
            rsi_results_summary['tested'] += 1
            
            if result.get('filtered', False):
                rsi_results_summary['filtered'] += 1
            else:
                rsi_results_summary['successful'] += 1
                # Only log promising results
                if result['return'] > 2 and result['trades'] >= 15:
                    log_print(f"  💎 {strategy_name}: Return={result['return']:.1f}%, Trades={result['trades']}")
                    
        except Exception as e:
            log_print(f"  ❌ {strategy_name}: Error: {str(e)}")
        
        tracker.complete_test()
    
    log_print(f"\nRSI Results: {rsi_results_summary['successful']} successful, {rsi_results_summary['filtered']} filtered")
    
    log_print("\n\nPhase 3: Testing Hybrid strategies...")
    log_print("-" * 60)
    
    # Test Hybrid strategies
    hybrid_results_summary = {'tested': 0, 'successful': 0, 'filtered': 0}
    
    for ma_short, ma_long, rsi_period, rsi_threshold in hybrid_configs:
        strategy_name = f"Hybrid MA{ma_short}/{ma_long} RSI{rsi_period}@{rsi_threshold}"
        tracker.start_test(strategy_name)
        
        try:
            result = backtest_hybrid_strategy(df_hourly, ma_short, ma_long, rsi_period, rsi_threshold, min_trades=10)
            results.append(result)
            hybrid_results_summary['tested'] += 1
            
            if result.get('filtered', False):
                hybrid_results_summary['filtered'] += 1
            else:
                hybrid_results_summary['successful'] += 1
                # Only log promising results
                if result['return'] > 2 and result['trades'] >= 15:
                    log_print(f"  💎 {strategy_name}: Return={result['return']:.1f}%, Trades={result['trades']}")
                    
        except Exception as e:
            log_print(f"  ❌ {strategy_name}: Error: {str(e)}")
        
        tracker.complete_test()
    
    log_print(f"\nHybrid Results: {hybrid_results_summary['successful']} successful, {hybrid_results_summary['filtered']} filtered")
    
    # Stop monitor thread
    stop_event.set()
    
    # Filter out strategies with too few trades
    valid_results = [r for r in results if not r.get('filtered', False)]
    
    # Sort by return
    valid_results.sort(key=lambda x: x['return'], reverse=True)
    
    # Also sort by Sharpe ratio
    valid_results_by_sharpe = sorted(valid_results, key=lambda x: x['sharpe'], reverse=True)
    
    # Display final results
    log_print("\n" + "=" * 80)
    log_print("TOP 30 STRATEGIES BY RETURN (min 10 trades)")
    log_print("=" * 80)
    log_print(f"{'Rank':<5} {'Strategy':<35} {'Return':>8} {'Trades':>7} {'T/Day':>6} {'MaxDD':>7} {'Sharpe':>7}")
    log_print("-" * 75)
    
    for i, r in enumerate(valid_results[:30], 1):
        log_print(f"{i:<5} {r['strategy']:<35} {r['return']:>7.2f}% {r['trades']:>7} {r['trades_per_day']:>6.2f} {r['max_drawdown']:>6.1f}% {r['sharpe']:>7.2f}")
    
    # Show best by Sharpe ratio too
    log_print("\n" + "=" * 80)
    log_print("TOP 10 BY SHARPE RATIO (risk-adjusted returns)")
    log_print("=" * 80)
    
    for i, r in enumerate(valid_results_by_sharpe[:10], 1):
        log_print(f"{i:<5} {r['strategy']:<35} Sharpe={r['sharpe']:>6.2f} Return={r['return']:>6.2f}% Trades={r['trades']:>3}")
    
    # Find best frequent traders (>20 trades)
    frequent_traders = [r for r in valid_results if r['trades'] >= 20]
    frequent_traders.sort(key=lambda x: x['return'], reverse=True)
    
    log_print("\n" + "=" * 80)
    log_print("BEST FREQUENT TRADERS (20+ trades)")
    log_print("=" * 80)
    
    for i, r in enumerate(frequent_traders[:10], 1):
        log_print(f"{i:<5} {r['strategy']:<35} {r['return']:>7.2f}% {r['trades']:>7} {r['trades_per_day']:>6.2f}")
    
    # Save comprehensive results
    with open('extensive_backtest_results.json', 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'period_days': 30,
            'total_tests': total_tests,
            'valid_results': len(valid_results),
            'filtered_results': len(results) - len(valid_results),
            'top_by_return': valid_results[:50],
            'top_by_sharpe': valid_results_by_sharpe[:50],
            'best_frequent_traders': frequent_traders[:20]
        }, f, indent=2)
    
    log_print(f"\n✅ Saved detailed results to: extensive_backtest_results.json")
    
    # Summary statistics
    elapsed = time.time() - tracker.start_time
    log_print(f"\n" + "=" * 80)
    log_print("SUMMARY")
    log_print("=" * 80)
    log_print(f"Total runtime: {elapsed/60:.1f} minutes")
    log_print(f"Tests completed: {tracker.completed_tests:,}")
    log_print(f"Valid strategies (10+ trades): {len(valid_results):,}")
    log_print(f"Tests per minute: {tracker.completed_tests/(elapsed/60):.1f}")
    
    if valid_results:
        winner = valid_results[0]
        log_print(f"\n🏆 OVERALL WINNER: {winner['strategy']}")
        log_print(f"   Return: {winner['return']:.2f}%")
        log_print(f"   Trades: {winner['trades']} ({winner['trades_per_day']:.2f} per day)")
        log_print(f"   Max Drawdown: {winner['max_drawdown']:.1f}%")
        log_print(f"   Sharpe Ratio: {winner['sharpe']:.2f}")
    
    log_print("\n" + "=" * 80)
    log_print("NOTE: All results based on HOURLY bars matching the live system")
    log_print("=" * 80)
    
    log_file.close()
    print("\n✅ Full log saved to: extensive_backtest_log.txt")


if __name__ == "__main__":
    main()