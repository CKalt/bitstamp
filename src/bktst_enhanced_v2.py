#!/usr/bin/env python3
"""
Enhanced Bitcoin Trading System Backtester - Version 2
Properly tests AdaptiveMultiStrategy with correct initialization and flow
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_log_analyzer import parse_log_file, analyze_data
from utils.helpers import ensure_datetime_index
from indicators.technical_indicators import (
    add_moving_averages, calculate_rsi, calculate_bollinger_bands, 
    calculate_macd, calculate_adaptive_vwma
)


class SimpleAdaptiveStrategy:
    """Simplified version of AdaptiveMultiStrategy for backtesting"""
    
    def __init__(self, short_window=10, long_window=46):
        self.short_window = short_window
        self.long_window = long_window
        self.regime_switch_threshold = 0.40
        self.signal_confirmation_bars = 2
        self.min_trade_gap_minutes = 15
        self.whipsaw_threshold = 8.0
        self.regime_lookback = 100
        
        # Initialize state
        self.current_regime = 'trending'
        self.last_trade_time = None
        self.signal_history = []
        self.position = 0
        
    def detect_market_regime(self, df):
        """Simplified regime detection"""
        if len(df) < self.regime_lookback:
            return 'trending', 0.5, {}
            
        # Calculate indicators
        df = add_moving_averages(df, self.short_window, self.long_window, price_col='close')
        df['ma_short'] = df[f'MA_{self.short_window}']
        df['ma_long'] = df[f'MA_{self.long_window}']
        df_with_rsi = calculate_rsi(df, price_col='close')
        df['rsi'] = df_with_rsi['RSI']
        
        # Recent data for analysis
        recent_df = df.iloc[-self.regime_lookback:].copy()
        
        # Calculate volatility
        returns = recent_df['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Calculate trend strength
        price_change = (recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]) / recent_df['close'].iloc[0]
        trend_strength = abs(price_change)
        
        # Count MA crossovers (whipsaws)
        ma_diff = recent_df['ma_short'] - recent_df['ma_long']
        crossovers = ((ma_diff > 0) != (ma_diff.shift(1) > 0)).sum()
        whipsaw_ratio = crossovers / len(recent_df) * 100
        
        # Determine regime
        if volatility > 0.03:
            regime = 'volatile'
            confidence = 0.7
        elif whipsaw_ratio > self.whipsaw_threshold:
            regime = 'ranging'
            confidence = 0.6
        else:
            regime = 'trending'
            confidence = 0.8
            
        metrics = {
            'volatility': float(volatility),
            'trend_strength': float(trend_strength),
            'whipsaw_ratio': float(whipsaw_ratio)
        }
        
        return regime, confidence, metrics
        
    def generate_signal(self, df, regime):
        """Generate trading signal based on regime"""
        if len(df) < self.long_window + 10:
            return 0, "Insufficient data"
            
        # Calculate indicators
        df = add_moving_averages(df, self.short_window, self.long_window, price_col='close')
        df['ma_short'] = df[f'MA_{self.short_window}']
        df['ma_long'] = df[f'MA_{self.long_window}']
        
        # Get current values
        current_ma_short = df['ma_short'].iloc[-1]
        current_ma_long = df['ma_long'].iloc[-1]
        prev_ma_short = df['ma_short'].iloc[-2]
        prev_ma_long = df['ma_long'].iloc[-2]
        
        if regime == 'trending':
            # MA crossover strategy
            if prev_ma_short <= prev_ma_long and current_ma_short > current_ma_long:
                return 1, "MA crossover - BUY"
            elif prev_ma_short >= prev_ma_long and current_ma_short < current_ma_long:
                return -1, "MA crossover - SELL"
                
        elif regime == 'ranging':
            # Mean reversion with RSI
            df_with_rsi = calculate_rsi(df, price_col='close')
            df['rsi'] = df_with_rsi['RSI']
            current_rsi = df['rsi'].iloc[-1]
            
            if current_rsi < 30:
                return 1, f"Oversold RSI={current_rsi:.1f}"
            elif current_rsi > 70:
                return -1, f"Overbought RSI={current_rsi:.1f}"
                
        elif regime == 'volatile':
            # Breakout strategy
            df = calculate_bollinger_bands(df, price_col='close')
            current_price = df['close'].iloc[-1]
            bb_upper = df['BB_Upper'].iloc[-1]
            bb_lower = df['BB_Lower'].iloc[-1]
            
            if current_price > bb_upper:
                return 1, "Breakout above BB"
            elif current_price < bb_lower:
                return -1, "Breakout below BB"
                
        return 0, "No signal"
        
    def check_signal_confirmation(self, signal, current_time):
        """Check if signal is confirmed"""
        # Add to history
        self.signal_history.append((current_time, signal))
        
        # Keep only recent signals
        cutoff_time = current_time - timedelta(hours=self.signal_confirmation_bars)
        self.signal_history = [(t, s) for t, s in self.signal_history if t >= cutoff_time]
        
        # Check if we have enough signals
        if len(self.signal_history) < self.signal_confirmation_bars:
            return False
            
        # Check if all recent signals agree
        recent_signals = [s for _, s in self.signal_history[-self.signal_confirmation_bars:]]
        if all(s == signal and s != 0 for s in recent_signals):
            return True
            
        return False
        
    def can_trade(self, current_time):
        """Check if we can trade based on time gap"""
        if self.last_trade_time is None:
            return True
            
        time_since_last = (current_time - self.last_trade_time).total_seconds() / 60
        return time_since_last >= self.min_trade_gap_minutes


class AdaptiveStrategyBacktester:
    """Enhanced backtester specifically for AdaptiveMultiStrategy"""
    
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.backtest_settings = config.get('backtest_settings', {})
        
    def simulate(self, params):
        """Run simplified adaptive strategy simulation"""
        try:
            # Initialize strategy
            strategy = SimpleAdaptiveStrategy(
                short_window=params['short_window'],
                long_window=params['long_window']
            )
            
            # Set parameters
            strategy.regime_switch_threshold = params.get('regime_switch_threshold', 0.40)
            strategy.min_trade_gap_minutes = params.get('min_trade_gap_minutes', 15)
            strategy.signal_confirmation_bars = params.get('signal_confirmation_bars', 2)
            strategy.whipsaw_threshold = params.get('whipsaw_threshold', 8.0)
            
            # Initialize tracking variables
            balance = self.backtest_settings.get('initial_balance', 10000)
            position = 0  # 0: no position, 1: long, -1: short
            btc_amount = 0
            trades = []
            regime_history = []
            
            # Track performance by regime
            regime_performance = {
                'trending': {'trades': 0, 'wins': 0, 'total_return': 0, 'time_in_regime': 0},
                'ranging': {'trades': 0, 'wins': 0, 'total_return': 0, 'time_in_regime': 0},
                'volatile': {'trades': 0, 'wins': 0, 'total_return': 0, 'time_in_regime': 0}
            }
            
            # Simulate trading
            df = self.df.copy()
            df = ensure_datetime_index(df)
            
            # Need minimum data for strategy
            min_periods = max(params['long_window'], strategy.regime_lookback) + 10
            if len(df) < min_periods:
                return {'error': 'Insufficient data for backtesting'}
            
            # Main simulation loop
            for i in range(min_periods, len(df)):
                current_time = df.index[i]
                current_price = df['price'].iloc[i]
                
                # Get data slice up to current point
                data_slice = df.iloc[:i+1].copy()
                
                # Detect market regime periodically (every hour in the data)
                if i % 60 == 0:
                    new_regime, confidence, metrics = strategy.detect_market_regime(data_slice)
                    
                    # Check if regime should switch
                    if strategy.current_regime != new_regime and confidence >= strategy.regime_switch_threshold:
                        strategy.current_regime = new_regime
                        regime_history.append({
                            'timestamp': current_time,
                            'regime': new_regime,
                            'confidence': confidence,
                            'metrics': metrics
                        })
                
                # Track time in regime
                regime_performance[strategy.current_regime]['time_in_regime'] += 1
                
                # Generate signal based on current regime
                signal, reason = strategy.generate_signal(data_slice, strategy.current_regime)
                
                # Check signal confirmation and trade gap
                if signal != 0 and signal != position:
                    if strategy.check_signal_confirmation(signal, current_time) and strategy.can_trade(current_time):
                        # Calculate trade metrics
                        trade_fee = abs(balance * self.backtest_settings.get('trading_fee', 0.001))
                        slippage = current_price * self.backtest_settings.get('slippage', 0.0005)
                        
                        profit = 0
                        # Execute trade
                        if signal == 1 and position <= 0:  # Buy signal
                            if position == -1:  # Close short
                                profit = btc_amount * current_price - balance
                                balance = btc_amount * current_price - trade_fee
                                btc_amount = 0
                            # Open long
                            btc_amount = (balance - trade_fee) / (current_price + slippage)
                            balance = 0
                            
                            trades.append({
                                'timestamp': current_time,
                                'type': 'BUY',
                                'price': current_price + slippage,
                                'amount': btc_amount,
                                'balance': balance,
                                'regime': strategy.current_regime,
                                'reason': reason,
                                'profit': profit if position == -1 else 0
                            })
                            
                            position = 1
                            regime_performance[strategy.current_regime]['trades'] += 1
                            if position == -1 and profit > 0:
                                regime_performance[strategy.current_regime]['wins'] += 1
                                regime_performance[strategy.current_regime]['total_return'] += profit / balance
                        
                        elif signal == -1 and position >= 0:  # Sell signal
                            if position == 1:  # Close long
                                balance = btc_amount * (current_price - slippage) - trade_fee
                                profit = balance - self.backtest_settings.get('initial_balance', 10000)
                                btc_amount = 0
                            
                            trades.append({
                                'timestamp': current_time,
                                'type': 'SELL',
                                'price': current_price - slippage,
                                'amount': btc_amount,
                                'balance': balance,
                                'regime': strategy.current_regime,
                                'reason': reason,
                                'profit': profit if position == 1 else 0
                            })
                            
                            position = -1
                            regime_performance[strategy.current_regime]['trades'] += 1
                            if position == 1 and profit > 0:
                                regime_performance[strategy.current_regime]['wins'] += 1
                                regime_performance[strategy.current_regime]['total_return'] += profit / self.backtest_settings.get('initial_balance', 10000) * 100
                        
                        # Update last trade time
                        strategy.last_trade_time = current_time
                        strategy.position = position
            
            # Calculate final balance
            if position == 1:
                balance = btc_amount * df['price'].iloc[-1]
            
            # Calculate performance metrics
            total_return = (balance - self.backtest_settings.get('initial_balance', 10000)) / self.backtest_settings.get('initial_balance', 10000) * 100
            
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df['returns'] = trades_df['profit'] / self.backtest_settings.get('initial_balance', 10000)
                
                # Calculate metrics
                sharpe_ratio = trades_df['returns'].mean() / trades_df['returns'].std() * np.sqrt(252) if trades_df['returns'].std() > 0 else 0
                win_rate = len(trades_df[trades_df['profit'] > 0]) / len(trades_df) * 100
                
                # Calculate max drawdown
                cumulative_returns = (1 + trades_df['returns']).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min() * 100
            else:
                sharpe_ratio = 0
                win_rate = 0
                max_drawdown = 0
            
            # Calculate regime performance metrics
            for regime in regime_performance:
                if regime_performance[regime]['trades'] > 0:
                    regime_performance[regime]['win_rate'] = (
                        regime_performance[regime]['wins'] / regime_performance[regime]['trades'] * 100
                    )
                else:
                    regime_performance[regime]['win_rate'] = 0
                    
                if regime_performance[regime]['time_in_regime'] > 0:
                    regime_performance[regime]['time_percentage'] = (
                        regime_performance[regime]['time_in_regime'] / (len(df) - min_periods) * 100
                    )
                else:
                    regime_performance[regime]['time_percentage'] = 0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_trades': len(trades),
                'win_rate': win_rate,
                'balance': balance,
                'trades': trades,
                'regime_history': regime_history,
                'regime_performance': regime_performance,
                'parameters': params
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def optimize(self, param_ranges):
        """Optimize strategy parameters"""
        results = []
        
        # Generate all parameter combinations
        param_combinations = []
        for short_window in param_ranges['short_window_range']:
            for long_window in param_ranges['long_window_range']:
                if short_window >= long_window:
                    continue
                for regime_threshold in param_ranges['regime_switch_threshold_range']:
                    for confirmation_bars in param_ranges['signal_confirmation_bars_range']:
                        for trade_gap in param_ranges['min_trade_gap_minutes_range']:
                            for whipsaw in param_ranges['whipsaw_threshold_range']:
                                param_combinations.append({
                                    'short_window': short_window,
                                    'long_window': long_window,
                                    'regime_switch_threshold': regime_threshold,
                                    'signal_confirmation_bars': confirmation_bars,
                                    'min_trade_gap_minutes': trade_gap,
                                    'whipsaw_threshold': whipsaw
                                })
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        # Test each combination
        for i, params in enumerate(param_combinations):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(param_combinations)} combinations tested")
            
            result = self.simulate(params)
            if 'error' not in result:
                results.append(result)
        
        # Sort by total return
        results.sort(key=lambda x: x['total_return'], reverse=True)
        
        return results


def run_enhanced_trading_system(df, config, include_adaptive=True):
    """Main function to run enhanced backtesting"""
    
    print("\nRunning enhanced trading system...")
    print("Starting run_trading_system function...")
    
    # Ensure datetime index
    df = ensure_datetime_index(df)
    print(f"DataFrame shape after ensuring datetime index: {df.shape}")
    
    # Resample data
    df_high_freq = df.resample(config['high_frequency']).agg({
        'price': 'last',
        'amount': 'sum',
        'volume': 'sum'
    }).dropna()
    print(f"Resampling data to higher timeframe ({config['high_frequency']})...")
    print(f"Higher timeframe DataFrame shape: {df_high_freq.shape}")
    
    df_low_freq = df.resample(config['low_frequency']).agg({
        'price': 'last',
        'amount': 'sum',
        'volume': 'sum'
    }).dropna()
    print(f"Resampling data to lower timeframe ({config['low_frequency']})...")
    print(f"Lower timeframe DataFrame shape: {df_low_freq.shape}")
    
    # Run standard strategies (existing code)
    all_results = []
    
    # ... [Keep existing strategy testing code here] ...
    
    # Run AdaptiveMultiStrategy optimization
    if include_adaptive:
        print("\n" + "="*80)
        print("Running AdaptiveMultiStrategy Optimization...")
        print("="*80)
        
        # Prepare data for adaptive strategy
        df_hourly = df.resample('1H').agg({
            'price': ['first', 'max', 'min', 'last'],
            'amount': 'sum',
            'volume': 'sum'
        }).dropna()
        # Flatten multi-level columns and create OHLC
        df_hourly.columns = ['open', 'high', 'low', 'close', 'amount', 'volume']
        df_hourly['price'] = df_hourly['close']  # Keep price column for compatibility
        
        # Initialize adaptive backtester
        adaptive_backtester = AdaptiveStrategyBacktester(df_hourly, config)
        
        # Define parameter ranges for optimization
        param_ranges = {
            'short_window_range': [8, 10, 12, 15],
            'long_window_range': [40, 46, 50, 60],
            'regime_switch_threshold_range': [0.35, 0.40, 0.45, 0.50],
            'signal_confirmation_bars_range': [1, 2, 3],
            'min_trade_gap_minutes_range': [10, 15, 20, 30],
            'whipsaw_threshold_range': [6.0, 8.0, 10.0]
        }
        
        # Run optimization
        adaptive_results = adaptive_backtester.optimize(param_ranges)
        
        # Convert results to standard format
        for result in adaptive_results[:10]:  # Take top 10 results
            if 'error' not in result:
                params = result['parameters']
                all_results.append({
                    'Strategy': 'AdaptiveMulti',
                    'Frequency': '1H',
                    'Total Return (%)': result['total_return'],
                    'Sharpe Ratio': result['sharpe_ratio'],
                    'Max Drawdown (%)': result['max_drawdown'],
                    'Number of Trades': result['num_trades'],
                    'Win Rate (%)': result['win_rate'],
                    'Short Window': params['short_window'],
                    'Long Window': params['long_window'],
                    'regime_switch_threshold': params['regime_switch_threshold'],
                    'signal_confirmation_bars': params['signal_confirmation_bars'],
                    'min_trade_gap_minutes': params['min_trade_gap_minutes'],
                    'whipsaw_threshold': params['whipsaw_threshold']
                })
        
        # Save detailed adaptive strategy results
        if adaptive_results:
            best_result = adaptive_results[0]
            
            # Save optimization results
            optimization_df = pd.DataFrame([{
                'short_window': r['parameters']['short_window'],
                'long_window': r['parameters']['long_window'],
                'regime_switch_threshold': r['parameters']['regime_switch_threshold'],
                'signal_confirmation_bars': r['parameters']['signal_confirmation_bars'],
                'min_trade_gap_minutes': r['parameters']['min_trade_gap_minutes'],
                'whipsaw_threshold': r['parameters']['whipsaw_threshold'],
                'total_return': r['total_return'],
                'sharpe_ratio': r['sharpe_ratio'],
                'max_drawdown': r['max_drawdown'],
                'num_trades': r['num_trades'],
                'win_rate': r['win_rate']
            } for r in adaptive_results if 'error' not in r])
            
            optimization_df.to_csv('adaptive_strategy_optimization.csv', index=False)
            
            # Save detailed results
            detailed_results = {
                'best_parameters': best_result['parameters'],
                'performance_metrics': {
                    'total_return_pct': best_result['total_return'],
                    'sharpe_ratio': best_result['sharpe_ratio'],
                    'max_drawdown_pct': best_result['max_drawdown'],
                    'num_trades': best_result['num_trades'],
                    'win_rate': best_result['win_rate']
                },
                'regime_performance': best_result['regime_performance'],
                'regime_history': best_result['regime_history'][:10],  # Sample of regime changes
                'backtest_period': {
                    'start': str(df.index[0]),
                    'end': str(df.index[-1]),
                    'days': (df.index[-1] - df.index[0]).days
                }
            }
            
            with open('adaptive_strategy_detailed_results.json', 'w') as f:
                json.dump(detailed_results, f, indent=2, default=str)
    
    # Create comparison dataframe
    if all_results:
        comparison_df = pd.DataFrame(all_results)
        print("\n" + "="*80)
        print("STRATEGY COMPARISON RESULTS")
        print("="*80)
        print(comparison_df.to_string(index=False))
        
        # Save results
        comparison_df.to_csv("all_strategy_results_enhanced.csv", index=False)
        
        # Get top strategies
        top_strategies = comparison_df.nlargest(10, 'Total Return (%)')
        top_strategies.to_csv("strategy_comparison_enhanced.csv", index=False)
        
        # Create recommended configuration
        if not comparison_df.empty and comparison_df['Total Return (%)'].max() > 0:
            best_strategy = comparison_df.loc[comparison_df['Total Return (%)'].idxmax()]
            
            # Enhanced configuration format
            recommended_config = {
                "backtest_metadata": {
                    "test_period_start": str(df.index[0]),
                    "test_period_end": str(df.index[-1]),
                    "total_days": (df.index[-1] - df.index[0]).days,
                    "data_frequency": config['high_frequency'],
                    "backtest_timestamp": datetime.now().isoformat() + 'Z',
                    "backtester_version": "2.0"
                },
                "performance_metrics": {
                    "total_return_pct": float(best_strategy['Total Return (%)']),
                    "sharpe_ratio": float(best_strategy['Sharpe Ratio']),
                    "max_drawdown_pct": float(best_strategy['Max Drawdown (%)']),
                    "win_rate": float(best_strategy['Win Rate (%)']) if 'Win Rate (%)' in best_strategy else 50.0,
                    "num_trades": int(best_strategy['Number of Trades']),
                    "avg_trades_per_day": float(best_strategy['Number of Trades']) / ((df.index[-1] - df.index[0]).days or 1)
                },
                "optimal_parameters": {
                    "strategy": best_strategy['Strategy'],
                    "frequency": best_strategy['Frequency'],
                    "short_window": int(best_strategy['Short Window']) if 'Short Window' in best_strategy else 10,
                    "long_window": int(best_strategy['Long Window']) if 'Long Window' in best_strategy else 46
                },
                "validation_status": {
                    "backtest_passed": True,
                    "risk_limits_ok": float(best_strategy['Max Drawdown (%)']) < 20,
                    "ready_for_deployment": False  # Always require manual review
                }
            }
            
            # Add regime performance if AdaptiveMulti
            if best_strategy['Strategy'] == 'AdaptiveMulti' and adaptive_results:
                best_adaptive = adaptive_results[0]
                recommended_config["regime_performance"] = best_adaptive['regime_performance']
                recommended_config["optimal_parameters"].update({
                    "regime_switch_threshold": float(best_strategy['regime_switch_threshold']),
                    "signal_confirmation_bars": int(best_strategy['signal_confirmation_bars']),
                    "min_trade_gap_minutes": int(best_strategy['min_trade_gap_minutes']),
                    "whipsaw_threshold": float(best_strategy['whipsaw_threshold'])
                })
            
            # Save recommended configuration
            with open('recommended_strategy.json', 'w') as f:
                json.dump(recommended_config, f, indent=2)
            
            print(f"\nBest strategy: {best_strategy['Strategy']} "
                  f"({best_strategy['Short Window']}, {best_strategy['Long Window']}) "
                  f"on {best_strategy['Frequency']} timeframe")
            print(f"Expected return: {best_strategy['Total Return (%)']}%")
        else:
            print("\nNo strategies met the criteria. Creating minimal configuration.")
            minimal_config = {
                "backtest_metadata": {
                    "test_period_start": str(df.index[0]),
                    "test_period_end": str(df.index[-1]),
                    "total_days": (df.index[-1] - df.index[0]).days,
                    "backtest_timestamp": datetime.now().isoformat() + 'Z',
                    "backtester_version": "2.0"
                },
                "performance_metrics": {
                    "total_return_pct": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown_pct": 0.0,
                    "win_rate": 0.0,
                    "num_trades": 0,
                    "avg_trades_per_day": 0.0
                },
                "optimal_parameters": {
                    "strategy": "MA",
                    "short_window": 10,
                    "long_window": 46
                },
                "validation_status": {
                    "backtest_passed": False,
                    "risk_limits_ok": False,
                    "ready_for_deployment": False
                }
            }
            with open('recommended_strategy.json', 'w') as f:
                json.dump(minimal_config, f, indent=2)
            print("\nMinimal configuration saved to 'recommended_strategy.json'")
    else:
        print("\nNo strategies to compare. Creating minimal configuration.")
        minimal_config = {
            "backtest_metadata": {
                "test_period_start": str(df.index[0]),
                "test_period_end": str(df.index[-1]),
                "total_days": (df.index[-1] - df.index[0]).days,
                "backtest_timestamp": datetime.now().isoformat() + 'Z'
            },
            "performance_metrics": {
                "total_return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0
            },
            "optimal_parameters": {
                "strategy": "MA",
                "short_window": 10,
                "long_window": 46
            },
            "validation_status": {
                "backtest_passed": False,
                "risk_limits_ok": False,
                "ready_for_deployment": False
            }
        }
        with open('recommended_strategy.json', 'w') as f:
            json.dump(minimal_config, f, indent=2)
        print("\nMinimal configuration saved to 'recommended_strategy.json'")
    
    print("\n" + "="*80)
    print("BACKTESTING COMPLETE")
    print("="*80)
    print("Results saved to:")
    print("  - all_strategy_results_enhanced.csv")
    print("  - strategy_comparison_enhanced.csv")
    print("  - recommended_strategy.json")
    print("  - adaptive_strategy_optimization.csv")
    print("  - adaptive_strategy_detailed_results.json")
    
    # Check if strategy meets risk criteria
    if comparison_df.empty or comparison_df['Total Return (%)'].max() <= 0:
        print("\n⚠️  WARNING: Strategy does not meet risk criteria for deployment")
        return None
    
    return comparison_df


def main():
    # Load configuration
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        print("Loaded config from 'config.json'")
    except FileNotFoundError:
        print("Config file not found. Using default configuration.")
        config = {
            "log_file": "btcusd.log",
            "start_window_days_back": 120,
            "end_window_days_back": 0,
            "high_frequency": "1H",
            "low_frequency": "15T",
            "backtest_settings": {
                "initial_balance": 10000,
                "trading_fee": 0.001,
                "slippage": 0.0005,
                "min_trade_amount": 10
            }
        }
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced Bitcoin Trading Strategy Backtester')
    parser.add_argument('--log-file', default=config.get('log_file', 'btcusd.log'),
                        help='Path to the log file')
    parser.add_argument('--start-window-days-back', type=int, 
                        default=config.get('start_window_days_back', 120),
                        help='Start of the analysis window (days back from now)')
    parser.add_argument('--end-window-days-back', type=int,
                        default=config.get('end_window_days_back', 0),
                        help='End of the analysis window (days back from now)')
    parser.add_argument('--high-frequency', default=config.get('high_frequency', '1H'),
                        help='High frequency for resampling (e.g., 1H, 4H)')
    parser.add_argument('--low-frequency', default=config.get('low_frequency', '15T'),
                        help='Low frequency for resampling (e.g., 15T, 30T)')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config['log_file'] = args.log_file
    config['start_window_days_back'] = args.start_window_days_back
    config['end_window_days_back'] = args.end_window_days_back
    config['high_frequency'] = args.high_frequency
    config['low_frequency'] = args.low_frequency
    
    # Calculate date range
    end_date = datetime.now() - timedelta(days=config['end_window_days_back'])
    start_date = datetime.now() - timedelta(days=config['start_window_days_back'])
    
    print(f"Analyzing data from {start_date} to {'present' if config['end_window_days_back'] == 0 else end_date}")
    
    # Parse log file
    df = parse_log_file(config['log_file'], start_date, end_date)
    
    if df.empty:
        print("No data found in the specified date range.")
        return
    
    print(f"Parsed {len(df)} trade events.")
    
    # Run basic analysis
    print("Starting data analysis...")
    analyze_data(df)
    
    # Run enhanced trading system
    comparison_df = run_enhanced_trading_system(df, config)


if __name__ == "__main__":
    main()