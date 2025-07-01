# src/bktst_enhanced.py
# Enhanced backtesting system that includes AdaptiveMultiStrategy testing

from utils.analysis import analyze_data, run_trading_system
from data.loader import create_metadata_file, parse_log_file
import argparse
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import traceback
from typing import Dict, List, Tuple, Any
import itertools

# Ensure that the 'src' directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

# Import the AdaptiveMultiStrategy and related components
from tdr_core.strategies import AdaptiveMultiStrategy, MACrossoverStrategy
from indicators.technical_indicators import (
    ensure_datetime_index, 
    add_moving_averages,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_macd
)
from backtesting.backtester import backtest
from optimization.optimizer import optimize_ma_parameters

###############################################################################
# Enhanced Configuration Loading
###############################################################################
def load_config(config_path='config.json'):
    """Load configuration with enhanced defaults for adaptive strategy testing."""
    if not os.path.exists(config_path):
        default_config = {
            "strategy_constraints": {
                "min_trades_per_day": 1,
                "max_trades_per_day": 5,
                "min_total_return": 0.0,
                "min_profit_per_trade": 0.0
            },
            "adaptive_strategy_params": {
                "regime_lookback": 50,
                "regime_switch_threshold": 0.40,
                "min_strategy_switch_minutes": 120,
                "min_trade_gap_minutes": 15,
                "signal_confirmation_bars": 2,
                "whipsaw_threshold": 8.0,
                "emergency_loss_threshold": -2000
            },
            "backtest_settings": {
                "initial_balance": 10000,
                "trading_fee": 0.001,
                "slippage": 0.0005,
                "enable_regime_tracking": True,
                "save_trade_history": True
            }
        }
        print(f"Config file '{config_path}' not found. Using enhanced default config.")
        return default_config
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from '{config_path}'")
        return config

###############################################################################
# AdaptiveMultiStrategy Backtesting
###############################################################################
class AdaptiveStrategyBacktester:
    """Specialized backtester for AdaptiveMultiStrategy."""
    
    def __init__(self, df: pd.DataFrame, config: Dict):
        self.df = df.copy()
        self.config = config
        self.backtest_settings = config.get('backtest_settings', {})
        self.results = {}
        
    def backtest_adaptive_strategy(self, params: Dict) -> Dict:
        """
        Run backtest for AdaptiveMultiStrategy with given parameters.
        
        Returns comprehensive results including regime performance breakdown.
        """
        try:
            # Create strategy instance with test parameters
            strategy = AdaptiveMultiStrategy(
                symbol='btcusd',
                short_window=params['short_window'],
                long_window=params['long_window']
            )
            
            # Override strategy parameters
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
            current_regime = 'TRENDING'  # Initialize with valid regime
            
            # Track performance by regime
            regime_performance = {
                'TRENDING': {'trades': 0, 'wins': 0, 'total_return': 0, 'time_in_regime': 0},
                'RANGING': {'trades': 0, 'wins': 0, 'total_return': 0, 'time_in_regime': 0},
                'VOLATILE': {'trades': 0, 'wins': 0, 'total_return': 0, 'time_in_regime': 0}
            }
            
            # Simulate trading
            df = self.df.copy()
            df = ensure_datetime_index(df)
            
            # Need minimum data for strategy
            min_periods = max(params['long_window'], strategy.regime_lookback) + 10
            if len(df) < min_periods:
                return {'error': 'Insufficient data for backtesting'}
            
            for i in range(min_periods, len(df)):
                current_time = df.index[i]
                current_price = df['price'].iloc[i]
                
                # Get data slice up to current point
                data_slice = df.iloc[:i+1].copy()
                # Fix column name mismatch
                data_slice['close'] = data_slice['price']
                
                # Detect market regime periodically (every hour in the data)
                if i % 60 == 0 or current_regime is None:
                    # Fix return format mismatch
                    new_regime, confidence, metrics = strategy.detect_market_regime(data_slice)
                    
                    # Check if regime should switch
                    if current_regime != new_regime and confidence >= strategy.regime_switch_threshold:
                        current_regime = new_regime
                        regime_history.append({
                            'timestamp': current_time,
                            'regime': new_regime,
                            'confidence': confidence,
                            'metrics': metrics
                        })
                
                # Track time in regime
                if current_regime:
                    regime_performance[current_regime]['time_in_regime'] += 1
                
                # Generate signals based on current regime
                signal = None
                if current_regime == 'TRENDING':
                    signal, reason = strategy.generate_trending_signal(data_slice)
                elif current_regime == 'RANGING':
                    signal, reason = strategy.generate_ranging_signal(data_slice)
                elif current_regime == 'VOLATILE':
                    signal, reason = strategy.generate_volatile_signal(data_slice)
                
                # Execute trades based on signals
                if signal and signal != position:
                    # Calculate trade metrics
                    trade_fee = abs(balance * self.backtest_settings.get('trading_fee', 0.001))
                    slippage = current_price * self.backtest_settings.get('slippage', 0.0005)
                    
                    profit = 0  # Initialize profit variable
                    # Execute trade
                    if signal == 1 and position <= 0:  # Buy signal
                        if position == -1:  # Close short
                            profit = btc_amount * current_price - balance
                            balance = btc_amount * current_price - trade_fee
                            btc_amount = 0
                        # Open long
                        btc_amount = (balance - trade_fee) / (current_price + slippage)
                        
                        trades.append({
                            'timestamp': current_time,
                            'type': 'BUY',
                            'price': current_price + slippage,
                            'amount': btc_amount,
                            'balance': balance,
                            'regime': current_regime,
                            'profit': profit if position == -1 else 0
                        })
                        
                        position = 1
                        if current_regime:
                            regime_performance[current_regime]['trades'] += 1
                            if position == -1 and profit > 0:
                                regime_performance[current_regime]['wins'] += 1
                                regime_performance[current_regime]['total_return'] += profit / balance
                    
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
                            'regime': current_regime,
                            'profit': profit if position == 1 else 0
                        })
                        
                        position = -1
                        if current_regime:
                            regime_performance[current_regime]['trades'] += 1
                            if position == 1 and profit > 0:
                                regime_performance[current_regime]['wins'] += 1
                                regime_performance[current_regime]['total_return'] += profit / self.backtest_settings.get('initial_balance', 10000)
            
            # Calculate final metrics
            final_balance = balance if position == 0 else (
                btc_amount * df['price'].iloc[-1] if position == 1 else balance
            )
            
            total_return = (final_balance - self.backtest_settings.get('initial_balance', 10000)) / self.backtest_settings.get('initial_balance', 10000)
            
            # Calculate additional metrics
            returns = []
            for i in range(1, len(trades)):
                if trades[i-1]['type'] == 'BUY' and trades[i]['type'] == 'SELL':
                    returns.append(trades[i]['profit'] / trades[i-1]['balance'])
            
            sharpe_ratio = 0
            if returns:
                returns_array = np.array(returns)
                if returns_array.std() > 0:
                    sharpe_ratio = (returns_array.mean() * 252) / (returns_array.std() * np.sqrt(252))
            
            win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
            
            # Calculate regime-specific metrics
            for regime, perf in regime_performance.items():
                if perf['trades'] > 0:
                    perf['win_rate'] = perf['wins'] / perf['trades']
                    perf['avg_return'] = perf['total_return'] / perf['trades']
                else:
                    perf['win_rate'] = 0
                    perf['avg_return'] = 0
                
                # Convert time in regime to percentage
                total_time = sum(p['time_in_regime'] for p in regime_performance.values())
                if total_time > 0:
                    perf['time_in_regime_pct'] = (perf['time_in_regime'] / total_time) * 100
                else:
                    perf['time_in_regime_pct'] = 0
            
            # Compile results
            results = {
                'params': params,
                'final_balance': final_balance,
                'total_return': total_return * 100,  # As percentage
                'total_trades': len(trades),
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate * 100,  # As percentage
                'max_drawdown': self._calculate_max_drawdown(trades),
                'regime_performance': regime_performance,
                'regime_history': regime_history,
                'trades': trades if self.backtest_settings.get('save_trade_history', False) else []
            }
            
            return results
            
        except Exception as e:
            return {'error': str(e), 'params': params}
    
    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown from trade history."""
        if not trades:
            return 0
        
        balances = [self.backtest_settings.get('initial_balance', 10000)]
        for trade in trades:
            balances.append(trade['balance'])
        
        peak = balances[0]
        max_dd = 0
        
        for balance in balances:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd * 100  # As percentage
    
    def optimize_parameters(self, param_ranges: Dict) -> pd.DataFrame:
        """
        Optimize AdaptiveMultiStrategy parameters using grid search.
        
        param_ranges should contain:
        - short_window_range: List of short MA windows
        - long_window_range: List of long MA windows
        - regime_switch_threshold_range: List of thresholds
        - signal_confirmation_bars_range: List of confirmation bars
        - min_trade_gap_minutes_range: List of trade gaps
        - whipsaw_threshold_range: List of whipsaw thresholds
        """
        results = []
        
        # Create parameter combinations
        param_combinations = list(itertools.product(
            param_ranges.get('short_window_range', [10]),
            param_ranges.get('long_window_range', [46]),
            param_ranges.get('regime_switch_threshold_range', [0.40]),
            param_ranges.get('signal_confirmation_bars_range', [2]),
            param_ranges.get('min_trade_gap_minutes_range', [15]),
            param_ranges.get('whipsaw_threshold_range', [8.0])
        ))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        # Use sequential processing for simplicity (multiprocessing can be added later)
        # For now, process sequentially to avoid complexity
        
        for i, params in enumerate(param_combinations):
            if i % 100 == 0:  # Reduce frequency of progress messages
                print(f"Progress: {i}/{len(param_combinations)} combinations tested")
                
            param_dict = {
                'short_window': params[0],
                'long_window': params[1],
                'regime_switch_threshold': params[2],
                'signal_confirmation_bars': params[3],
                'min_trade_gap_minutes': params[4],
                'whipsaw_threshold': params[5]
            }
            
            try:
                result = self.backtest_adaptive_strategy(param_dict)
                if 'error' not in result:
                    results.append({
                        'Short_Window': result['params']['short_window'],
                        'Long_Window': result['params']['long_window'],
                        'Regime_Switch_Threshold': result['params']['regime_switch_threshold'],
                        'Signal_Confirmation_Bars': result['params']['signal_confirmation_bars'],
                        'Min_Trade_Gap_Minutes': result['params']['min_trade_gap_minutes'],
                        'Whipsaw_Threshold': result['params']['whipsaw_threshold'],
                        'Final_Balance': result['final_balance'],
                        'Total_Return': result['total_return'],
                        'Total_Trades': result['total_trades'],
                        'Sharpe_Ratio': result['sharpe_ratio'],
                        'Win_Rate': result['win_rate'],
                        'Max_Drawdown': result['max_drawdown'],
                        'Trending_Trades': result['regime_performance']['TRENDING']['trades'],
                        'Trending_Win_Rate': result['regime_performance']['TRENDING']['win_rate'],
                        'Ranging_Trades': result['regime_performance']['RANGING']['trades'],
                        'Ranging_Win_Rate': result['regime_performance']['RANGING']['win_rate'],
                        'Volatile_Trades': result['regime_performance']['VOLATILE']['trades'],
                        'Volatile_Win_Rate': result['regime_performance']['VOLATILE']['win_rate']
                    })
            except Exception as e:
                print(f"Error processing params {param_dict}: {e}")
        
        return pd.DataFrame(results)

###############################################################################
# Enhanced Trading System Runner
###############################################################################
def run_enhanced_trading_system(df, config, include_adaptive=True):
    """
    Run the enhanced trading system including AdaptiveMultiStrategy.
    
    This maintains backward compatibility while adding adaptive strategy testing.
    """
    # First run the original strategies
    optimization_results, strategy_comparison = run_trading_system(
        df,
        high_frequency=config.get('high_frequency', '1H'),
        low_frequency=config.get('low_frequency', '15T'),
        max_iterations=config.get('max_iterations', 50),
        config=config
    )
    
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
        adaptive_results = adaptive_backtester.optimize_parameters(param_ranges)
        
        if not adaptive_results.empty:
            # Find best adaptive strategy
            best_adaptive = adaptive_results.loc[adaptive_results['Total_Return'].idxmax()]
            
            print("\nBest AdaptiveMultiStrategy parameters:")
            print(best_adaptive)
            
            # Add to strategy comparison
            adaptive_comparison = pd.DataFrame([{
                'Strategy': 'AdaptiveMulti',
                'Total_Return': best_adaptive['Total_Return'],
                'Total_Trades': best_adaptive['Total_Trades'],
                'Sharpe_Ratio': best_adaptive['Sharpe_Ratio'],
                'Win_Rate': best_adaptive['Win_Rate'],
                'Max_Drawdown': best_adaptive['Max_Drawdown'],
                'Short_Window': best_adaptive['Short_Window'],
                'Long_Window': best_adaptive['Long_Window']
            }])
            
            strategy_comparison = pd.concat([strategy_comparison, adaptive_comparison], ignore_index=True)
            
            # Save detailed adaptive results
            adaptive_results.to_csv('adaptive_strategy_optimization.csv', index=False)
            print("Adaptive strategy optimization results saved to 'adaptive_strategy_optimization.csv'")
            
            # Get detailed backtest for best parameters
            best_params = {
                'short_window': int(best_adaptive['Short_Window']),
                'long_window': int(best_adaptive['Long_Window']),
                'regime_switch_threshold': best_adaptive['Regime_Switch_Threshold'],
                'signal_confirmation_bars': int(best_adaptive['Signal_Confirmation_Bars']),
                'min_trade_gap_minutes': int(best_adaptive['Min_Trade_Gap_Minutes']),
                'whipsaw_threshold': best_adaptive['Whipsaw_Threshold']
            }
            
            detailed_result = adaptive_backtester.backtest_adaptive_strategy(best_params)
            
            # Save detailed results with regime performance
            with open('adaptive_strategy_detailed_results.json', 'w') as f:
                json.dump(detailed_result, f, indent=4, default=str)
            print("Detailed adaptive strategy results saved to 'adaptive_strategy_detailed_results.json'")
    
    return optimization_results, strategy_comparison

###############################################################################
# Enhanced Best Strategy JSON Generator
###############################################################################
def generate_enhanced_best_strategy_json(strategy_comparison, config, df, output_file='recommended_strategy.json'):
    """
    Generate an enhanced best_strategy.json with comprehensive metadata.
    """
    if strategy_comparison.empty:
        print("No strategies to compare. Creating minimal configuration.")
        
        # Create a minimal configuration with current parameters
        minimal_config = {
            "backtest_metadata": {
                "test_period_start": df.index.min().strftime("%Y-%m-%d") if not df.empty else "unknown",
                "test_period_end": df.index.max().strftime("%Y-%m-%d") if not df.empty else "unknown",
                "total_days": (df.index.max() - df.index.min()).days if not df.empty else 0,
                "data_frequency": config.get('high_frequency', '1H'),
                "backtest_timestamp": datetime.now().isoformat() + 'Z',
                "backtester_version": "2.0",
                "note": "No strategies met the criteria. Using default parameters."
            },
            "performance_metrics": {
                "total_return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "avg_trades_per_day": 0.0,
                "profit_factor": 1.0
            },
            "optimal_parameters": {
                "strategy": "MA",
                "short_window": 10,
                "long_window": 46
            },
            "live_trading_config": {
                "do_live_trades": False,
                "auto_align_position": False,
                "emergency_override_enabled": True
            },
            "validation_status": {
                "backtest_passed": False,
                "risk_limits_ok": False,
                "trade_frequency_ok": False,
                "ready_for_deployment": False,
                "reason": "No strategies met the backtesting criteria"
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(minimal_config, f, indent=4)
            
        print(f"\nMinimal configuration saved to '{output_file}'")
        return minimal_config
    
    # Find best strategy
    best_idx = strategy_comparison['Total_Return'].idxmax()
    best_row = strategy_comparison.loc[best_idx]
    
    # Calculate test period
    test_start = df.index.min()
    test_end = df.index.max()
    total_days = (test_end - test_start).days
    
    # Load detailed results if adaptive strategy
    regime_performance = None
    if best_row['Strategy'] == 'AdaptiveMulti':
        try:
            with open('adaptive_strategy_detailed_results.json', 'r') as f:
                detailed = json.load(f)
                regime_performance = detailed.get('regime_performance', {})
        except:
            pass
    
    # Create enhanced configuration
    enhanced_config = {
        "backtest_metadata": {
            "test_period_start": test_start.strftime("%Y-%m-%d"),
            "test_period_end": test_end.strftime("%Y-%m-%d"),
            "total_days": total_days,
            "data_frequency": config.get('high_frequency', '1H'),
            "backtest_timestamp": datetime.now().isoformat() + 'Z',
            "backtester_version": "2.0",
            "total_strategies_tested": len(strategy_comparison)
        },
        "performance_metrics": {
            "total_return_pct": float(best_row.get('Total_Return', 0)),
            "sharpe_ratio": float(best_row.get('Sharpe_Ratio', 0)),
            "max_drawdown_pct": float(best_row.get('Max_Drawdown', 0)),
            "win_rate": float(best_row.get('Win_Rate', 0)),
            "total_trades": int(best_row.get('Total_Trades', 0)),
            "avg_trades_per_day": float(best_row.get('Total_Trades', 0)) / max(total_days, 1),
            "profit_factor": float(best_row.get('Profit_Factor', 1.0))
        },
        "optimal_parameters": {
            "strategy": best_row['Strategy'],
            "short_window": int(best_row.get('Short_Window', 10)),
            "long_window": int(best_row.get('Long_Window', 46))
        },
        "live_trading_config": {
            "do_live_trades": False,  # Always start with false for safety
            "auto_align_position": False,
            "emergency_override_enabled": True
        },
        "validation_status": {
            "backtest_passed": True,
            "risk_limits_ok": best_row.get('Max_Drawdown', 0) < 20,  # Max 20% drawdown
            "trade_frequency_ok": best_row.get('Total_Trades', 0) / max(total_days, 1) <= 5,  # Max 5 trades per day
            "ready_for_deployment": False  # Requires manual review
        }
    }
    
    # Add strategy-specific parameters
    if best_row['Strategy'] == 'AdaptiveMulti':
        enhanced_config["optimal_parameters"].update({
            "regime_switch_threshold": 0.40,  # Will be updated from detailed results
            "signal_confirmation_bars": 2,
            "min_trade_gap_minutes": 15,
            "whipsaw_threshold": 8.0,
            "emergency_loss_threshold": -2000,
            "max_trades_per_day": 5
        })
        
        # Add regime performance if available
        if regime_performance:
            enhanced_config["regime_performance"] = regime_performance
            
            # Determine current market recommendation
            trending_score = regime_performance.get('TRENDING', {}).get('win_rate', 0)
            ranging_score = regime_performance.get('RANGING', {}).get('win_rate', 0)
            volatile_score = regime_performance.get('VOLATILE', {}).get('win_rate', 0)
            
            best_regime = max(
                [('TRENDING', trending_score), ('RANGING', ranging_score), ('VOLATILE', volatile_score)],
                key=lambda x: x[1]
            )[0]
            
            enhanced_config["market_conditions"] = {
                "best_performing_regime": best_regime,
                "recommendation": f"Strategy performs best in {best_regime} markets with {max(trending_score, ranging_score, volatile_score):.1f}% win rate"
            }
    
    # Add comparison with current strategy
    try:
        with open('best_strategy.json', 'r') as f:
            current_config = json.load(f)
            
        enhanced_config["comparison_with_current"] = {
            "current_strategy": current_config.get('Strategy', 'Unknown'),
            "current_return": current_config.get('Total_Return', 0),
            "new_return": float(best_row.get('Total_Return', 0)),
            "improvement_pct": float(best_row.get('Total_Return', 0)) - current_config.get('Total_Return', 0),
            "parameter_changes": {
                "short_window": {
                    "current": current_config.get('Short_Window', 0),
                    "new": int(best_row.get('Short_Window', 10)),
                    "changed": current_config.get('Short_Window', 0) != int(best_row.get('Short_Window', 10))
                },
                "long_window": {
                    "current": current_config.get('Long_Window', 0),
                    "new": int(best_row.get('Long_Window', 46)),
                    "changed": current_config.get('Long_Window', 0) != int(best_row.get('Long_Window', 46))
                }
            }
        }
    except:
        enhanced_config["comparison_with_current"] = {
            "note": "Could not compare with current strategy - file not found or invalid"
        }
    
    # Write the enhanced configuration
    with open(output_file, 'w') as f:
        json.dump(enhanced_config, f, indent=4)
    
    print(f"\nEnhanced strategy configuration saved to '{output_file}'")
    return enhanced_config

###############################################################################
# Main Function
###############################################################################
def main():
    """
    Enhanced main function with adaptive strategy testing.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Enhanced Bitcoin backtesting with AdaptiveMultiStrategy")
    parser.add_argument('--start-window-days-back', type=int, default=120,
                        help='Number of days to subtract from the current date as the start window')
    parser.add_argument('--end-window-days-back', type=int, default=0,
                        help='Number of days to subtract from the current date as the end window')
    parser.add_argument('--trading-window-days', type=int,
                        help='Number of days to analyze from the start date')
    parser.add_argument('--max-iterations', type=int, default=50,
                        help='Maximum iterations for optimization')
    parser.add_argument('--high-frequency', type=str, default='1H',
                        help='Sampling frequency for higher timeframe')
    parser.add_argument('--low-frequency', type=str, default='15T',
                        help='Sampling frequency for lower timeframe')
    parser.add_argument('--skip-adaptive', action='store_true',
                        help='Skip AdaptiveMultiStrategy testing')
    parser.add_argument('--output-file', type=str, default='recommended_strategy.json',
                        help='Output file for recommended strategy')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config('config.json')
    config['high_frequency'] = args.high_frequency
    config['low_frequency'] = args.low_frequency
    config['max_iterations'] = args.max_iterations
    
    # Setup and load data
    file_path = 'btcusd.log'
    metadata_file_path = f"{file_path}.metadata"
    if not os.path.exists(metadata_file_path):
        print("Metadata file not found. Creating it now...")
        create_metadata_file(file_path, metadata_file_path)
        print("Metadata file created.")
    
    # Determine date range
    current_date = datetime.now()
    start_date = current_date - timedelta(days=args.start_window_days_back)
    
    if args.trading_window_days:
        end_date = start_date + timedelta(days=args.trading_window_days)
    elif args.end_window_days_back > 0:
        end_date = current_date - timedelta(days=args.end_window_days_back)
    else:
        end_date = None
    
    print(f"Analyzing data from {start_date} to {end_date or 'present'}")
    
    # Load and parse data
    df = parse_log_file(file_path, start_date, end_date)
    print(f"Parsed {len(df)} trade events.")
    
    # Ensure datetime index
    df = ensure_datetime_index(df)
    
    # Analyze data
    print("Starting data analysis...")
    analyze_data(df)
    
    # Run enhanced trading system
    print("\nRunning enhanced trading system...")
    try:
        optimization_results, strategy_comparison = run_enhanced_trading_system(
            df, 
            config, 
            include_adaptive=not args.skip_adaptive
        )
        
        # Display results
        print("\n" + "="*80)
        print("STRATEGY COMPARISON RESULTS")
        print("="*80)
        print(strategy_comparison.to_string(index=False))
        
        # Generate enhanced configuration
        enhanced_config = generate_enhanced_best_strategy_json(
            strategy_comparison, 
            config, 
            df,
            args.output_file
        )
        
        # Save all results
        optimization_results.to_csv("all_strategy_results_enhanced.csv", index=False)
        strategy_comparison.to_csv("strategy_comparison_enhanced.csv", index=False)
        
        print("\n" + "="*80)
        print("BACKTESTING COMPLETE")
        print("="*80)
        print(f"Results saved to:")
        print(f"  - all_strategy_results_enhanced.csv")
        print(f"  - strategy_comparison_enhanced.csv")
        print(f"  - {args.output_file}")
        if not args.skip_adaptive:
            print(f"  - adaptive_strategy_optimization.csv")
            print(f"  - adaptive_strategy_detailed_results.json")
        
        # Display deployment recommendations
        if enhanced_config and enhanced_config.get('validation_status', {}).get('risk_limits_ok', False):
            print("\n" + "="*80)
            print("DEPLOYMENT RECOMMENDATIONS")
            print("="*80)
            print(f"✓ Best strategy: {enhanced_config['optimal_parameters']['strategy']}")
            print(f"✓ Expected return: {enhanced_config['performance_metrics']['total_return_pct']:.2f}%")
            print(f"✓ Risk metrics within limits (max drawdown: {enhanced_config['performance_metrics']['max_drawdown_pct']:.2f}%)")
            print("\nTo deploy:")
            print(f"1. Review {args.output_file}")
            print("2. Run validation: python validate_strategy.py")
            print("3. If validation passes, copy to best_strategy.json")
            print("4. Set 'do_live_trades' to true when ready")
        else:
            print("\n⚠️  WARNING: Strategy does not meet risk criteria for deployment")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())