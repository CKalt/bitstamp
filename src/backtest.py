# src/bktst_enhanced_shared.py
# Enhanced backtesting system that uses shared strategy code with live trading

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

# Import the SHARED strategy core
from tdr_core.strategy_core import AdaptiveStrategyCore
from indicators.technical_indicators import ensure_datetime_index
from backtesting.backtester import backtest
from optimization.optimizer import optimize_ma_parameters

###############################################################################
# Enhanced Configuration Loading
###############################################################################
def load_config(config_path='config.json'):
    """Load configuration with enhanced defaults for adaptive strategy testing."""
    default_config = {
        "strategy_constraints": {
            "min_trades_per_day": 0.1,  # Relaxed from 1
            "max_trades_per_day": 10,   # Increased from 4
            "min_total_return": -20.0,   # Allow some losses
            "min_profit_per_trade": -5.0  # Allow some losing trades
        },
        "adaptive_strategy_params": {
            "regime_lookback": 100,
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
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge with defaults
            for key in default_config:
                if key in user_config:
                    default_config[key].update(user_config[key])
            print(f"Loaded and merged config from '{config_path}'")
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
    else:
        print(f"Config file '{config_path}' not found. Using enhanced default config.")
        
    return default_config

###############################################################################
# AdaptiveMultiStrategy Backtesting using SHARED code
###############################################################################
class AdaptiveStrategyBacktester:
    """Specialized backtester for AdaptiveMultiStrategy using shared strategy core."""
    
    def __init__(self, df: pd.DataFrame, config: Dict):
        self.df = df.copy()
        self.config = config
        self.backtest_settings = config.get('backtest_settings', {})
        self.strategy_constraints = config.get('strategy_constraints', {})
        self.results = {}
        
    def backtest_adaptive_strategy(self, params: Dict) -> Dict:
        """
        Run backtest for AdaptiveMultiStrategy with given parameters.
        Uses the SHARED AdaptiveStrategyCore that is also used in live trading.
        
        Returns comprehensive results including regime performance breakdown.
        """
        try:
            # Create strategy instance using SHARED code
            strategy = AdaptiveStrategyCore(
                short_window=params['short_window'],
                long_window=params['long_window']
            )
            
            # Update strategy parameters
            strategy.update_parameters(params)
            
            # Initialize tracking variables
            balance = self.backtest_settings.get('initial_balance', 10000)
            position = 0  # 0: no position, 1: long, -1: short (though we always have position in real system)
            btc_amount = 0
            trades = []
            regime_history = []
            current_regime = 'trending'
            last_trade_time = None
            
            # Track performance by regime
            regime_performance = {
                'trending': {'trades': 0, 'wins': 0, 'total_return': 0, 'time_in_regime': 0},
                'ranging': {'trades': 0, 'wins': 0, 'total_return': 0, 'time_in_regime': 0},
                'volatile': {'trades': 0, 'wins': 0, 'total_return': 0, 'time_in_regime': 0}
            }
            
            # Prepare data
            df = self.df.copy()
            if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df = ensure_datetime_index(df)
            
            # Need minimum data for strategy
            min_periods = max(params['long_window'], strategy.regime_lookback) + 10
            if len(df) < min_periods:
                return {'error': 'Insufficient data for backtesting'}
            
            # Ensure we have close column
            if 'close' not in df.columns and 'price' in df.columns:
                df['close'] = df['price']
            
            # Track position for 100% position system
            for i in range(min_periods, len(df)):
                current_time = df.index[i]
                current_price = df['close'].iloc[i] if 'close' in df.columns else df['price'].iloc[i]
                
                # Get data slice up to current point
                data_slice = df.iloc[:i+1].copy()
                
                # Detect market regime periodically (every hour in the data)
                if i % 60 == 0 or current_regime is None:
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
                
                # Check trade gap constraint
                if last_trade_time and (current_time - last_trade_time).total_seconds() < strategy.min_trade_gap_minutes * 60:
                    continue
                
                # Generate signals based on current regime using SHARED logic
                signal, reason = strategy.should_generate_signal(data_slice, current_regime)
                
                # Execute trades based on signals
                # In real system we're always 100% positioned, so we flip positions
                if signal and signal != position:
                    # Calculate trade metrics
                    trade_fee = abs(balance * self.backtest_settings.get('trading_fee', 0.001))
                    slippage = current_price * self.backtest_settings.get('slippage', 0.0005)
                    
                    profit = 0
                    # Execute trade
                    if signal == 1 and position <= 0:  # Buy signal
                        if position == -1:  # Closing short (in real system this would be flipping from USD to BTC)
                            # When short, balance is already in USD
                            profit = 0  # No profit calculation needed for position flip
                        
                        # Open long (buy BTC)
                        btc_amount = (balance - trade_fee) / (current_price + slippage)
                        entry_price = current_price + slippage
                        
                        trades.append({
                            'timestamp': current_time,
                            'type': 'BUY',
                            'price': entry_price,
                            'amount': btc_amount,
                            'balance': balance,
                            'regime': current_regime,
                            'profit': profit,
                            'reason': reason
                        })
                        
                        position = 1
                        last_trade_time = current_time
                        
                        if current_regime:
                            regime_performance[current_regime]['trades'] += 1
                            if profit > 0:
                                regime_performance[current_regime]['wins'] += 1
                                regime_performance[current_regime]['total_return'] += profit / self.backtest_settings.get('initial_balance', 10000)
                    
                    elif signal == -1 and position >= 0:  # Sell signal
                        if position == 1:  # Close long (sell BTC for USD)
                            balance = btc_amount * (current_price - slippage) - trade_fee
                            profit = balance - self.backtest_settings.get('initial_balance', 10000)
                        
                        trades.append({
                            'timestamp': current_time,
                            'type': 'SELL',
                            'price': current_price - slippage,
                            'amount': btc_amount,
                            'balance': balance,
                            'regime': current_regime,
                            'profit': profit,
                            'reason': reason
                        })
                        
                        position = -1
                        last_trade_time = current_time
                        btc_amount = 0  # When short, we hold USD, not BTC
                        
                        if current_regime:
                            regime_performance[current_regime]['trades'] += 1
                            if profit > 0:
                                regime_performance[current_regime]['wins'] += 1
                                regime_performance[current_regime]['total_return'] += profit / self.backtest_settings.get('initial_balance', 10000)
            
            # Calculate final metrics
            if position == 1:
                # Currently holding BTC
                final_price = df['close'].iloc[-1] if 'close' in df.columns else df['price'].iloc[-1]
                final_balance = btc_amount * final_price
            else:
                # Currently holding USD
                final_balance = balance
            
            total_return = (final_balance - self.backtest_settings.get('initial_balance', 10000)) / self.backtest_settings.get('initial_balance', 10000)
            
            # Calculate additional metrics
            returns = []
            for i in range(1, len(trades)):
                if trades[i]['profit'] != 0:
                    returns.append(trades[i]['profit'] / self.backtest_settings.get('initial_balance', 10000))
            
            sharpe_ratio = 0
            if returns:
                returns_array = np.array(returns)
                if returns_array.std() > 0:
                    sharpe_ratio = (returns_array.mean() * 252) / (returns_array.std() * np.sqrt(252))
            
            win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
            
            # Calculate regime-specific metrics
            for regime, perf in regime_performance.items():
                if perf['trades'] > 0:
                    perf['win_rate'] = (perf['wins'] / perf['trades']) * 100
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
            
            # Calculate whipsaw statistics
            whipsaw_stats = self._analyze_whipsaws(trades)
            
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
                'whipsaw_stats': whipsaw_stats,
                'trades': trades if self.backtest_settings.get('save_trade_history', False) else []
            }
            
            return results
            
        except Exception as e:
            return {'error': str(e), 'params': params, 'traceback': traceback.format_exc()}
    
    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown from trade history."""
        if not trades:
            return 0
        
        balances = [self.backtest_settings.get('initial_balance', 10000)]
        for trade in trades:
            if trade['type'] == 'SELL' and trade['profit'] != 0:
                balances.append(balances[-1] + trade['profit'])
        
        peak = balances[0]
        max_dd = 0
        
        for balance in balances:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd * 100  # As percentage
    
    def _analyze_whipsaws(self, trades: List[Dict]) -> Dict:
        """Analyze trades for whipsaw patterns."""
        if len(trades) < 3:
            return {
                'total_whipsaws': 0,
                'whipsaw_losses': 0.0,
                'avg_whipsaw_cost': 0.0,
                'whipsaw_rate': 0.0,
                'whipsaw_patterns': []
            }
        
        whipsaws = []
        whipsaw_losses = 0.0
        detection_window = timedelta(hours=4)  # 4 hour window for whipsaw detection
        
        for i in range(len(trades) - 2):
            t1, t2, t3 = trades[i], trades[i+1], trades[i+2]
            
            # Check if trades form a whipsaw pattern (BUY->SELL->BUY or SELL->BUY->SELL)
            if t1['type'] == t3['type'] and t1['type'] != t2['type']:
                time_diff = t3['timestamp'] - t1['timestamp']
                
                if time_diff <= detection_window:
                    # Calculate loss from whipsaw
                    if t1['type'] == 'BUY':
                        # BUY -> SELL -> BUY pattern
                        loss = (t1['price'] - t2['price']) + (t3['price'] - t2['price'])
                    else:
                        # SELL -> BUY -> SELL pattern
                        loss = (t2['price'] - t1['price']) + (t2['price'] - t3['price'])
                    
                    whipsaw = {
                        'pattern': f"{t1['type']} -> {t2['type']} -> {t3['type']}",
                        'timestamps': [t1['timestamp'], t2['timestamp'], t3['timestamp']],
                        'prices': [t1['price'], t2['price'], t3['price']],
                        'loss': loss * t1['amount'],  # Multiply by position size
                        'duration': str(time_diff)
                    }
                    
                    whipsaws.append(whipsaw)
                    whipsaw_losses += max(0, whipsaw['loss'])
        
        whipsaw_stats = {
            'total_whipsaws': len(whipsaws),
            'whipsaw_losses': whipsaw_losses,
            'avg_whipsaw_cost': whipsaw_losses / len(whipsaws) if whipsaws else 0.0,
            'whipsaw_rate': (len(whipsaws) * 3) / len(trades) if len(trades) >= 3 else 0.0,
            'whipsaw_patterns': whipsaws[:5]  # Include first 5 whipsaws for review
        }
        
        return whipsaw_stats
    
    def optimize_parameters(self, param_ranges: Dict) -> pd.DataFrame:
        """
        Optimize AdaptiveMultiStrategy parameters using grid search.
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
        
        for i, params in enumerate(param_combinations):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(param_combinations)} combinations tested")
                
            param_dict = {
                'short_window': params[0],
                'long_window': params[1],
                'regime_switch_threshold': params[2],
                'signal_confirmation_bars': params[3],
                'min_trade_gap_minutes': params[4],
                'whipsaw_threshold': params[5]
            }
            
            # Skip invalid combinations
            if param_dict['short_window'] >= param_dict['long_window']:
                continue
            
            try:
                result = self.backtest_adaptive_strategy(param_dict)
                # Relaxed constraint - at least 1 trade per week on average
                min_trades = max(1, len(self.df) / (24 * 7))
                if 'error' not in result and result['total_trades'] >= min_trades:
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
                        'Trending_Trades': result['regime_performance']['trending']['trades'],
                        'Trending_Win_Rate': result['regime_performance']['trending']['win_rate'],
                        'Ranging_Trades': result['regime_performance']['ranging']['trades'],
                        'Ranging_Win_Rate': result['regime_performance']['ranging']['win_rate'],
                        'Volatile_Trades': result['regime_performance']['volatile']['trades'],
                        'Volatile_Win_Rate': result['regime_performance']['volatile']['win_rate']
                    })
            except Exception as e:
                print(f"Error processing params {param_dict}: {e}")
        
        return pd.DataFrame(results)

###############################################################################
# Interactive Configuration
###############################################################################
def get_interactive_param_ranges():
    """Interactive prompting for parameter configuration."""
    print("\n" + "="*80)
    print("INTERACTIVE PARAMETER CONFIGURATION")
    print("="*80)
    print("\nLet's configure the optimization parameters for your backtesting.")
    print("Press Enter to use the default values shown in brackets.\n")
    
    # Ask about data range
    print("Data Range Selection:")
    print("1. Last 30 days (recent market conditions)")
    print("2. Last 60 days (balanced recent history)")
    print("3. Last 120 days (default - broader market cycles)")
    print("4. Last 180 days (extended history)")
    print("5. Custom range")
    
    range_choice = input("\nSelect data range [3]: ").strip() or "3"
    
    if range_choice == "1":
        date_range = {'start_days_back': 30, 'end_days_back': 0}
    elif range_choice == "2":
        date_range = {'start_days_back': 60, 'end_days_back': 0}
    elif range_choice == "3":
        date_range = {'start_days_back': 120, 'end_days_back': 0}
    elif range_choice == "4":
        date_range = {'start_days_back': 180, 'end_days_back': 0}
    else:
        start_days = input("Days back from today to start [120]: ").strip()
        start_days = int(start_days) if start_days else 120
        end_days = input("Days back from today to end [0]: ").strip()
        end_days = int(end_days) if end_days else 0
        date_range = {'start_days_back': start_days, 'end_days_back': end_days}
    
    print(f"\n✓ Using data from {date_range['start_days_back']} to {date_range['end_days_back']} days ago\n")
    
    # Preset selection
    print("Choose optimization depth:")
    print("1. Quick Test (2 combinations - 1 minute)")
    print("2. Conservative (64 combinations - 5-10 minutes)")
    print("3. Moderate (216 combinations - 30-60 minutes)")
    print("4. Thorough (2304 combinations - 2-4 hours)")
    print("5. Aggressive (12500 combinations - overnight)")
    print("6. Custom (define your own ranges)")
    
    choice = input("\nSelect option [3]: ").strip() or "3"
    
    if choice == "1":
        return {
            'short_window_range': [10],
            'long_window_range': [46],
            'regime_switch_threshold_range': [0.40],
            'signal_confirmation_bars_range': [2],
            'min_trade_gap_minutes_range': [15],
            'whipsaw_threshold_range': [8.0],
            'date_range': date_range
        }
    elif choice == "2":
        return {
            'short_window_range': [10, 12],
            'long_window_range': [46, 50],
            'regime_switch_threshold_range': [0.40, 0.45],
            'signal_confirmation_bars_range': [2, 3],
            'min_trade_gap_minutes_range': [20, 30],
            'whipsaw_threshold_range': [8.0, 10.0],
            'date_range': date_range
        }
    elif choice == "3":
        return {
            'short_window_range': [8, 10, 12],
            'long_window_range': [40, 46, 50],
            'regime_switch_threshold_range': [0.35, 0.40, 0.45],
            'signal_confirmation_bars_range': [1, 2],
            'min_trade_gap_minutes_range': [15, 30],
            'whipsaw_threshold_range': [6.0, 8.0],
            'date_range': date_range
        }
    elif choice == "4":
        return {
            'short_window_range': [8, 10, 12, 15],
            'long_window_range': [40, 46, 50, 60],
            'regime_switch_threshold_range': [0.35, 0.40, 0.45, 0.50],
            'signal_confirmation_bars_range': [1, 2, 3],
            'min_trade_gap_minutes_range': [10, 15, 20, 30],
            'whipsaw_threshold_range': [6.0, 8.0, 10.0],
            'date_range': date_range
        }
    elif choice == "5":
        return {
            'short_window_range': [5, 8, 10, 12, 15],
            'long_window_range': [30, 40, 46, 50, 60],
            'regime_switch_threshold_range': [0.30, 0.35, 0.40, 0.45, 0.50],
            'signal_confirmation_bars_range': [1, 2, 3, 4],
            'min_trade_gap_minutes_range': [5, 10, 15, 20, 30],
            'whipsaw_threshold_range': [4.0, 6.0, 8.0, 10.0, 12.0],
            'date_range': date_range
        }
    else:  # Custom
        print("\n" + "-"*60)
        print("CUSTOM PARAMETER CONFIGURATION")
        print("-"*60)
        
        # Short window
        print("\nShort Moving Average Window (hours):")
        print("This is the fast MA period. Lower values = more responsive to price changes")
        short_input = input("Enter values separated by spaces [8 10 12]: ").strip()
        short_windows = [int(x) for x in (short_input.split() if short_input else ["8", "10", "12"])]
        
        # Long window
        print("\nLong Moving Average Window (hours):")
        print("This is the slow MA period. Should be significantly larger than short window")
        long_input = input("Enter values separated by spaces [40 46 50]: ").strip()
        long_windows = [int(x) for x in (long_input.split() if long_input else ["40", "46", "50"])]
        
        # Validate windows
        if min(long_windows) <= max(short_windows):
            print("\n⚠️  Warning: Long windows should be larger than short windows!")
            print("   Adjusting long windows to ensure valid combinations...")
            long_windows = [w for w in long_windows if w > max(short_windows)]
            if not long_windows:
                long_windows = [max(short_windows) + 10, max(short_windows) + 20]
            print(f"   New long windows: {long_windows}")
        
        # Regime switch threshold
        print("\nRegime Switch Threshold (0.0-1.0):")
        print("Confidence level required to switch market regimes (higher = more stable)")
        regime_input = input("Enter values separated by spaces [0.35 0.40 0.45]: ").strip()
        regime_thresholds = [float(x) for x in (regime_input.split() if regime_input else ["0.35", "0.40", "0.45"])]
        
        # Signal confirmation
        print("\nSignal Confirmation Bars:")
        print("Number of bars to confirm signal before trading (higher = fewer false signals)")
        confirm_input = input("Enter values separated by spaces [1 2]: ").strip()
        confirm_bars = [int(x) for x in (confirm_input.split() if confirm_input else ["1", "2"])]
        
        # Trade gap
        print("\nMinimum Trade Gap (minutes):")
        print("Minimum time between trades to avoid overtrading")
        gap_input = input("Enter values separated by spaces [15 30]: ").strip()
        trade_gaps = [int(x) for x in (gap_input.split() if gap_input else ["15", "30"])]
        
        # Whipsaw threshold
        print("\nWhipsaw Protection Threshold:")
        print("Volatility threshold to avoid trading in choppy markets")
        whipsaw_input = input("Enter values separated by spaces [6.0 8.0]: ").strip()
        whipsaw_thresholds = [float(x) for x in (whipsaw_input.split() if whipsaw_input else ["6.0", "8.0"])]
        
        # Calculate total combinations
        total_combos = (len(short_windows) * len(long_windows) * len(regime_thresholds) * 
                       len(confirm_bars) * len(trade_gaps) * len(whipsaw_thresholds))
        
        print(f"\n📊 Total combinations to test: {total_combos}")
        
        if total_combos > 1000:
            print(f"⚠️  Warning: {total_combos} combinations may take several hours!")
            proceed = input("Continue? (y/n) [y]: ").strip().lower() or "y"
            if proceed != "y":
                print("Reducing to moderate preset...")
                return {
                    'short_window_range': [8, 10, 12],
                    'long_window_range': [40, 46, 50],
                    'regime_switch_threshold_range': [0.35, 0.40, 0.45],
                    'signal_confirmation_bars_range': [1, 2],
                    'min_trade_gap_minutes_range': [15, 30],
                    'whipsaw_threshold_range': [6.0, 8.0],
                    'date_range': date_range
                }
        
        return {
            'short_window_range': short_windows,
            'long_window_range': long_windows,
            'regime_switch_threshold_range': regime_thresholds,
            'signal_confirmation_bars_range': confirm_bars,
            'min_trade_gap_minutes_range': trade_gaps,
            'whipsaw_threshold_range': whipsaw_thresholds,
            'date_range': date_range
        }

###############################################################################
# Enhanced Trading System Runner
###############################################################################
def run_enhanced_trading_system(df, config, include_adaptive=True, args=None):
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
        if hasattr(args, 'user_prompts') and args.user_prompts:
            # Interactive mode
            param_ranges = get_interactive_param_ranges()
            
            # Extract date range if provided
            if 'date_range' in param_ranges:
                date_range = param_ranges.pop('date_range')
                # Update args with interactive date range
                args.start_window_days_back = date_range['start_days_back']
                args.end_window_days_back = date_range['end_days_back']
            
            total_combos = (len(param_ranges['short_window_range']) * 
                           len(param_ranges['long_window_range']) * 
                           len(param_ranges['regime_switch_threshold_range']) * 
                           len(param_ranges['signal_confirmation_bars_range']) * 
                           len(param_ranges['min_trade_gap_minutes_range']) * 
                           len(param_ranges['whipsaw_threshold_range']))
            print(f"\nStarting optimization with {total_combos} parameter combinations...")
        elif hasattr(args, 'test_run') and args.test_run:
            # Minimal ranges for test run - only 2 combinations total
            param_ranges = {
                'short_window_range': [10],              # 1 value
                'long_window_range': [46],               # 1 value
                'regime_switch_threshold_range': [0.40], # 1 value
                'signal_confirmation_bars_range': [2],   # 1 value
                'min_trade_gap_minutes_range': [15, 30], # 2 values for testing
                'whipsaw_threshold_range': [8.0]         # 1 value
            }
            print("Running in TEST MODE with minimal parameter combinations (2 total)")
        elif hasattr(args, 'optimization_preset') and args.optimization_preset:
            # Use preset ranges
            presets = {
                'conservative': {
                    'short_window_range': [10, 12],                    # 2 values
                    'long_window_range': [46, 50],                     # 2 values
                    'regime_switch_threshold_range': [0.40, 0.45],     # 2 values
                    'signal_confirmation_bars_range': [2, 3],          # 2 values
                    'min_trade_gap_minutes_range': [20, 30],           # 2 values
                    'whipsaw_threshold_range': [8.0, 10.0]             # 2 values
                    # Total: 64 combinations
                },
                'moderate': {
                    'short_window_range': [8, 10, 12],                 # 3 values
                    'long_window_range': [40, 46, 50],                 # 3 values
                    'regime_switch_threshold_range': [0.35, 0.40, 0.45], # 3 values
                    'signal_confirmation_bars_range': [1, 2],          # 2 values
                    'min_trade_gap_minutes_range': [15, 30],           # 2 values
                    'whipsaw_threshold_range': [6.0, 8.0]              # 2 values
                    # Total: 216 combinations
                },
                'aggressive': {
                    'short_window_range': [5, 8, 10, 12, 15],         # 5 values
                    'long_window_range': [30, 40, 46, 50, 60],        # 5 values
                    'regime_switch_threshold_range': [0.30, 0.35, 0.40, 0.45, 0.50], # 5 values
                    'signal_confirmation_bars_range': [1, 2, 3, 4],    # 4 values
                    'min_trade_gap_minutes_range': [5, 10, 15, 20, 30], # 5 values
                    'whipsaw_threshold_range': [4.0, 6.0, 8.0, 10.0, 12.0] # 5 values
                    # Total: 12,500 combinations
                }
            }
            
            if args.optimization_preset == 'custom':
                # Build custom ranges from command line arguments
                param_ranges = {
                    'short_window_range': args.short_windows or [10, 12],
                    'long_window_range': args.long_windows or [46, 50],
                    'regime_switch_threshold_range': [0.35, 0.40, 0.45],
                    'signal_confirmation_bars_range': [1, 2],
                    'min_trade_gap_minutes_range': [15, 30],
                    'whipsaw_threshold_range': [6.0, 8.0]
                }
                total_combos = len(param_ranges['short_window_range']) * len(param_ranges['long_window_range']) * 3 * 2 * 2 * 2
                print(f"Using CUSTOM parameter ranges ({total_combos} combinations)")
            else:
                param_ranges = presets[args.optimization_preset]
                combo_counts = {'conservative': 64, 'moderate': 216, 'aggressive': 12500}
                print(f"Using {args.optimization_preset.upper()} preset ({combo_counts[args.optimization_preset]} combinations)")
        else:
            # Check if parameter ranges are defined in config
            if 'optimization_ranges' in config:
                param_ranges = config['optimization_ranges']
                print(f"Using parameter ranges from config.json")
            else:
                # Default full parameter ranges for thorough optimization
                param_ranges = {
                    'short_window_range': [8, 10, 12, 15],              # 4 values
                    'long_window_range': [40, 46, 50, 60],              # 4 values
                    'regime_switch_threshold_range': [0.35, 0.40, 0.45, 0.50], # 4 values
                    'signal_confirmation_bars_range': [1, 2, 3],        # 3 values
                    'min_trade_gap_minutes_range': [10, 15, 20, 30],   # 4 values
                    'whipsaw_threshold_range': [6.0, 8.0, 10.0]        # 3 values
                }
                print("Using default parameter ranges (2304 combinations)")
        
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
            
            # Display whipsaw statistics if available
            if 'whipsaw_stats' in detailed_result:
                whipsaw_stats = detailed_result['whipsaw_stats']
                print("\n🌊 WHIPSAW ANALYSIS:")
                print(f"  • Total Whipsaws: {whipsaw_stats['total_whipsaws']}")
                print(f"  • Whipsaw Losses: ${whipsaw_stats['whipsaw_losses']:.2f}")
                print(f"  • Average Whipsaw Cost: ${whipsaw_stats['avg_whipsaw_cost']:.2f}")
                print(f"  • Whipsaw Rate: {whipsaw_stats['whipsaw_rate']:.1%}")
                
                if whipsaw_stats['whipsaw_patterns']:
                    print("\n  Recent Whipsaw Patterns:")
                    for i, pattern in enumerate(whipsaw_stats['whipsaw_patterns'][:3], 1):
                        print(f"    {i}. {pattern['pattern']} - Loss: ${pattern['loss']:.2f}")
            
            # Save detailed results with regime performance
            with open('adaptive_strategy_detailed_results.json', 'w') as f:
                json.dump(detailed_result, f, indent=4, default=str)
            print("\nDetailed adaptive strategy results saved to 'adaptive_strategy_detailed_results.json'")
    
    return optimization_results, strategy_comparison

###############################################################################
# Enhanced Best Strategy JSON Generator
###############################################################################
def generate_enhanced_best_strategy_json(strategy_comparison, config, df, output_file='recommended_strategy.json'):
    """
    Generate an enhanced best_strategy.json with comprehensive metadata.
    This creates a format compatible with live trading (src/tdr.py).
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
                "backtester_version": "2.0-shared",
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
            "backtester_version": "2.0-shared",
            "total_strategies_tested": len(strategy_comparison),
            "uses_shared_code": True
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
            "strategy": best_row['Strategy']
        },
        "live_trading_config": {
            "do_live_trades": False,  # Always start with false for safety
            "auto_align_position": False,
            "emergency_override_enabled": True
        },
        "validation_status": {
            "backtest_passed": True,
            "risk_limits_ok": bool(best_row.get('Max_Drawdown', 0) < 20),  # Max 20% drawdown
            "trade_frequency_ok": bool(best_row.get('Total_Trades', 0) / max(total_days, 1) <= 5),  # Max 5 trades per day
            "ready_for_deployment": False  # Requires manual review
        }
    }
    
    # Add window parameters if they exist and are not NaN
    if 'Short_Window' in best_row and pd.notna(best_row['Short_Window']):
        enhanced_config["optimal_parameters"]["short_window"] = int(best_row['Short_Window'])
    elif best_row['Strategy'] in ['MA', 'AdaptiveMulti']:
        enhanced_config["optimal_parameters"]["short_window"] = 10  # Default
        
    if 'Long_Window' in best_row and pd.notna(best_row['Long_Window']):
        enhanced_config["optimal_parameters"]["long_window"] = int(best_row['Long_Window'])
    elif best_row['Strategy'] in ['MA', 'AdaptiveMulti']:
        enhanced_config["optimal_parameters"]["long_window"] = 46  # Default
    
    # Add RAMM-specific parameters if present
    if best_row['Strategy'] == 'RAMM':
        if 'MA_Short' in best_row and pd.notna(best_row['MA_Short']):
            enhanced_config["optimal_parameters"]["MA_Short"] = int(best_row['MA_Short'])
        if 'MA_Long' in best_row and pd.notna(best_row['MA_Long']):
            enhanced_config["optimal_parameters"]["MA_Long"] = int(best_row['MA_Long'])
        if 'RSI_Period' in best_row and pd.notna(best_row['RSI_Period']):
            enhanced_config["optimal_parameters"]["RSI_Period"] = int(best_row['RSI_Period'])
        if 'RSI_Overbought' in best_row and pd.notna(best_row['RSI_Overbought']):
            enhanced_config["optimal_parameters"]["RSI_Overbought"] = int(best_row['RSI_Overbought'])
        if 'RSI_Oversold' in best_row and pd.notna(best_row['RSI_Oversold']):
            enhanced_config["optimal_parameters"]["RSI_Oversold"] = int(best_row['RSI_Oversold'])
        if 'Regime_Lookback' in best_row and pd.notna(best_row['Regime_Lookback']):
            enhanced_config["optimal_parameters"]["Regime_Lookback"] = int(best_row['Regime_Lookback'])
    
    # Add strategy-specific parameters
    if best_row['Strategy'] == 'AdaptiveMulti':
        # Load from optimization results
        try:
            opt_df = pd.read_csv('adaptive_strategy_optimization.csv')
            best_opt = opt_df.loc[opt_df['Total_Return'].idxmax()]
            
            enhanced_config["optimal_parameters"].update({
                "regime_switch_threshold": float(best_opt.get('Regime_Switch_Threshold', 0.40)),
                "signal_confirmation_bars": int(best_opt.get('Signal_Confirmation_Bars', 2)),
                "min_trade_gap_minutes": int(best_opt.get('Min_Trade_Gap_Minutes', 15)),
                "whipsaw_threshold": float(best_opt.get('Whipsaw_Threshold', 8.0)),
                "emergency_loss_threshold": -2000,
                "max_trades_per_day": 5
            })
        except:
            # Use defaults compatible with live trading
            enhanced_config["optimal_parameters"].update({
                "regime_switch_threshold": 0.40,
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
            trending_score = regime_performance.get('trending', {}).get('win_rate', 0)
            ranging_score = regime_performance.get('ranging', {}).get('win_rate', 0)
            volatile_score = regime_performance.get('volatile', {}).get('win_rate', 0)
            
            best_regime = max(
                [('trending', trending_score), ('ranging', ranging_score), ('volatile', volatile_score)],
                key=lambda x: x[1]
            )[0]
            
            enhanced_config["market_conditions"] = {
                "best_performing_regime": best_regime,
                "recommendation": f"Strategy performs best in {best_regime.upper()} markets with {max(trending_score, ranging_score, volatile_score):.1f}% win rate"
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
            "parameter_changes": {}
        }
        
        # Add parameter changes based on strategy type
        if pd.notna(best_row.get('Short_Window')):
            enhanced_config["comparison_with_current"]["parameter_changes"]["short_window"] = {
                "current": current_config.get('Short_Window', 0),
                "new": int(best_row['Short_Window']),
                "changed": bool(current_config.get('Short_Window', 0) != int(best_row['Short_Window']))
            }
        
        if pd.notna(best_row.get('Long_Window')):
            enhanced_config["comparison_with_current"]["parameter_changes"]["long_window"] = {
                "current": current_config.get('Long_Window', 0),
                "new": int(best_row['Long_Window']),
                "changed": bool(current_config.get('Long_Window', 0) != int(best_row['Long_Window']))
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
    Enhanced main function with adaptive strategy testing using SHARED code.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Enhanced Bitcoin backtesting with shared AdaptiveMultiStrategy")
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
    parser.add_argument('--test-run', action='store_true',
                        help='Quick test run with minimal parameter combinations')
    parser.add_argument('--optimization-preset', type=str, choices=['conservative', 'moderate', 'aggressive', 'custom'],
                        help='Use preset parameter ranges for optimization')
    parser.add_argument('--short-windows', type=int, nargs='+',
                        help='Custom short window values (e.g., --short-windows 8 10 12)')
    parser.add_argument('--long-windows', type=int, nargs='+',
                        help='Custom long window values (e.g., --long-windows 40 46 50)')
    parser.add_argument('--user-prompts', action='store_true',
                        help='Interactive mode to configure optimization parameters')
    
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
    
    # Add volume column if not present
    if 'volume' not in df.columns:
        df['volume'] = df['price'] * df['amount']
    
    # Analyze data
    print("Starting data analysis...")
    analyze_data(df)
    
    # Run enhanced trading system
    print("\nRunning enhanced trading system with SHARED strategy code...")
    try:
        optimization_results, strategy_comparison = run_enhanced_trading_system(
            df, 
            config, 
            include_adaptive=not args.skip_adaptive,
            args=args
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
            print("2. Run validation: python src/validate_strategy.py")
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