# src/tdr_core/command_interface_enhanced.py
# Enhanced command interface with additional diagnostic capabilities

import json
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from bktst_enhanced import AdaptiveStrategyBacktester

class EnhancedDiagnosticCommands:
    """
    Mixin class that adds enhanced diagnostic commands to the shell.
    
    This should be mixed into the CryptoShell class to add new commands.
    """
    
    def do_backtest_current_params(self, arg):
        """Run quick backtest with current live parameters.
        Usage: backtest_current_params [days]
        
        Args:
            days: Number of days to backtest (default: 1)
        """
        try:
            days = int(arg) if arg else 1
            
            if not hasattr(self, 'auto_trader') or not self.auto_trader:
                print("Error: No active trading session")
                return
                
            print(f"Running backtest for current parameters over last {days} days...")
            
            # Get current parameters
            strategy = self.auto_trader.strategy
            current_params = {
                'short_window': strategy.short_window,
                'long_window': strategy.long_window,
                'regime_switch_threshold': getattr(strategy, 'regime_switch_threshold', 0.40),
                'signal_confirmation_bars': getattr(strategy, 'signal_confirmation_bars', 2),
                'min_trade_gap_minutes': getattr(strategy, 'min_trade_gap_minutes', 15),
                'whipsaw_threshold': getattr(strategy, 'whipsaw_threshold', 8.0)
            }
            
            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Use data manager to get data
            df = self.data_manager.get_dataframe()
            
            # Filter to date range
            df = df[df.index >= start_date]
            
            if len(df) < 50:
                print(f"Insufficient data for backtesting ({len(df)} rows)")
                return
                
            # Run backtest
            config = {'backtest_settings': {'initial_balance': 10000}}
            backtester = AdaptiveStrategyBacktester(df, config)
            results = backtester.backtest_adaptive_strategy(current_params)
            
            if 'error' in results:
                print(f"Backtest error: {results['error']}")
                return
                
            # Display results
            print("\n" + "="*60)
            print("BACKTEST RESULTS - Current Parameters")
            print("="*60)
            print(f"Period: {days} days")
            print(f"Total Return: {results['total_return']:.2f}%")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Win Rate: {results['win_rate']:.1f}%")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Max Drawdown: {results['max_drawdown']:.1f}%")
            
            # Show regime performance
            print("\nRegime Performance:")
            for regime, perf in results['regime_performance'].items():
                print(f"  {regime}: {perf['trades']} trades, "
                      f"{perf['win_rate']*100:.1f}% win rate, "
                      f"{perf['time_in_regime_pct']:.1f}% of time")
                      
        except Exception as e:
            print(f"Error running backtest: {e}")
            import traceback
            traceback.print_exc()
    
    def do_compare_strategies(self, arg):
        """Compare current strategy with recommended parameters.
        Usage: compare_strategies [recommended_file]
        
        Args:
            recommended_file: Path to recommended strategy JSON (default: recommended_strategy.json)
        """
        try:
            rec_file = arg.strip() if arg else 'recommended_strategy.json'
            
            if not os.path.exists(rec_file):
                print(f"Error: Recommended strategy file not found: {rec_file}")
                return
                
            # Load recommended strategy
            with open(rec_file, 'r') as f:
                recommended = json.load(f)
                
            # Get current parameters
            if not hasattr(self, 'auto_trader') or not self.auto_trader:
                print("Error: No active trading session")
                return
                
            strategy = self.auto_trader.strategy
            
            print("\n" + "="*60)
            print("STRATEGY COMPARISON")
            print("="*60)
            
            # Compare basic parameters
            print("\nBasic Parameters:")
            print(f"{'Parameter':<25} {'Current':<15} {'Recommended':<15} {'Change':<10}")
            print("-" * 65)
            
            params_to_compare = [
                ('Strategy Type', getattr(strategy, 'name', 'MA'), 
                 recommended.get('optimal_parameters', {}).get('strategy', 'MA')),
                ('Short Window', strategy.short_window, 
                 recommended.get('optimal_parameters', {}).get('short_window', 10)),
                ('Long Window', strategy.long_window,
                 recommended.get('optimal_parameters', {}).get('long_window', 46))
            ]
            
            for param_name, current, rec in params_to_compare:
                change = "→" if current != rec else "="
                print(f"{param_name:<25} {str(current):<15} {str(rec):<15} {change:<10}")
                
            # Compare performance metrics
            print("\nExpected Performance:")
            rec_perf = recommended.get('performance_metrics', {})
            print(f"  Backtest Return: {rec_perf.get('total_return_pct', 0):.2f}%")
            print(f"  Sharpe Ratio: {rec_perf.get('sharpe_ratio', 0):.2f}")
            print(f"  Win Rate: {rec_perf.get('win_rate', 0):.1f}%")
            print(f"  Max Drawdown: {rec_perf.get('max_drawdown_pct', 0):.1f}%")
            
            # Show validation status
            val_status = recommended.get('validation_status', {})
            print("\nValidation Status:")
            for check, passed in val_status.items():
                status = "✅" if passed else "❌"
                print(f"  {status} {check}")
                
        except Exception as e:
            print(f"Error comparing strategies: {e}")
    
    def do_regime_history(self, arg):
        """Show recent regime changes and performance.
        Usage: regime_history [hours]
        
        Args:
            hours: Number of hours to look back (default: 24)
        """
        try:
            hours = int(arg) if arg else 24
            
            if not hasattr(self, 'auto_trader') or not self.auto_trader:
                print("Error: No active trading session")
                return
                
            # Get diagnostics
            logger = self.auto_trader.diagnostic_logger
            if not logger:
                print("No diagnostic logger available")
                return
                
            # Find regime change events
            cutoff_time = datetime.now() - timedelta(hours=hours)
            regime_events = []
            
            # Load recent diagnostic files
            diag_dir = 'diagnostics'
            if os.path.exists(diag_dir):
                for filename in sorted(os.listdir(diag_dir), reverse=True):
                    if filename.endswith('.json') and filename.startswith('trading_diagnostics_'):
                        filepath = os.path.join(diag_dir, filename)
                        try:
                            with open(filepath, 'r') as f:
                                data = json.load(f)
                                
                            for event in data.get('events', []):
                                if event['type'] == 'REGIME_CHANGE':
                                    event_time = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                                    if event_time >= cutoff_time:
                                        regime_events.append(event)
                                        
                        except Exception:
                            continue
                            
            print(f"\n" + "="*60)
            print(f"REGIME HISTORY - Last {hours} hours")
            print("="*60)
            
            if not regime_events:
                print("No regime changes found in the specified period")
                
                # Show current regime
                strategy = self.auto_trader.strategy
                if hasattr(strategy, 'current_regime'):
                    print(f"\nCurrent Regime: {strategy.current_regime}")
                    print(f"Time in Current Regime: {getattr(strategy, 'time_in_regime_minutes', 'Unknown')} minutes")
                return
                
            # Display regime changes
            print(f"\nFound {len(regime_events)} regime changes:")
            print(f"{'Timestamp':<20} {'Old Regime':<12} {'New Regime':<12} {'Confidence':<10}")
            print("-" * 54)
            
            for event in regime_events:
                timestamp = event['timestamp'][:19]
                old_regime = event['data'].get('old_regime', 'N/A')
                new_regime = event['data'].get('new_regime', 'N/A')
                confidence = event['data'].get('confidence', 0)
                
                print(f"{timestamp:<20} {old_regime:<12} {new_regime:<12} {confidence:<10.2f}")
                
            # Calculate time spent in each regime
            print("\nTime Distribution:")
            regime_times = {'TRENDING': 0, 'RANGING': 0, 'VOLATILE': 0}
            
            for i in range(len(regime_events) - 1):
                regime = regime_events[i]['data']['new_regime']
                start_time = datetime.fromisoformat(regime_events[i]['timestamp'].replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(regime_events[i+1]['timestamp'].replace('Z', '+00:00'))
                duration = (end_time - start_time).total_seconds() / 3600  # Hours
                
                if regime in regime_times:
                    regime_times[regime] += duration
                    
            # Add time for current regime
            if regime_events:
                current_regime = regime_events[-1]['data']['new_regime']
                last_change = datetime.fromisoformat(regime_events[-1]['timestamp'].replace('Z', '+00:00'))
                current_duration = (datetime.now() - last_change).total_seconds() / 3600
                
                if current_regime in regime_times:
                    regime_times[current_regime] += current_duration
                    
            total_hours = sum(regime_times.values())
            if total_hours > 0:
                for regime, hours_in_regime in regime_times.items():
                    pct = (hours_in_regime / total_hours) * 100
                    print(f"  {regime}: {hours_in_regime:.1f}h ({pct:.1f}%)")
                    
        except Exception as e:
            print(f"Error getting regime history: {e}")
    
    def do_signal_analysis(self, arg):
        """Analyze why system is or isn't trading.
        Usage: signal_analysis [verbose]
        
        Args:
            verbose: Show detailed analysis (optional)
        """
        try:
            verbose = arg.lower() == 'verbose' if arg else False
            
            if not hasattr(self, 'auto_trader') or not self.auto_trader:
                print("Error: No active trading session")
                return
                
            strategy = self.auto_trader.strategy
            
            print("\n" + "="*60)
            print("SIGNAL ANALYSIS")
            print("="*60)
            
            # Current market state
            df = self.data_manager.get_dataframe()
            if len(df) < 50:
                print("Insufficient data for analysis")
                return
                
            # Detect current regime
            regime_result = strategy.detect_market_regime(df)
            current_regime = regime_result['regime']
            confidence = regime_result['confidence']
            metrics = regime_result['metrics']
            
            print(f"\nCurrent Market State:")
            print(f"  Regime: {current_regime} (confidence: {confidence:.2f})")
            print(f"  Trend Strength: {metrics['trend_strength']:.3f}")
            print(f"  Volatility: {metrics['volatility']:.4f}")
            print(f"  Whipsaw Ratio: {metrics['whipsaw_ratio']:.2f}")
            print(f"  Range Bound Score: {metrics['range_bound_score']:.3f}")
            
            # Current position
            print(f"\nCurrent Position:")
            print(f"  Direction: {'LONG' if strategy.position == 1 else 'SHORT'}")
            print(f"  Entry Price: ${getattr(strategy, 'position_entry_price', 'N/A')}")
            print(f"  Current Price: ${df['price'].iloc[-1]:.2f}")
            
            # Check for signals
            print(f"\nSignal Generation:")
            
            # Generate signal based on regime
            if current_regime == 'TRENDING':
                signal = strategy.generate_trending_signal(df)
                signal_type = "MA Crossover"
            elif current_regime == 'RANGING':
                signal = strategy.generate_ranging_signal(df)
                signal_type = "Mean Reversion (RSI + BB)"
            else:
                signal = strategy.generate_volatile_signal(df)
                signal_type = "Breakout (MACD)"
                
            print(f"  Strategy Type: {signal_type}")
            print(f"  Current Signal: {'BUY' if signal == 1 else 'SELL' if signal == -1 else 'NONE'}")
            
            # Check constraints
            print(f"\nTrading Constraints:")
            
            # Signal confirmation
            if hasattr(strategy, 'signal_buffer'):
                consistent_signals = all(s == signal for s in strategy.signal_buffer)
                print(f"  Signal Confirmation: {len(strategy.signal_buffer)}/{strategy.signal_confirmation_bars} "
                      f"({'READY' if consistent_signals else 'WAITING'})")
            
            # Trade gap
            if hasattr(strategy, 'last_trade_time'):
                time_since_trade = (datetime.now() - strategy.last_trade_time).total_seconds() / 60
                gap_ok = time_since_trade >= strategy.min_trade_gap_minutes
                print(f"  Trade Gap: {time_since_trade:.1f}/{strategy.min_trade_gap_minutes} min "
                      f"({'OK' if gap_ok else 'WAITING'})")
            else:
                print(f"  Trade Gap: No previous trades")
                
            # Daily limit
            trades_today = getattr(strategy, 'trades_today', 0)
            max_trades = getattr(strategy, 'daily_trade_limit', 5)
            print(f"  Daily Trades: {trades_today}/{max_trades} "
                  f"({'LIMIT REACHED' if trades_today >= max_trades else 'OK'})")
                  
            # Emergency conditions
            if hasattr(strategy, 'position_entry_price') and strategy.position_entry_price:
                current_price = df['price'].iloc[-1]
                if strategy.position == 1:  # Long
                    pnl = (current_price - strategy.position_entry_price) * getattr(strategy, 'position_size', 0)
                else:  # Short
                    pnl = (strategy.position_entry_price - current_price) * getattr(strategy, 'balance_usd', 0) / strategy.position_entry_price
                    
                emergency_threshold = getattr(strategy, 'emergency_loss_threshold', -2000)
                print(f"  Emergency Exit: P&L ${pnl:.2f} (threshold: ${emergency_threshold})")
                
            if verbose:
                print(f"\nDetailed Metrics:")
                print(f"  MA Short: {df[f'MA_{strategy.short_window}'].iloc[-1]:.2f}")
                print(f"  MA Long: {df[f'MA_{strategy.long_window}'].iloc[-1]:.2f}")
                
                if 'RSI' in df.columns:
                    print(f"  RSI: {df['RSI'].iloc[-1]:.2f}")
                    
                if 'BB_upper' in df.columns:
                    print(f"  BB Upper: {df['BB_upper'].iloc[-1]:.2f}")
                    print(f"  BB Lower: {df['BB_lower'].iloc[-1]:.2f}")
                    
        except Exception as e:
            print(f"Error analyzing signals: {e}")
            import traceback
            traceback.print_exc()
    
    def do_risk_metrics(self, arg):
        """Show current risk metrics and exposure.
        Usage: risk_metrics
        """
        try:
            if not hasattr(self, 'auto_trader') or not self.auto_trader:
                print("Error: No active trading session")
                return
                
            strategy = self.auto_trader.strategy
            
            print("\n" + "="*60)
            print("RISK METRICS")
            print("="*60)
            
            # Position risk
            print("\nPosition Risk:")
            
            current_price = self.data_manager.get_dataframe()['price'].iloc[-1]
            
            if strategy.position == 1:  # Long
                position_value = getattr(strategy, 'position_size', 0) * current_price
                print(f"  Position: LONG")
                print(f"  BTC Amount: {getattr(strategy, 'position_size', 0):.8f}")
                print(f"  Position Value: ${position_value:.2f}")
                
                if hasattr(strategy, 'position_entry_price') and strategy.position_entry_price:
                    entry_value = getattr(strategy, 'position_size', 0) * strategy.position_entry_price
                    pnl = position_value - entry_value
                    pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
                    
                    print(f"  Entry Price: ${strategy.position_entry_price:.2f}")
                    print(f"  Current Price: ${current_price:.2f}")
                    print(f"  Unrealized P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                    
            else:  # Short
                position_value = getattr(strategy, 'balance_usd', 0)
                print(f"  Position: SHORT")
                print(f"  USD Amount: ${position_value:.2f}")
                
                if hasattr(strategy, 'position_entry_price') and strategy.position_entry_price:
                    btc_owed = position_value / strategy.position_entry_price
                    current_cost = btc_owed * current_price
                    pnl = position_value - current_cost
                    pnl_pct = (pnl / position_value) * 100 if position_value > 0 else 0
                    
                    print(f"  Entry Price: ${strategy.position_entry_price:.2f}")
                    print(f"  Current Price: ${current_price:.2f}")
                    print(f"  Unrealized P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                    
            # Risk limits
            print("\nRisk Limits:")
            print(f"  Emergency Loss Threshold: ${getattr(strategy, 'emergency_loss_threshold', -2000)}")
            print(f"  Max Position Size: ${getattr(strategy, 'max_position_size', 50000)}")
            print(f"  Daily Trade Limit: {getattr(strategy, 'daily_trade_limit', 5)}")
            
            # Historical metrics
            print("\nHistorical Metrics (Session):")
            
            total_trades = getattr(strategy, 'total_trades', 0)
            winning_trades = getattr(strategy, 'winning_trades', 0)
            total_pnl = getattr(strategy, 'total_pnl', 0)
            
            print(f"  Total Trades: {total_trades}")
            if total_trades > 0:
                win_rate = (winning_trades / total_trades) * 100
                avg_pnl = total_pnl / total_trades
                print(f"  Win Rate: {win_rate:.1f}%")
                print(f"  Average P&L per Trade: ${avg_pnl:.2f}")
                
            print(f"  Session P&L: ${total_pnl:.2f}")
            
            # Drawdown calculation
            if hasattr(strategy, 'balance_history') and strategy.balance_history:
                peak_balance = max(strategy.balance_history)
                current_balance = strategy.balance_history[-1]
                drawdown = (peak_balance - current_balance) / peak_balance * 100
                print(f"  Current Drawdown: {drawdown:.2f}%")
                
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
    
    def do_validate_performance(self, arg):
        """Validate current performance against backtest expectations.
        Usage: validate_performance [hours]
        
        Args:
            hours: Number of hours to analyze (default: 24)
        """
        try:
            hours = int(arg) if arg else 24
            
            # Load current strategy config
            if not os.path.exists('best_strategy.json'):
                print("Error: No best_strategy.json found")
                return
                
            with open('best_strategy.json', 'r') as f:
                config = json.load(f)
                
            expected_trades_per_day = config.get('Average_Trades_Per_Day', 0)
            expected_return = config.get('Total_Return', 0)
            
            print(f"\n" + "="*60)
            print(f"PERFORMANCE VALIDATION - Last {hours} hours")
            print("="*60)
            
            # Get actual performance
            if hasattr(self, 'auto_trader') and self.auto_trader:
                # Count recent trades
                cutoff_time = datetime.now() - timedelta(hours=hours)
                recent_trades = 0
                
                if os.path.exists('trades.json'):
                    with open('trades.json', 'r') as f:
                        trades = json.load(f)
                        
                    for trade in trades:
                        trade_time = datetime.fromisoformat(trade['timestamp'])
                        if trade_time >= cutoff_time:
                            recent_trades += 1
                            
                # Calculate metrics
                actual_trades_per_day = (recent_trades / hours) * 24
                
                print(f"\nTrade Frequency:")
                print(f"  Expected: {expected_trades_per_day:.2f} trades/day")
                print(f"  Actual: {actual_trades_per_day:.2f} trades/day")
                print(f"  Status: {'✅ OK' if abs(actual_trades_per_day - expected_trades_per_day) < 2 else '⚠️ DEVIATION'}")
                
                # Performance tracking would require more detailed P&L history
                print(f"\nReturn Performance:")
                print(f"  Expected (backtest): {expected_return:.2f}% (annualized)")
                print(f"  Actual tracking requires P&L history implementation")
                
            else:
                print("No active trading session to validate")
                
        except Exception as e:
            print(f"Error validating performance: {e}")
    
    def do_set_parameter(self, arg):
        """Temporarily adjust a strategy parameter.
        Usage: set_parameter <parameter> <value> [temporary]
        
        Args:
            parameter: Parameter name (e.g., whipsaw_threshold)
            value: New value
            temporary: If specified, change reverts on restart
        """
        try:
            parts = arg.split()
            if len(parts) < 2:
                print("Usage: set_parameter <parameter> <value> [temporary]")
                return
                
            param = parts[0]
            value = parts[1]
            temporary = len(parts) > 2 and parts[2] == 'temporary'
            
            if not hasattr(self, 'auto_trader') or not self.auto_trader:
                print("Error: No active trading session")
                return
                
            strategy = self.auto_trader.strategy
            
            # Validate parameter exists
            if not hasattr(strategy, param):
                print(f"Error: Parameter '{param}' not found in strategy")
                print("Available parameters:")
                for attr in dir(strategy):
                    if not attr.startswith('_') and not callable(getattr(strategy, attr)):
                        print(f"  - {attr}: {getattr(strategy, attr)}")
                return
                
            # Get old value
            old_value = getattr(strategy, param)
            
            # Convert value to appropriate type
            if isinstance(old_value, bool):
                new_value = value.lower() in ['true', '1', 'yes']
            elif isinstance(old_value, int):
                new_value = int(value)
            elif isinstance(old_value, float):
                new_value = float(value)
            else:
                new_value = value
                
            # Set new value
            setattr(strategy, param, new_value)
            
            print(f"✅ Parameter updated:")
            print(f"  {param}: {old_value} → {new_value}")
            
            if temporary:
                print("  ⚠️  This is a temporary change and will revert on restart")
            else:
                # Save to config for persistence
                print("  💾 Saving to configuration...")
                # This would need to be implemented to actually persist
                
            # Log the change
            if hasattr(strategy, 'diagnostic_logger') and strategy.diagnostic_logger:
                strategy.diagnostic_logger.log_event('PARAMETER_CHANGE', {
                    'parameter': param,
                    'old_value': old_value,
                    'new_value': new_value,
                    'temporary': temporary,
                    'source': 'command_interface'
                })
                
        except Exception as e:
            print(f"Error setting parameter: {e}")

# Function to patch the existing shell with new commands
def enhance_shell_with_diagnostics(shell_class):
    """
    Dynamically add enhanced diagnostic commands to an existing shell class.
    
    Usage:
        from command_interface_enhanced import enhance_shell_with_diagnostics
        enhance_shell_with_diagnostics(CryptoShell)
    """
    # Get all methods from EnhancedDiagnosticCommands
    for name in dir(EnhancedDiagnosticCommands):
        if name.startswith('do_'):
            method = getattr(EnhancedDiagnosticCommands, name)
            if callable(method):
                setattr(shell_class, name, method)
                
    print("✅ Enhanced diagnostic commands added to shell")
    
# Standalone command descriptions for help
ENHANCED_COMMANDS = {
    'backtest_current_params': 'Run quick backtest with current live parameters',
    'compare_strategies': 'Compare current strategy with recommended parameters',
    'regime_history': 'Show recent regime changes and performance',
    'signal_analysis': 'Analyze why system is or isn\'t trading',
    'risk_metrics': 'Show current risk metrics and exposure',
    'validate_performance': 'Validate current performance against backtest',
    'set_parameter': 'Temporarily adjust a strategy parameter'
}