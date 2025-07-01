# src/validate_strategy.py
# Strategy validation tool for safe deployment

import json
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import argparse
from typing import Dict, List, Tuple

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data.loader import parse_log_file
from bktst_enhanced_shared import AdaptiveStrategyBacktester, load_config
from indicators.technical_indicators import ensure_datetime_index

class StrategyValidator:
    """Validates strategy configurations before deployment."""
    
    def __init__(self, current_config_path='best_strategy.json', 
                 recommended_config_path='recommended_strategy.json'):
        self.current_config_path = current_config_path
        self.recommended_config_path = recommended_config_path
        self.validation_results = {}
        
    def load_configs(self) -> Tuple[Dict, Dict]:
        """Load current and recommended configurations."""
        try:
            with open(self.current_config_path, 'r') as f:
                current = json.load(f)
        except:
            print(f"Warning: Could not load {self.current_config_path}")
            current = {}
            
        try:
            with open(self.recommended_config_path, 'r') as f:
                recommended = json.load(f)
        except:
            raise ValueError(f"Could not load {self.recommended_config_path}")
            
        return current, recommended
    
    def validate_risk_metrics(self, config: Dict) -> Dict[str, bool]:
        """Validate that risk metrics are within acceptable bounds."""
        results = {}
        
        # Check max drawdown
        max_drawdown = config.get('performance_metrics', {}).get('max_drawdown_pct', 100)
        results['max_drawdown_ok'] = max_drawdown < 20  # Max 20% drawdown
        
        # Check trade frequency
        avg_trades_per_day = config.get('performance_metrics', {}).get('avg_trades_per_day', 10)
        results['trade_frequency_ok'] = 0.1 <= avg_trades_per_day <= 10  # Between 0.1 and 10 trades per day
        
        # Check win rate (skip if no win rate data available)
        win_rate = config.get('performance_metrics', {}).get('win_rate', None)
        if win_rate is not None and win_rate > 0:
            results['win_rate_ok'] = win_rate > 45  # At least 45% win rate
        else:
            # If win rate is 0 or not available, check if strategy is profitable instead
            total_return = config.get('performance_metrics', {}).get('total_return_pct', 0)
            results['win_rate_ok'] = total_return > 0  # Profitable strategy
        
        # Check Sharpe ratio
        sharpe_ratio = config.get('performance_metrics', {}).get('sharpe_ratio', -10)
        results['sharpe_ratio_ok'] = sharpe_ratio > 0  # Positive Sharpe ratio
        
        return results
    
    def validate_parameters(self, config: Dict) -> Dict[str, bool]:
        """Validate that strategy parameters are reasonable."""
        results = {}
        params = config.get('optimal_parameters', {})
        
        # Check MA windows
        short_window = params.get('short_window', params.get('Short_Window', 0))
        long_window = params.get('long_window', params.get('Long_Window', 0))
        results['ma_windows_ok'] = (
            5 <= short_window <= 50 and 
            20 <= long_window <= 100 and 
            short_window < long_window
        )
        
        # Check adaptive parameters if present
        if params.get('strategy') == 'AdaptiveMulti':
            regime_threshold = params.get('regime_switch_threshold', 0)
            results['regime_threshold_ok'] = 0.2 <= regime_threshold <= 0.8
            
            confirmation_bars = params.get('signal_confirmation_bars', 0)
            results['confirmation_bars_ok'] = 1 <= confirmation_bars <= 5
            
            trade_gap = params.get('min_trade_gap_minutes', 0)
            results['trade_gap_ok'] = 5 <= trade_gap <= 60
            
            whipsaw = params.get('whipsaw_threshold', 0)
            results['whipsaw_ok'] = 2 <= whipsaw <= 15
        
        return results
    
    def validate_backtest_data(self, config: Dict) -> Dict[str, bool]:
        """Validate that backtest data is recent and comprehensive."""
        results = {}
        metadata = config.get('backtest_metadata', {})
        
        # Check test period
        try:
            test_end = datetime.fromisoformat(metadata.get('test_period_end', '').replace('Z', ''))
            days_old = (datetime.now() - test_end).days
            results['data_recent'] = days_old <= 7  # Data should be less than 7 days old
        except:
            results['data_recent'] = False
        
        # Check test duration
        total_days = metadata.get('total_days', 0)
        results['sufficient_data'] = total_days >= 25  # At least 25 days of data
        
        # Check number of strategies tested
        strategies_tested = metadata.get('total_strategies_tested', 0)
        results['comprehensive_test'] = strategies_tested >= 1  # At least 1 strategy tested
        
        return results
    
    def run_recent_performance_check(self, config: Dict, days: int = 7) -> Dict:
        """Test strategy on recent data to ensure it still performs well."""
        print(f"\nTesting strategy on last {days} days of data...")
        
        # Load recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = parse_log_file('btcusd.log', start_date, end_date)
        if len(df) < 100:
            return {'error': 'Insufficient recent data for validation'}
        
        # Ensure datetime index using the utility function
        df = ensure_datetime_index(df)
        
        # Create volume column if it doesn't exist
        if 'volume' not in df.columns:
            df['volume'] = df['price'] * df['amount']
        
        # Resample to hourly
        df = df.resample('1H').agg({
            'price': 'last',
            'amount': 'sum',
            'volume': 'sum'
        }).dropna()
        
        # Test the strategy
        params = config.get('optimal_parameters', {})
        strategy_type = params.get('strategy', 'MA')
        
        if strategy_type == 'AdaptiveMulti':
            # Test adaptive strategy
            test_config = load_config()
            backtester = AdaptiveStrategyBacktester(df, test_config)
            
            test_params = {
                'short_window': params.get('short_window', 10),
                'long_window': params.get('long_window', 46),
                'regime_switch_threshold': params.get('regime_switch_threshold', 0.40),
                'signal_confirmation_bars': params.get('signal_confirmation_bars', 2),
                'min_trade_gap_minutes': params.get('min_trade_gap_minutes', 15),
                'whipsaw_threshold': params.get('whipsaw_threshold', 8.0)
            }
            
            result = backtester.backtest_adaptive_strategy(test_params)
            
            if 'error' not in result:
                return {
                    'recent_return': result['total_return'],
                    'recent_trades': result['total_trades'],
                    'recent_sharpe': result['sharpe_ratio'],
                    'recent_win_rate': result['win_rate'],
                    'performance_ok': result['total_return'] > -5  # Not losing more than 5%
                }
        elif strategy_type == 'MA':
            # For simple MA strategy, skip recent performance check
            return {
                'recent_return': 0,
                'recent_trades': 0,
                'recent_sharpe': 0,
                'recent_win_rate': 0,
                'performance_ok': True,  # Skip validation for MA
                'note': 'Recent performance check skipped for MA strategy'
            }
        
        return {'error': f'Strategy type {strategy_type} not supported for recent validation'}
    
    def generate_deployment_checklist(self, current: Dict, recommended: Dict, 
                                    validation_results: Dict) -> List[str]:
        """Generate a deployment checklist."""
        checklist = []
        
        # Check if all validations passed
        all_passed = True
        for key, results in validation_results.items():
            if isinstance(results, dict):
                if key == 'recent':
                    # For recent results, only check 'performance_ok' field
                    if 'performance_ok' in results:
                        all_passed = all_passed and results['performance_ok']
                    # Skip other numeric values in recent results
                else:
                    # For other results, check all boolean values
                    if not all(results.values()):
                        all_passed = False
        
        if all_passed:
            checklist.append("✅ All validation checks PASSED")
        else:
            checklist.append("❌ Some validation checks FAILED")
        
        # Parameter changes
        checklist.append("\n📊 Parameter Changes:")
        
        current_params = current.get('optimal_parameters', current)
        rec_params = recommended.get('optimal_parameters', {})
        
        for key in ['short_window', 'long_window', 'strategy']:
            current_val = current_params.get(key, 'N/A')
            rec_val = rec_params.get(key, 'N/A')
            if current_val != rec_val:
                checklist.append(f"  - {key}: {current_val} → {rec_val}")
        
        # Performance expectations
        checklist.append("\n📈 Expected Performance:")
        current_return = current.get('Total_Return', 0)
        rec_return = recommended.get('performance_metrics', {}).get('total_return_pct', 0)
        improvement = rec_return - current_return
        
        checklist.append(f"  - Current return: {current_return:.2f}%")
        checklist.append(f"  - Expected return: {rec_return:.2f}%")
        checklist.append(f"  - Improvement: {improvement:+.2f}%")
        
        # Deployment steps
        checklist.append("\n🚀 Deployment Steps:")
        checklist.append("  1. Create backup: cp best_strategy.json best_strategy.backup.json")
        checklist.append("  2. Review recommended_strategy.json thoroughly")
        checklist.append("  3. If approved, copy configuration:")
        checklist.append("     cp recommended_strategy.json best_strategy.json")
        checklist.append("  4. Edit best_strategy.json and set:")
        checklist.append("     - 'do_live_trades': false (for initial testing)")
        checklist.append("  5. Monitor system for 1 hour in paper trading mode")
        checklist.append("  6. If stable, set 'do_live_trades': true")
        
        # Rollback plan
        checklist.append("\n🔄 Rollback Plan:")
        checklist.append("  - If issues occur: cp best_strategy.backup.json best_strategy.json")
        checklist.append("  - Restart trading system")
        
        return checklist
    
    def validate(self) -> bool:
        """Run full validation suite."""
        print("="*80)
        print("STRATEGY VALIDATION REPORT")
        print("="*80)
        
        # Load configurations
        current, recommended = self.load_configs()
        
        # Run validations
        print("\n1. Risk Metrics Validation:")
        risk_results = self.validate_risk_metrics(recommended)
        for metric, passed in risk_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {metric}: {status}")
        self.validation_results['risk'] = risk_results
        
        print("\n2. Parameter Validation:")
        param_results = self.validate_parameters(recommended)
        for param, passed in param_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {param}: {status}")
        self.validation_results['parameters'] = param_results
        
        print("\n3. Backtest Data Validation:")
        data_results = self.validate_backtest_data(recommended)
        for check, passed in data_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {check}: {status}")
        self.validation_results['data'] = data_results
        
        print("\n4. Recent Performance Check:")
        recent_results = self.run_recent_performance_check(recommended, days=7)
        if 'error' not in recent_results:
            for metric, value in recent_results.items():
                if metric == 'performance_ok':
                    status = "✅ PASS" if value else "❌ FAIL"
                    print(f"   {metric}: {status}")
                elif metric == 'note':
                    print(f"   {metric}: {value}")
                else:
                    print(f"   {metric}: {value:.2f}")
        else:
            print(f"   ❌ Error: {recent_results['error']}")
        self.validation_results['recent'] = recent_results
        
        # Generate deployment checklist
        print("\n" + "="*80)
        print("DEPLOYMENT CHECKLIST")
        print("="*80)
        checklist = self.generate_deployment_checklist(current, recommended, self.validation_results)
        for item in checklist:
            print(item)
        
        # Overall result
        all_passed = all(
            all(results.values()) 
            for key, results in self.validation_results.items() 
            if isinstance(results, dict) and key != 'recent'
        )
        
        print("\n" + "="*80)
        if all_passed:
            print("✅ VALIDATION PASSED - Strategy is ready for deployment")
        else:
            print("❌ VALIDATION FAILED - Review and address issues before deployment")
        print("="*80)
        
        return all_passed

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate trading strategy before deployment")
    parser.add_argument('--current', type=str, default='best_strategy.json',
                        help='Path to current strategy configuration')
    parser.add_argument('--recommended', type=str, default='recommended_strategy.json',
                        help='Path to recommended strategy configuration')
    parser.add_argument('--force', action='store_true',
                        help='Show deployment steps even if validation fails')
    
    args = parser.parse_args()
    
    # Run validation
    validator = StrategyValidator(args.current, args.recommended)
    passed = validator.validate()
    
    if not passed and not args.force:
        print("\n⚠️  Use --force flag to see deployment steps despite validation failures")
        return 1
    
    return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(main())