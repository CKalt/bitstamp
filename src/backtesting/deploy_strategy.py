#!/usr/bin/env python3
"""
Deploy backtest results to live trading
Converts backtest configuration to best_strategy.json format
"""
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
import shutil

def convert_backtest_to_live_format(backtest_config):
    """
    Convert new backtest config format to legacy best_strategy.json format
    """
    # Extract parameters from nested config
    regime = backtest_config.get('regime_detection', {})
    trending = backtest_config.get('trending_strategy', {})
    ranging = backtest_config.get('ranging_strategy', {})
    volatile = backtest_config.get('volatile_strategy', {})
    
    # Map to legacy format expected by live system
    legacy_format = {
        # MA Crossover parameters (from trending strategy)
        "short_window": trending.get('short_window', 10),
        "long_window": trending.get('long_window', 30),
        
        # RSI parameters (from ranging strategy)
        "rsi_threshold": ranging.get('rsi_oversold', 30),
        "rsi_overbought": ranging.get('rsi_overbought', 70),
        "rsi_window": ranging.get('rsi_window', 14),
        
        # Bollinger Band parameters
        "bb_window": ranging.get('bb_window', 20),
        "bb_std_dev": ranging.get('bb_std_dev', 2.0),
        
        # MACD parameters (from volatile strategy)
        "macd_fast": volatile.get('macd_fast', 12),
        "macd_slow": volatile.get('macd_slow', 26),
        "macd_signal": volatile.get('macd_signal', 9),
        
        # Regime detection parameters
        "lookback_bars": regime.get('lookback_bars', 100),
        "whipsaw_threshold": regime.get('whipsaw_threshold', 0.65),
        "trend_strength_threshold": regime.get('trend_strength_threshold', 0.3),
        "confidence_threshold": regime.get('confidence_threshold', 0.6),
        
        # Strategy type
        "strategy": "multi",  # Always multi for adaptive strategy
        
        # Additional parameters
        "confirmation_bars": trending.get('confirmation_bars', 2),
        "exit_at_opposite_band": ranging.get('exit_at_opposite_band', True)
    }
    
    return legacy_format


def main():
    parser = argparse.ArgumentParser(
        description='Deploy backtest results to live trading system'
    )
    
    parser.add_argument(
        'backtest_file',
        help='Path to backtest results JSON file'
    )
    
    parser.add_argument(
        '--output',
        default='best_strategy.json',
        help='Output file (default: best_strategy.json)'
    )
    
    parser.add_argument(
        '--min-sharpe',
        type=float,
        default=0.5,
        help='Minimum Sharpe ratio required for deployment (default: 0.5)'
    )
    
    parser.add_argument(
        '--min-win-rate',
        type=float,
        default=0.45,
        help='Minimum win rate required for deployment (default: 0.45)'
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup of existing best_strategy.json'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Deploy even if metrics are below thresholds'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deployed without actually doing it'
    )
    
    args = parser.parse_args()
    
    # Load backtest results
    try:
        with open(args.backtest_file, 'r') as f:
            backtest_results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Backtest file not found: {args.backtest_file}")
        return 1
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in backtest file: {args.backtest_file}")
        return 1
    
    # Extract metrics and config
    metrics = backtest_results.get('metrics', {})
    config = backtest_results.get('config', {})
    
    # Check performance thresholds
    sharpe = metrics.get('risk', {}).get('sharpe_ratio', 0)
    win_rate = metrics.get('win_loss_analysis', {}).get('win_rate', 0)
    total_return = backtest_results.get('total_return', 0)
    
    print("Backtest Performance Summary:")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.3f} (threshold: {args.min_sharpe})")
    print(f"  Win Rate: {win_rate:.1%} (threshold: {args.min_win_rate:.1%})")
    print()
    
    # Check if metrics meet thresholds
    if not args.force:
        if sharpe < args.min_sharpe:
            print(f"❌ Sharpe ratio {sharpe:.3f} is below threshold {args.min_sharpe}")
            print("   Use --force to deploy anyway")
            return 1
        
        if win_rate < args.min_win_rate:
            print(f"❌ Win rate {win_rate:.1%} is below threshold {args.min_win_rate:.1%}")
            print("   Use --force to deploy anyway")
            return 1
    
    # Convert to live format
    live_config = convert_backtest_to_live_format(config)
    
    # Add metadata
    live_config['_metadata'] = {
        'deployed_from': str(Path(args.backtest_file).absolute()),
        'deployed_at': datetime.now().isoformat(),
        'backtest_performance': {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'num_trades': backtest_results.get('num_trades', 0)
        }
    }
    
    if args.dry_run:
        print("Would deploy the following configuration:")
        print(json.dumps(live_config, indent=2))
        return 0
    
    # Backup existing file if requested
    if args.backup and Path(args.output).exists():
        backup_name = f"{args.output}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(args.output, backup_name)
        print(f"✓ Created backup: {backup_name}")
    
    # Write new configuration
    with open(args.output, 'w') as f:
        json.dump(live_config, f, indent=2)
    
    print(f"✓ Successfully deployed configuration to {args.output}")
    print()
    print("⚠️  IMPORTANT: The auto trade system will use these parameters on next restart")
    print("   Monitor initial trades carefully to ensure expected behavior")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())