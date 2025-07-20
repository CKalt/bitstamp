#!/usr/bin/env python3
"""
Main backtest runner script
Provides a clean interface for running backtests with configuration files
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from backtesting.core.engine import BacktestEngine, BacktestConfig
from backtesting.core.config import BacktestConfigManager
from backtesting.data.data_loader import BacktestDataManager, BacktestDataValidator
from backtesting.metrics.performance import PerformanceMetrics


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('backtest.log')
        ]
    )


def main():
    """Main backtest execution"""
    parser = argparse.ArgumentParser(description='Run cryptocurrency trading backtest')
    
    # Configuration options
    parser.add_argument('--config', type=str, default='config/strategies/adaptive_default.yaml',
                      help='Path to configuration file (YAML or JSON)')
    
    # Date range options (override config file)
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--days-back', type=int, 
                      help='Alternative to start-date: backtest last N days')
    
    # Timeframe options (matching main branch functionality)
    parser.add_argument('--low-frequency', type=str, default='15T',
                      help='Primary timeframe for trading signals (default: 15T for 15-minute)')
    parser.add_argument('--high-frequency', type=str, default='1H',
                      help='Secondary timeframe for regime detection (default: 1H for 1-hour)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--output-file', type=str, 
                      help='Specific output file for results (JSON)')
    parser.add_argument('--no-save', action='store_true',
                      help='Do not save results to file')
    
    # Display options
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--show-trades', action='store_true', 
                      help='Display all trades in output')
    
    # Quick presets
    parser.add_argument('--quick', action='store_true',
                      help='Quick test: last 7 days')
    parser.add_argument('--month', action='store_true',
                      help='Test last 30 days')
    parser.add_argument('--year', action='store_true',
                      help='Test last 365 days')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_manager = BacktestConfigManager()
    
    try:
        if args.config.endswith('.yaml') or args.config.endswith('.yml'):
            config_schema = config_manager.load_yaml(args.config)
        else:
            config_schema = config_manager.load_json(args.config)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Apply command line overrides
    if args.days_back or args.quick or args.month or args.year:
        # Calculate start date based on days back
        if args.quick:
            days = 7
        elif args.month:
            days = 30
        elif args.year:
            days = 365
        else:
            days = args.days_back
        
        config_schema.start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        config_schema.end_date = datetime.now().strftime('%Y-%m-%d')
    
    if args.start_date:
        config_schema.start_date = args.start_date
    if args.end_date:
        config_schema.end_date = args.end_date
    if args.output_dir:
        config_schema.output_dir = args.output_dir
    
    # Validate configuration
    issues = config_manager.validate_config(config_schema)
    if issues:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        sys.exit(1)
    
    # Convert to BacktestConfig for engine
    backtest_config = BacktestConfig(
        initial_btc=config_schema.initial_btc,
        initial_usd=config_schema.initial_usd,
        fee_percentage=config_schema.fee_percentage,
        max_trades_per_day=config_schema.max_trades_per_day,
        max_trades_per_hour=config_schema.max_trades_per_hour,
        min_trade_gap_minutes=config_schema.min_trade_gap_minutes,
        min_btc_trade_size=config_schema.min_btc_trade_size,
        always_in_market=config_schema.always_in_market,
        enable_pivot_protection=config_schema.enable_pivot_protection,
        enable_trailing_stops=config_schema.enable_trailing_stops,
        emergency_exit_loss=config_schema.emergency_exit_loss
    )
    
    # Load data
    logger.info("Loading historical data...")
    data_manager = BacktestDataManager(
        config_schema.data_source,
        primary_timeframe=args.low_frequency,
        secondary_timeframe=args.high_frequency
    )
    
    try:
        # Parse dates if provided
        start_dt = datetime.fromisoformat(config_schema.start_date) if config_schema.start_date else None
        end_dt = datetime.fromisoformat(config_schema.end_date) if config_schema.end_date else None
        
        # Load data with progress callback
        def progress_callback(current, total):
            if not args.quiet:
                pct = (current / total) * 100
                print(f"\rLoading data: {current:,}/{total:,} ({pct:.1f}%)", end='', flush=True)
        
        data = data_manager.load_data(
            start_date=start_dt,
            end_date=end_dt,
            progress_callback=progress_callback if not args.quiet else None
        )
        
        if not args.quiet:
            print()  # New line after progress
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Validate data
    validator = BacktestDataValidator()
    data_issues = validator.validate_ohlcv(data)
    if data_issues:
        logger.warning("Data validation warnings:")
        for issue in data_issues:
            logger.warning(f"  - {issue}")
    
    # Display data info
    data_info = data_manager.get_data_info()
    if not args.quiet:
        logger.info(f"Loaded {data_info.get(f'{args.low_frequency}_candles', 0)} {args.low_frequency} candles")
        logger.info(f"Date range: {data_info['date_range']['start']} to {data_info['date_range']['end']}")
        logger.info(f"Price range: ${data_info['price_range']['min']:.2f} to ${data_info['price_range']['max']:.2f}")
    
    # Run backtest
    logger.info("Running backtest...")
    engine = BacktestEngine(backtest_config)
    
    try:
        results = engine.run(data)
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Calculate performance metrics
    logger.info("Calculating performance metrics...")
    metrics_calculator = PerformanceMetrics(initial_capital=backtest_config.initial_usd)
    
    metrics = metrics_calculator.calculate_all(
        equity_curve=results['equity_curve'],
        trades=results['trades'],
        signals=results['signals'],
        regime_history=results['regime_history']
    )
    
    # Add metrics to results
    results['metrics'] = metrics
    
    # Display summary
    if not args.quiet:
        print("\n" + "="*60)
        print(metrics_calculator.generate_summary_report(metrics))
        print("="*60)
    
    # Display trades if requested
    if args.show_trades and results['trades']:
        print("\nTrade History:")
        print("-" * 100)
        print(f"{'Timestamp':<20} {'Side':<5} {'Price':>10} {'Amount':>12} {'Fee':>8} {'P&L':>10} {'Regime':<10}")
        print("-" * 100)
        
        cumulative_pnl = 0
        for i, trade in enumerate(results['trades']):
            # Calculate P&L for sells
            if i > 0 and trade['side'] == 'sell':
                buy_cost = results['trades'][i-1]['total_cost']
                sell_proceeds = trade['total_cost']
                pnl = sell_proceeds - buy_cost
                cumulative_pnl += pnl
                pnl_str = f"${pnl:,.2f}"
            else:
                pnl_str = "-"
            
            print(f"{trade['timestamp']:<20} {trade['side']:<5} "
                  f"${trade['price']:>9,.2f} {trade['amount']:>12.6f} "
                  f"${trade['fee']:>7,.2f} {pnl_str:>10} {trade['market_regime']:<10}")
        
        print("-" * 100)
        print(f"Total P&L: ${cumulative_pnl:,.2f}")
    
    # Save results
    if not args.no_save:
        # Determine output path
        if args.output_file:
            output_path = Path(args.output_file)
        else:
            # Create timestamped filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(config_schema.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"backtest_{config_schema.strategy.type}_{timestamp}.json"
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")
        
        # Save configuration used
        config_path = output_path.with_suffix('.config.yaml')
        config_manager.save_yaml(config_schema, str(config_path))
        logger.info(f"Configuration saved to: {config_path}")
    
    # Return success
    return 0


if __name__ == '__main__':
    sys.exit(main())