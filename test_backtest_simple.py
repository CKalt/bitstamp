#!/usr/bin/env python3
"""
Simple test of backtesting system
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path BEFORE any imports that might need it
src_path = str(Path(__file__).parent / 'src')
sys.path.insert(0, src_path)

# Import data.loader first to ensure it's available
import data.loader

# Now import everything we need
from backtesting.core.engine import BacktestEngine, BacktestConfig
from backtesting.data.data_loader import BacktestDataManager
from backtesting.metrics.performance import PerformanceMetrics

def main():
    print("Testing backtesting system...")
    
    # Configuration
    config = BacktestConfig(
        initial_btc=0.0,
        initial_usd=100000.0,
        fee_percentage=0.0012,
        max_trades_per_day=5,
        max_trades_per_hour=3,
        min_trade_gap_minutes=15,
        always_in_market=True
    )
    
    # Load data for last 7 days
    print("\nLoading data...")
    data_manager = BacktestDataManager("btcusd.log")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    try:
        data = data_manager.load_data(start_date=start_date, end_date=end_date)
        print(f"Loaded {len(data)} hourly candles")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print(f"Price range: ${data['low'].min():.2f} to ${data['high'].max():.2f}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return 1
    
    # Run backtest
    print("\nRunning backtest...")
    engine = BacktestEngine(config)
    
    try:
        results = engine.run(data)
        print(f"Initial value: ${results['initial_value']:,.2f}")
        print(f"Final value: ${results['final_value']:,.2f}")
        print(f"Total return: {results['total_return']:.2%}")
        print(f"Number of trades: {results['num_trades']}")
    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics_calc = PerformanceMetrics()
    metrics = metrics_calc.calculate_all(
        equity_curve=results['equity_curve'],
        trades=results['trades'],
        signals=results['signals'],
        regime_history=results['regime_history']
    )
    
    # Display summary
    print("\n" + "="*60)
    print(metrics_calc.generate_summary_report(metrics))
    print("="*60)
    
    # Show some trades
    if results['trades']:
        print("\nFirst 5 trades:")
        for i, trade in enumerate(results['trades'][:5]):
            print(f"{i+1}. {trade['timestamp']} - {trade['side'].upper()} "
                  f"{trade['amount']:.6f} BTC @ ${trade['price']:,.2f} "
                  f"(fee: ${trade['fee']:.2f})")
    
    print("\nBacktest completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())