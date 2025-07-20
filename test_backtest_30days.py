#!/usr/bin/env python3
"""
Test backtesting system with 30 days of data
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
    print("Testing backtesting system with 30 days of data...")
    
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
    
    # Load data for last 30 days
    print("\nLoading 30 days of data...")
    data_manager = BacktestDataManager("btcusd.log")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
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
        print(f"\nBacktest Results:")
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
    if results['num_trades'] > 0:
        print("\nCalculating performance metrics...")
        metrics_calc = PerformanceMetrics()
        metrics = metrics_calc.calculate_all(
            equity_curve=results['equity_curve'],
            trades=results['trades'],
            signals=results['signals'],
            regime_history=results['regime_history']
        )
        
        # Display key metrics
        if 'risk' in metrics:
            print(f"\nRisk Metrics:")
            print(f"  Sharpe Ratio: {metrics['risk'].get('sharpe_ratio', 0):.3f}")
            print(f"  Max Drawdown: {metrics['drawdown_analysis'].get('max_drawdown_pct', 0):.2f}%")
        
        if 'win_loss_analysis' in metrics:
            print(f"\nTrading Performance:")
            print(f"  Win Rate: {metrics['win_loss_analysis'].get('win_rate', 0):.1%}")
            print(f"  Profit Factor: {metrics['win_loss_analysis'].get('profit_factor', 0):.2f}")
        
        # Show trades
        print(f"\nTrade History ({len(results['trades'])} trades):")
        for i, trade in enumerate(results['trades'][:10]):  # Show first 10
            print(f"{i+1}. {trade['timestamp']} - {trade['side'].upper()} "
                  f"{trade['amount']:.6f} BTC @ ${trade['price']:,.2f} "
                  f"({trade['market_regime']}) Fee: ${trade['fee']:.2f}")
        
        if len(results['trades']) > 10:
            print(f"... and {len(results['trades']) - 10} more trades")
    else:
        print("\nNo trades executed during backtest period.")
        print("This could be due to:")
        print("- Conservative strategy parameters")
        print("- Stable market conditions")
        print("- Not enough data for indicators")
    
    # Save results
    import json
    output_file = "backtest_results/test_30days.json"
    Path("backtest_results").mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())