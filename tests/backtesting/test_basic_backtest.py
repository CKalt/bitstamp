#!/usr/bin/env python3
"""
Basic integration test for backtesting system
Tests that the backtest runs and produces expected results
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from backtesting.core.engine import BacktestEngine, BacktestConfig
from backtesting.data.data_loader import BacktestDataManager
from backtesting.metrics.performance import PerformanceMetrics


def test_basic_backtest():
    """Test basic backtest functionality"""
    print("Testing basic backtest functionality...")
    
    # Create test configuration
    config = BacktestConfig(
        initial_btc=0.0,
        initial_usd=100000.0,
        fee_percentage=0.0012,
        max_trades_per_day=5,
        max_trades_per_hour=3,
        min_trade_gap_minutes=15,
        always_in_market=True
    )
    
    # Load test data (last 30 days)
    print("Loading data...")
    data_manager = BacktestDataManager("btcusd.log")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        data = data_manager.load_data(start_date=start_date, end_date=end_date)
        print(f"Loaded {len(data)} candles")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return False
    
    # Run backtest
    print("Running backtest...")
    engine = BacktestEngine(config)
    
    try:
        results = engine.run(data)
    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify results structure
    print("Verifying results...")
    required_keys = ['initial_value', 'final_value', 'total_return', 'num_trades', 
                     'trades', 'equity_curve', 'signals', 'config']
    
    for key in required_keys:
        if key not in results:
            print(f"Missing required key in results: {key}")
            return False
    
    # Basic sanity checks
    if results['initial_value'] != config.initial_usd:
        print(f"Initial value mismatch: {results['initial_value']} != {config.initial_usd}")
        return False
    
    if results['final_value'] <= 0:
        print(f"Invalid final value: {results['final_value']}")
        return False
    
    if len(results['trades']) > config.max_trades_per_day * 30:
        print(f"Too many trades: {len(results['trades'])}")
        return False
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics_calc = PerformanceMetrics()
    metrics = metrics_calc.calculate_all(
        equity_curve=results['equity_curve'],
        trades=results['trades'],
        signals=results['signals'],
        regime_history=results['regime_history']
    )
    
    # Display summary
    print("\nBacktest Summary:")
    print(f"Initial Capital: ${results['initial_value']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Number of Trades: {results['num_trades']}")
    
    if 'risk' in metrics and 'sharpe_ratio' in metrics['risk']:
        print(f"Sharpe Ratio: {metrics['risk']['sharpe_ratio']:.3f}")
    
    if 'win_loss_analysis' in metrics and metrics['win_loss_analysis']:
        print(f"Win Rate: {metrics['win_loss_analysis'].get('win_rate', 0):.1%}")
    
    # Test passed
    print("\n✓ Basic backtest test PASSED")
    return True


def test_fee_calculation():
    """Test that fees are calculated correctly"""
    print("\nTesting fee calculation...")
    
    config = BacktestConfig(
        initial_btc=0.0,
        initial_usd=10000.0,
        fee_percentage=0.0012,  # 0.12%
        always_in_market=True
    )
    
    # Create a simple backtest with known data
    from backtesting.core.engine import BacktestPositionTracker
    tracker = BacktestPositionTracker(config)
    
    # Test buy trade
    buy_trade = tracker.execute_trade(
        timestamp=datetime.now(),
        side='buy',
        price=50000.0,
        signal_reason='test',
        market_regime='test'
    )
    
    if buy_trade:
        expected_fee = buy_trade.amount * 50000.0 * 0.0012
        if abs(buy_trade.fee - expected_fee) > 0.01:
            print(f"Fee calculation error: {buy_trade.fee} != {expected_fee}")
            return False
        print(f"✓ Buy fee correct: ${buy_trade.fee:.2f}")
    
    # Test sell trade
    sell_trade = tracker.execute_trade(
        timestamp=datetime.now() + timedelta(hours=1),
        side='sell',
        price=51000.0,
        signal_reason='test',
        market_regime='test'
    )
    
    if sell_trade:
        expected_fee = sell_trade.amount * 51000.0 * 0.0012
        if abs(sell_trade.fee - expected_fee) > 0.01:
            print(f"Fee calculation error: {sell_trade.fee} != {expected_fee}")
            return False
        print(f"✓ Sell fee correct: ${sell_trade.fee:.2f}")
    
    print("✓ Fee calculation test PASSED")
    return True


def test_position_tracking():
    """Test position tracking logic"""
    print("\nTesting position tracking...")
    
    config = BacktestConfig(
        initial_btc=0.0,
        initial_usd=100000.0,
        always_in_market=True
    )
    
    from backtesting.core.engine import BacktestPositionTracker
    tracker = BacktestPositionTracker(config)
    
    # Initial state should be in USD (position = -1)
    if tracker.position != -1:
        print(f"Initial position should be -1 (USD), got {tracker.position}")
        return False
    
    # Buy BTC
    buy_trade = tracker.execute_trade(
        timestamp=datetime.now(),
        side='buy',
        price=50000.0,
        signal_reason='test',
        market_regime='test'
    )
    
    if tracker.position != 1:
        print(f"Position after buy should be 1 (BTC), got {tracker.position}")
        return False
    
    if tracker.balance_btc <= 0:
        print(f"BTC balance should be positive after buy, got {tracker.balance_btc}")
        return False
    
    if tracker.balance_usd != 0:
        print(f"USD balance should be 0 after full buy, got {tracker.balance_usd}")
        return False
    
    # Sell BTC
    sell_trade = tracker.execute_trade(
        timestamp=datetime.now() + timedelta(hours=1),
        side='sell',
        price=51000.0,
        signal_reason='test',
        market_regime='test'
    )
    
    if tracker.position != -1:
        print(f"Position after sell should be -1 (USD), got {tracker.position}")
        return False
    
    if tracker.balance_btc != 0:
        print(f"BTC balance should be 0 after sell, got {tracker.balance_btc}")
        return False
    
    if tracker.balance_usd <= 0:
        print(f"USD balance should be positive after sell, got {tracker.balance_usd}")
        return False
    
    print("✓ Position tracking test PASSED")
    return True


def test_trade_limits():
    """Test trade limit enforcement"""
    print("\nTesting trade limits...")
    
    config = BacktestConfig(
        initial_btc=0.0,
        initial_usd=100000.0,
        max_trades_per_day=2,
        max_trades_per_hour=1,
        min_trade_gap_minutes=15
    )
    
    from backtesting.core.engine import BacktestPositionTracker
    tracker = BacktestPositionTracker(config)
    
    base_time = datetime.now()
    
    # First trade should succeed
    trade1 = tracker.execute_trade(
        timestamp=base_time,
        side='buy',
        price=50000.0,
        signal_reason='test',
        market_regime='test'
    )
    
    if trade1 is None:
        print("First trade should succeed")
        return False
    
    # Second trade too soon should fail (min gap)
    trade2 = tracker.execute_trade(
        timestamp=base_time + timedelta(minutes=5),
        side='sell',
        price=50000.0,
        signal_reason='test',
        market_regime='test'
    )
    
    if trade2 is not None:
        print("Trade within min gap should fail")
        return False
    
    # Trade after min gap should succeed
    trade3 = tracker.execute_trade(
        timestamp=base_time + timedelta(minutes=20),
        side='sell',
        price=50000.0,
        signal_reason='test',
        market_regime='test'
    )
    
    if trade3 is None:
        print("Trade after min gap should succeed")
        return False
    
    # Third trade in same hour should fail (hourly limit)
    trade4 = tracker.execute_trade(
        timestamp=base_time + timedelta(minutes=40),
        side='buy',
        price=50000.0,
        signal_reason='test',
        market_regime='test'
    )
    
    if trade4 is not None:
        print("Third trade in hour should fail")
        return False
    
    print("✓ Trade limits test PASSED")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Running Backtest Integration Tests")
    print("=" * 60)
    
    tests = [
        test_basic_backtest,
        test_fee_calculation,
        test_position_tracking,
        test_trade_limits
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())