# Backtesting System Documentation

This backtesting system provides accurate historical simulation of the live trading system, ensuring exact behavior matching for reliable strategy evaluation.

## Key Features

- **Exact Live System Replication**: Uses the same `AdaptiveStrategyCore` as live trading
- **Accurate Fee Modeling**: 0.12% Bitstamp fees applied exactly as in live trading
- **Position Tracking**: Always 100% invested (BTC or USD), matching live behavior
- **Trade Limits**: Enforces daily/hourly limits and minimum gaps between trades
- **Comprehensive Metrics**: Sharpe ratio, drawdown analysis, win rate, and more
- **YAML Configuration**: Easy parameter management and testing
- **Fast Data Loading**: Efficient processing of years of tick data

## Quick Start

### 1. Basic Usage

```bash
# Quick 7-day test
./run_backtest.sh --quick

# Last 30 days with trade details
./run_backtest.sh --month --trades

# Full year backtest
./run_backtest.sh --year

# Specific date range
./run_backtest.sh --start 2024-01-01 --end 2024-12-31
```

### 2. Using Python Directly

```python
from backtesting.core.engine import BacktestEngine, BacktestConfig
from backtesting.data.data_loader import BacktestDataManager
from backtesting.metrics.performance import PerformanceMetrics

# Load configuration
config = BacktestConfig(
    initial_usd=100000.0,
    fee_percentage=0.0012,  # 0.12% Bitstamp fee
    max_trades_per_day=5,
    always_in_market=True
)

# Load data
data_manager = BacktestDataManager("btcusd.log")
data = data_manager.load_data(start_date=start_date, end_date=end_date)

# Run backtest
engine = BacktestEngine(config)
results = engine.run(data)

# Calculate metrics
metrics_calc = PerformanceMetrics()
metrics = metrics_calc.calculate_all(
    equity_curve=results['equity_curve'],
    trades=results['trades'],
    signals=results['signals'],
    regime_history=results['regime_history']
)
```

## Configuration

### YAML Configuration Files

Configuration files are stored in `config/strategies/`. Example:

```yaml
name: "Adaptive Multi-Strategy - Conservative"
description: "Lower risk configuration"

# Capital settings
initial_usd: 100000.0
fee_percentage: 0.0012

# Trade limits
max_trades_per_day: 3
max_trades_per_hour: 2
min_trade_gap_minutes: 30

# Strategy parameters
regime_detection:
  confidence_threshold: 0.7
  
trending_strategy:
  short_window: 10
  long_window: 30
  confirmation_bars: 3
```

### Available Configurations

- `adaptive_default.yaml`: Default parameters matching live system
- `adaptive_conservative.yaml`: Lower risk, fewer trades
- `adaptive_aggressive.yaml`: Higher risk, more active trading

## Running Tests

```bash
# Run integration tests
python tests/backtesting/test_basic_backtest.py

# Test specific functionality
python -m pytest tests/backtesting/
```

## Output Files

Results are saved to `backtest_results/` with timestamps:
- `backtest_adaptive_20241231_143022.json`: Full results with trades and metrics
- `backtest_adaptive_20241231_143022.config.yaml`: Configuration used

## Performance Metrics

The system calculates comprehensive metrics including:

### Returns
- Total return and annualized return
- Monthly return statistics
- Best/worst periods

### Risk Metrics
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Value at Risk (95%)
- Calmar ratio

### Trading Analysis
- Win rate
- Profit factor
- Average win/loss ratio
- Trade frequency
- Regime performance breakdown

### Example Output

```
=== BACKTEST PERFORMANCE SUMMARY ===

Initial Capital: $100,000.00
Final Value: $125,432.50
Total Return: 25.43%
Annualized Return: 23.87%
Trading Period: 365 days

Risk Metrics:
  Sharpe Ratio: 1.234
  Sortino Ratio: 1.567
  Calmar Ratio: 0.892
  Annual Volatility: 18.45%
  Value at Risk (95%): -2.34%

Drawdown Analysis:
  Max Drawdown: -12.34%
  Drawdown Duration: 45 days
  Recovery Duration: 23 days

Trading Activity:
  Total Trades: 234
  Trades per Day: 0.64
  Total Fees Paid: $3,234.56

Win/Loss Analysis:
  Win Rate: 54.3%
  Profit Factor: 1.45
  Avg Win/Loss Ratio: 1.23
  Average P&L per Trade: $108.23
  Kelly Criterion: 8.7%
```

## Architecture

### Core Components

1. **BacktestEngine** (`core/engine.py`)
   - Orchestrates the backtest simulation
   - Uses existing `AdaptiveStrategyCore` for signals
   - Manages position tracking and trade execution

2. **BacktestDataManager** (`data/data_loader.py`)
   - Loads and prepares historical data from btcusd.log
   - Converts ticks to 1-minute then 1-hour candles
   - Provides tick simulation for realistic execution

3. **PerformanceMetrics** (`metrics/performance.py`)
   - Calculates comprehensive performance metrics
   - Analyzes trades, returns, and risk
   - Generates human-readable reports

4. **Configuration System** (`core/config.py`)
   - YAML-based configuration management
   - Parameter validation
   - Easy A/B testing setup

### Data Flow

```
btcusd.log → Parse Trades → 1-min Candles → 1-hour Candles → Strategy → Signals → Trades → Metrics
```

## Important Notes

1. **Exact Behavior Match**: This backtester uses the SAME strategy code as live trading
2. **No Look-Ahead Bias**: Data is processed sequentially as in live trading
3. **Realistic Execution**: Fees and position tracking match live system exactly
4. **Safety First**: Never overwrites production files without explicit paths

## Troubleshooting

### Common Issues

1. **"btcusd.log not found"**
   - Ensure you're running from the project root directory
   - Check that btcusd.log exists and has data

2. **"Configuration file not found"**
   - Use relative paths from project root
   - Check config/strategies/ directory

3. **Memory issues with large datasets**
   - Use date ranges to limit data
   - Process in chunks for multi-year backtests

4. **Import errors**
   - Run from project root directory
   - Ensure Python path includes src/

## Future Enhancements

1. **Parameter Optimization**: Grid search and Bayesian optimization
2. **Walk-Forward Analysis**: Rolling window optimization
3. **Monte Carlo Simulation**: Robustness testing
4. **Multi-Strategy Portfolios**: Combine multiple strategies
5. **Real-time Comparison**: Compare backtest vs live performance