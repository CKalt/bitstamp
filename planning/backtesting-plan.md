Here is the planning/backtesting-plan.md document referenced above. You can
see this file in the planning directory as well.

# Backtesting and Auto Trade System Enhancement Plan

## CRITICAL: Main Branch vs Current Branch Backtesting Comparison

### Overview of Changes

The backtesting system has undergone significant evolution between the `main` branch and the current `stable-added-adaptive-trad-n-chart-more` branch. This section documents these critical differences.

### Main Branch Backtesting Approach

In the `main` branch, backtesting was performed using:

1. **Simple run scripts** (`run1.sh`, `run2.sh`, `run3.sh`, `run4.sh`):
   ```bash
   # run1.sh - Test 90-60 days ago
   python src/bktst.py \
     --start-window-days-back 90 \
     --end-window-days-back 60 \
     --high-frequency 1H \
     --low-frequency 15T
   
   # run2.sh - Test 60-30 days ago  
   python src/bktst.py \
     --start-window-days-back 60 \
     --end-window-days-back 30 \
     --high-frequency 1H \
     --low-frequency 15T
   
   # run3.sh - Test last 30 days
   python src/bktst.py \
     --start-window-days-back 30 \
     --end-window-days-back 0 \
     --high-frequency 1H \
     --low-frequency 15T
   ```

2. **Key characteristics of main branch `bktst.py`**:
   - Used `utils.analysis.run_trading_system` for optimization
   - Focused on MA crossover and RSI strategies
   - Simple parameter optimization (MA windows, RSI thresholds)
   - Direct output to `best_strategy.json` (dangerous for production)
   - No pivot protection simulation
   - No adaptive multi-strategy support
   - Basic fee structure (flat percentage)

3. **Limitations**:
   - No simulation of actual trading system features
   - Couldn't test pivot protection or trailing stops
   - No regime detection testing
   - Results often didn't match live trading performance

### Current Branch Enhanced Backtesting

The current branch has completely redesigned the backtesting system:

1. **New Architecture**:
   - **Unified Code Base**: Uses SAME strategy code as live trading (`AdaptiveStrategyCore`)
   - **Complete Feature Parity**: Simulates ALL live trading features including:
     - Pivot protection with sticky levels
     - Profit-aware trailing pivots
     - Adaptive multi-strategy regime switching
     - 100% position flips (always fully invested)
     - Realistic Bitstamp fees (0.12%) and slippage

2. **New File Structure**:
   ```
   src/bktst.py          # Enhanced backtester with full pivot logic
   src/backtest.py       # Uses shared AdaptiveStrategyCore
   run_backtest.sh       # Helper script with safety features
   compare_strategies.py # Strategy comparison tool
   ```

3. **Safety Improvements**:
   - **Never overwrites `best_strategy.json` by default**
   - Requires explicit `--output-file` parameter
   - Timestamped output files prevent accidents
   - Clear warnings before production changes

4. **Enhanced Testing Scripts**:
   ```bash
   # New approach with safety
   ./run_backtest.sh --quick    # Last 7 days test
   ./run_backtest.sh --month    # Last 30 days
   ./run_backtest.sh --year     # Full year backtest
   
   # Or direct with output file (REQUIRED)
   python src/bktst.py \
     --start-window-days-back 90 \
     --output-file test_results.json  # MUST specify output
   ```

5. **New Features in Backtesting**:
   - **Pivot Protection Simulation**: 
     - Sticky support/resistance levels
     - Profit-aware trailing that only moves favorably
     - Accurate trigger simulation
   - **Regime Detection**:
     - TRENDING/RANGING/VOLATILE classification
     - Strategy switching based on market conditions
   - **Realistic Execution**:
     - Bitstamp's actual fee structure (0.12%)
     - Slippage modeling
     - Position size constraints

### Migration Guide

If you have existing scripts using the old approach:

1. **Replace old run scripts**:
   ```bash
   # Old way (DANGEROUS - overwrites production config)
   ./run1.sh
   
   # New way (SAFE - explicit output file)
   python src/bktst.py \
     --start-window-days-back 90 \
     --end-window-days-back 60 \
     --output-file backtest_90_60.json
   ```

2. **Update result analysis**:
   - Old: Results directly in `best_strategy.json`
   - New: Results in specified output file
   - New: More detailed metrics including pivot performance

3. **Parameter differences**:
   - Old: Simple MA windows and RSI thresholds
   - New: Complex adaptive strategy parameters
   - New: Pivot protection parameters

### Critical Warnings

⚠️ **NEVER run old-style backtests that overwrite `best_strategy.json`**
⚠️ **The enhanced backtester gives DIFFERENT (more accurate) results**
⚠️ **Old backtest results may not reflect actual trading performance**

### Summary of Improvements

1. **Accuracy**: New backtester mirrors live trading exactly
2. **Safety**: Cannot accidentally overwrite production config
3. **Features**: Tests all advanced features (pivots, regimes, etc.)
4. **Realism**: Proper fee and slippage modeling
5. **Analysis**: Detailed performance breakdowns by regime

The current branch's backtesting system is a complete rewrite that provides accurate simulation of the live trading system, unlike the simplified approach in the main branch.

## Current System Overview

### 1. Architecture
The current system consists of several interconnected components:

```
btcusd.log → Data Loader → Strategy Engine → Auto Trader → Trade Execution
     ↓            ↓             ↓                ↓              ↓
  Historical   Resampled    Signals &      Position      trades.json
    Data       Dataframes   Indicators     Management    
```

### 2. Key Components

#### Data Pipeline
- **Source**: `btcusd.log` contains raw trade data (timestamp, price, amount)
- **Loading**: `src/data/loader.py` reads historical data with configurable timeframes
- **Resampling**: Converts tick data to 1-minute OHLCV candles
- **Storage**: In-memory DataFrames with optional file caching

#### Strategy System
- **Base Class**: `TradingStrategy` in `src/tdr_core/strategy_core.py`
- **Adaptive Strategy**: `AdaptiveMultiStrategy` switches between:
  - TRENDING: MA crossover strategy
  - RANGING: Bollinger Band mean reversion
  - VOLATILE: MACD breakout strategy
- **Regime Detection**: Uses whipsaw ratio, trend strength, and volatility metrics

#### Position Management
- **Tracking**: Maintains position, entry price, P&L, fees
- **Constraints**: Daily trade limits, minimum gap between trades
- **Resume**: Saves state to `resume-auto-trade.json` for restart capability

### 3. Current Limitations

1. **No True Backtesting Engine**
   - Current "backtesting" runs live strategy on historical data
   - No separation between backtest and live trading logic
   - Cannot easily test parameter variations

2. **Limited Parameter Control**
   - Parameters scattered across multiple files
   - No central configuration for strategy variations
   - Difficult to A/B test different approaches

3. **Insufficient Metrics**
   - Basic P&L and win rate only
   - No risk metrics (Sharpe, max drawdown, etc.)
   - No performance attribution by market regime

4. **No Walk-Forward Analysis**
   - Cannot validate strategy on out-of-sample data
   - No rolling window optimization
   - Risk of overfitting to historical data

## Proposed Improvements

### 1. Dedicated Backtesting Engine

Create `src/backtesting/engine.py`:

```python
class BacktestEngine:
    """
    Separate backtesting engine that simulates trading without affecting live state
    """
    def __init__(self, strategy_class, config):
        self.strategy_class = strategy_class
        self.config = config
        self.results = []
        
    def run(self, data, start_date, end_date):
        # Simulate trading on historical data
        # Track all metrics and decisions
        # Return comprehensive results
        
    def optimize(self, param_grid, metric='sharpe'):
        # Grid search over parameter combinations
        # Find optimal parameters for given metric
```

### 2. Centralized Configuration System

Create `config/strategies/` directory with YAML configs:

```yaml
# config/strategies/adaptive_multi.yaml
name: "Adaptive Multi-Strategy"
version: "2.0"

global:
  initial_capital: 100000
  position_sizing: "full"  # or "kelly", "fixed_risk"
  max_daily_trades: 10
  min_trade_gap_minutes: 15

regime_detection:
  lookback_bars: 100
  whipsaw_threshold: 0.65
  trend_strength_threshold: 0.3
  volatility_window: 50
  confidence_threshold: 0.6

strategies:
  trending:
    type: "ma_crossover"
    short_window: 10
    long_window: 30
    confirmation_bars: 2
    
  ranging:
    type: "bollinger_mean_reversion"
    bb_window: 20
    bb_std_dev: 2.0
    rsi_window: 14
    rsi_oversold: 30
    rsi_overbought: 70
    exit_at_opposite_band: true  # New feature
    
  volatile:
    type: "macd_breakout"
    macd_fast: 12
    macd_slow: 26
    macd_signal: 9
    breakout_threshold: 100

risk_management:
  stop_loss_pct: 0.02  # 2% stop loss
  take_profit_pct: 0.05  # 5% take profit
  trailing_stop: 
    enabled: true
    activation_profit: 0.01  # Activate after 1% profit
    trail_distance: 0.005  # Trail by 0.5%
```

### 3. Enhanced Metrics and Reporting

Create `src/backtesting/metrics.py`:

```python
class PerformanceMetrics:
    """Calculate comprehensive trading metrics"""
    
    def calculate_all(self, trades, equity_curve):
        return {
            # Returns
            'total_return': self.total_return(),
            'annualized_return': self.annualized_return(),
            'monthly_returns': self.monthly_returns(),
            
            # Risk
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'max_drawdown': self.max_drawdown(),
            'var_95': self.value_at_risk(0.95),
            
            # Trading
            'win_rate': self.win_rate(),
            'profit_factor': self.profit_factor(),
            'avg_win_loss_ratio': self.avg_win_loss_ratio(),
            'trades_per_day': self.trades_per_day(),
            
            # By Regime
            'performance_by_regime': self.analyze_by_regime(),
            'best_performing_regime': self.best_regime(),
            
            # Risk-Adjusted
            'calmar_ratio': self.calmar_ratio(),
            'recovery_factor': self.recovery_factor()
        }
```

### 4. Parameter Optimization Framework

Create `src/optimization/optimizer.py`:

```python
class StrategyOptimizer:
    """
    Optimize strategy parameters using various methods
    """
    
    def grid_search(self, param_grid, validation_method='walk_forward'):
        # Exhaustive search over parameter combinations
        
    def bayesian_optimization(self, param_space, n_calls=100):
        # Efficient optimization using Gaussian processes
        
    def genetic_algorithm(self, param_bounds, generations=50):
        # Evolutionary optimization for complex parameter spaces
        
    def walk_forward_analysis(self, window_size, step_size):
        # Rolling window optimization to prevent overfitting
```

### 5. Advanced Features

#### A. Market Regime Analysis
```python
class RegimeAnalyzer:
    def identify_regimes(self, data):
        # Use Hidden Markov Models or clustering
        # Return regime labels and transition probabilities
        
    def optimize_per_regime(self, data, regimes):
        # Find optimal parameters for each market regime
```

#### B. Monte Carlo Simulation
```python
class MonteCarloSimulator:
    def simulate_paths(self, strategy, n_simulations=1000):
        # Generate random price paths based on historical properties
        # Test strategy robustness across scenarios
        
    def calculate_confidence_intervals(self, results):
        # Return expected performance ranges
```

#### C. Multi-Strategy Portfolio
```python
class PortfolioManager:
    def combine_strategies(self, strategies, allocation_method='equal'):
        # Run multiple strategies with capital allocation
        
    def optimize_allocation(self, strategies, risk_target):
        # Find optimal strategy weights for risk/return target
```

### 6. Implementation Roadmap

#### Phase 1: Foundation (Week 1-2)
1. Create backtesting engine with event-driven architecture
2. Implement centralized configuration system
3. Add comprehensive metrics calculation

#### Phase 2: Optimization (Week 3-4)
1. Build parameter optimization framework
2. Implement walk-forward analysis
3. Add regime-specific optimization

#### Phase 3: Advanced Features (Week 5-6)
1. Monte Carlo simulation
2. Multi-strategy portfolio management
3. Real-time performance monitoring

#### Phase 4: Integration (Week 7-8)
1. Web dashboard for backtest results
2. Automated parameter updates based on performance
3. A/B testing framework for live trading

### 7. Testing Strategy

1. **Unit Tests**: Each component thoroughly tested
2. **Integration Tests**: Full backtest runs with known results
3. **Performance Tests**: Ensure backtesting is fast enough
4. **Validation Tests**: Compare backtest vs live results

### 8. Configuration Examples

#### Running a Simple Backtest
```bash
python backtest.py --strategy adaptive_multi --config config/strategies/adaptive_multi.yaml --start 2024-01-01 --end 2024-12-31
```

#### Parameter Optimization
```bash
python optimize.py --strategy adaptive_multi --method grid --metric sharpe --validation walk_forward
```

#### Regime Analysis
```bash
python analyze_regimes.py --data btcusd.log --method hmm --states 3
```

## Benefits of This Approach

1. **Separation of Concerns**: Backtesting logic separate from live trading
2. **Reproducibility**: All parameters in version-controlled configs
3. **Scientific Approach**: Proper validation and optimization methods
4. **Risk Management**: Comprehensive metrics and analysis
5. **Scalability**: Easy to add new strategies and features
6. **Transparency**: Clear audit trail of all decisions

## Next Steps

1. Review and approve this plan
2. Create directory structure and base classes
3. Implement Phase 1 features
4. Migrate existing strategies to new framework
5. Begin systematic parameter optimization
6. Deploy improved strategies to production

This enhanced system will provide the control and visibility needed to develop consistently profitable trading strategies while minimizing risk and avoiding overfitting.


  High Priority Deep Analysis:
  - How to ensure backtesting EXACTLY matches live trading behavior
  - Designing the tick simulation algorithm for realistic intra-bar prices
  - Architecture for configuration system that scales
  - Metric calculation that handles edge cases properly

  Medium Priority:
  - Optimization techniques for processing years of tick data
  - Visualization strategy for results
  - Testing framework design

  I recommend maximum thinking depth for:
  1. Initial architecture design (Phase 1)
  2. Tick simulation algorithm (Phase 3)
  3. Configuration system design (Phase 4)

  This is a foundational system that will drive all future strategy development,
so
  getting it right from the start is critical. The investment in deep thinking
upfront
   will save significant time and prevent costly mistakes later.

