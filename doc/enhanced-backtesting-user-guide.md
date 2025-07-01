# Enhanced Backtesting System User Guide

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Running Backtests](#running-backtests)
4. [Parameter Configuration](#parameter-configuration)
5. [Understanding Results](#understanding-results)
6. [Deployment Process](#deployment-process)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

## Overview

The Enhanced Backtesting System tests trading strategies using historical Bitcoin data, with special support for the AdaptiveMultiStrategy that automatically adjusts to different market conditions (trending, ranging, volatile).

### Key Features
- **Shared Code Architecture**: Backtesting uses the exact same strategy code as live trading
- **Market Regime Detection**: Tests how strategies perform in different market conditions
- **Parameter Optimization**: Automatically finds optimal strategy parameters
- **Interactive Configuration**: User-friendly prompts for easy setup
- **Multiple Presets**: From quick tests to overnight deep analysis

## Quick Start

### 1. Quick Test (2 minutes)
```bash
./run_enhanced_test.sh
```

### 2. Interactive Mode (recommended for beginners)
```bash
./run_enhanced_interactive.sh
```

### 3. Standard Run (30-60 minutes)
```bash
./run_enhanced.sh
```

## Running Backtests

### Available Scripts

| Script | Purpose | Duration | Use When |
|--------|---------|----------|----------|
| `run_enhanced_test.sh` | Quick validation | 1-2 min | Testing script changes |
| `run_enhanced_interactive.sh` | Guided setup | Varies | First time users |
| `run_enhanced.sh` | Full optimization | 2-4 hours | Production optimization |

### Command Line Options

```bash
python src/bktst_enhanced_shared.py [OPTIONS]
```

**Basic Options:**
- `--start-window-days-back N`: Days of historical data (default: 120)
- `--end-window-days-back N`: End date offset (default: 0 = today)
- `--trading-window-days N`: Analyze N days from start
- `--high-frequency 1H`: Higher timeframe (default: 1H)
- `--low-frequency 15T`: Lower timeframe (default: 15T)

**Optimization Options:**
- `--test-run`: Minimal 2-combination test
- `--optimization-preset [conservative|moderate|aggressive]`: Use presets
- `--user-prompts`: Interactive configuration mode
- `--skip-adaptive`: Skip AdaptiveMultiStrategy testing

**Custom Parameters:**
- `--short-windows 8 10 12`: Custom short MA windows
- `--long-windows 40 46 50`: Custom long MA windows

**Output:**
- `--output-file FILE`: Output filename (default: recommended_strategy.json)

## Parameter Configuration

### Method 1: Interactive Mode (Recommended)

Run with `--user-prompts` for guided configuration:

```bash
python src/bktst_enhanced_shared.py --user-prompts
```

You'll be prompted for:
1. **Date Range**: How much historical data to use
2. **Optimization Depth**: How many parameter combinations to test
3. **Custom Parameters** (if selected): Specific values for each parameter

### Method 2: Optimization Presets

Use `--optimization-preset` for predefined configurations:

| Preset | Combinations | Duration | Best For |
|--------|--------------|----------|----------|
| `conservative` | 64 | 5-10 min | Daily checks |
| `moderate` | 216 | 30-60 min | Weekly optimization |
| `aggressive` | 12,500 | 8-12 hours | Monthly deep analysis |

Example:
```bash
python src/bktst_enhanced_shared.py --optimization-preset moderate
```

### Method 3: Config File

Add custom ranges to `config.json`:

```json
{
  "optimization_ranges": {
    "short_window_range": [8, 10, 12],
    "long_window_range": [40, 46, 50],
    "regime_switch_threshold_range": [0.35, 0.40, 0.45],
    "signal_confirmation_bars_range": [1, 2],
    "min_trade_gap_minutes_range": [15, 30],
    "whipsaw_threshold_range": [6.0, 8.0]
  }
}
```

### Method 4: Command Line

Specify exact parameters:

```bash
python src/bktst_enhanced_shared.py \
  --optimization-preset custom \
  --short-windows 8 10 12 \
  --long-windows 40 46 50
```

## Understanding Results

### Output Files

After backtesting completes, you'll find:

1. **recommended_strategy.json**: Main output with best strategy configuration
2. **adaptive_strategy_optimization.csv**: All tested parameter combinations
3. **adaptive_strategy_detailed_results.json**: Detailed performance metrics
4. **strategy_comparison_enhanced.csv**: Comparison of all strategies

### Key Metrics

**Performance Metrics:**
- **Total Return**: Percentage profit/loss over test period
- **Sharpe Ratio**: Risk-adjusted returns (higher is better, >1 is good)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

**Regime Performance:**
- Shows how strategy performs in trending/ranging/volatile markets
- Helps identify which market conditions suit your strategy

### Reading recommended_strategy.json

```json
{
  "backtest_metadata": {
    "test_period_start": "2025-03-03",
    "test_period_end": "2025-07-01",
    "total_days": 119
  },
  "performance_metrics": {
    "total_return_pct": 15.42,      // 15.42% profit
    "sharpe_ratio": 1.23,           // Good risk-adjusted returns
    "max_drawdown_pct": 8.5,        // Maximum 8.5% loss
    "win_rate": 58.3,               // 58.3% winning trades
    "total_trades": 48              // Reasonable trading frequency
  },
  "optimal_parameters": {
    "strategy": "AdaptiveMulti",
    "short_window": 10,
    "long_window": 46,
    "regime_switch_threshold": 0.40
  }
}
```

## Deployment Process

### Step 1: Run Backtesting
```bash
./run_enhanced.sh
```

### Step 2: Validate Results
```bash
python src/validate_strategy.py
```

Check for:
- ✅ Positive returns
- ✅ Acceptable drawdown (<20%)
- ✅ Reasonable trade frequency
- ✅ Good Sharpe ratio (>0.5)

### Step 3: Review Changes
```bash
python src/strategy_migrator.py --dry-run
```

### Step 4: Deploy Strategy
```bash
python src/strategy_migrator.py
```

### Step 5: Enable Live Trading
Edit `best_strategy.json`:
```json
{
  "do_live_trades": true
}
```

### Step 6: Start Trading
```bash
python src/tdr.py
```

## Advanced Usage

### Testing Different Time Periods

**Recent Market Only (30 days):**
```bash
python src/bktst_enhanced_shared.py --start-window-days-back 30
```

**Specific Date Range:**
```bash
python src/bktst_enhanced_shared.py \
  --start-window-days-back 180 \
  --trading-window-days 90
```

### Combining Options

**Quick test with custom windows:**
```bash
python src/bktst_enhanced_shared.py \
  --test-run \
  --short-windows 10 \
  --long-windows 46
```

**Conservative optimization on recent data:**
```bash
python src/bktst_enhanced_shared.py \
  --optimization-preset conservative \
  --start-window-days-back 60
```

### Analyzing Specific Strategies

**Skip adaptive strategy testing:**
```bash
python src/bktst_enhanced_shared.py --skip-adaptive
```

## Troubleshooting

### Common Issues

**1. "No strategies met the criteria"**
- Relax constraints in `config.json`:
```json
{
  "strategy_constraints": {
    "min_trades_per_day": 0.1,
    "min_total_return": -10.0
  }
}
```

**2. Backtesting takes too long**
- Use `--test-run` for quick validation
- Use `--optimization-preset conservative` for faster results
- Reduce parameter ranges in config

**3. Import errors**
- Ensure virtual environment is activated:
```bash
source source-venv.sh
```

**4. Insufficient data**
- Check available data range:
```bash
python src/check_date_range.py
```

### Performance Tips

1. **Start with moderate optimization** before running aggressive
2. **Use recent data** (60-90 days) for faster, more relevant results
3. **Run overnight** for thorough optimization
4. **Monitor progress** - output shows "Progress: X/Y combinations"

## Best Practices

### 1. Regular Optimization Schedule
- **Daily**: Quick test (`--test-run`)
- **Weekly**: Moderate optimization
- **Monthly**: Aggressive deep analysis

### 2. Market Condition Awareness
- Review regime performance in results
- Adjust parameters based on current market
- Trust the adaptive features

### 3. Risk Management
- Never deploy without validation
- Start with `do_live_trades: false`
- Monitor initial trades closely
- Set appropriate emergency thresholds

### 4. Parameter Selection
- Avoid overfitting (too many combinations)
- Test on multiple time periods
- Prefer robust parameters over perfect ones

### 5. Documentation
- Document why you chose specific parameters
- Keep notes on market conditions
- Track performance over time

## Example Workflows

### Beginner Workflow
1. Run interactive mode: `./run_enhanced_interactive.sh`
2. Select option 3 (Moderate - 216 combinations)
3. Review results in recommended_strategy.json
4. If profitable, run validation
5. Deploy with caution

### Advanced Workflow
1. Test on multiple periods:
   ```bash
   # Last 30 days
   python src/bktst_enhanced_shared.py --start-window-days-back 30 --output-file results_30d.json
   
   # Last 90 days
   python src/bktst_enhanced_shared.py --start-window-days-back 90 --output-file results_90d.json
   ```
2. Compare results
3. Choose parameters that work well across periods
4. Run final validation
5. Deploy

### Production Workflow
1. Weekly moderate optimization
2. Compare with current performance
3. Only update if significant improvement
4. Always validate before deployment
5. Monitor for 24 hours after changes

## Getting Help

- **Issues**: Report at https://github.com/anthropics/claude-code/issues
- **Logs**: Check output for detailed error messages
- **Debug**: Add verbose output with standard Python logging

Remember: The goal is finding robust parameters that work well across different market conditions, not perfect parameters for historical data.