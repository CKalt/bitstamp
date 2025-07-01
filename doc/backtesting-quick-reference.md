# Enhanced Backtesting Quick Reference

## 🚀 Common Commands

### Quick Test (2 combinations, ~1 minute)
```bash
./run_enhanced_test.sh
```

### Interactive Mode (guided setup)
```bash
./run_enhanced_interactive.sh
```

### Standard Optimization (2304 combinations, 2-4 hours)
```bash
./run_enhanced.sh
```

### Moderate Optimization (216 combinations, 30-60 min)
```bash
python src/bktst_enhanced_shared.py --optimization-preset moderate
```

## 📊 Optimization Presets

| Command | Combinations | Time | Use Case |
|---------|--------------|------|----------|
| `--test-run` | 2 | 1 min | Quick validation |
| `--optimization-preset conservative` | 64 | 5-10 min | Daily checks |
| `--optimization-preset moderate` | 216 | 30-60 min | Weekly optimization |
| `--optimization-preset aggressive` | 12,500 | 8-12 hrs | Monthly deep dive |

## 📅 Date Range Options

```bash
# Last 30 days
--start-window-days-back 30

# Last 60 days  
--start-window-days-back 60

# Custom range (e.g., 90 days starting 180 days ago)
--start-window-days-back 180 --trading-window-days 90
```

## 🎯 Custom Parameters

```bash
# Specify exact windows to test
python src/bktst_enhanced_shared.py \
  --optimization-preset custom \
  --short-windows 8 10 12 \
  --long-windows 40 46 50
```

## 📁 Output Files

- `recommended_strategy.json` - Best strategy configuration
- `adaptive_strategy_optimization.csv` - All tested combinations
- `strategy_comparison_enhanced.csv` - Strategy comparison
- `adaptive_strategy_detailed_results.json` - Detailed metrics

## 🔄 Full Deployment Process

```bash
# 1. Run optimization
./run_enhanced.sh

# 2. Validate results
python src/validate_strategy.py

# 3. Preview changes
python src/strategy_migrator.py --dry-run

# 4. Deploy strategy
python src/strategy_migrator.py

# 5. Enable live trading (edit best_strategy.json)
# Set "do_live_trades": true

# 6. Start trading
python src/tdr.py
```

## ⚡ Performance Tips

1. **Too slow?** Use `--optimization-preset conservative`
2. **Need recent data?** Add `--start-window-days-back 30`
3. **Quick validation?** Use `--test-run`
4. **Overnight run?** Use `--optimization-preset aggressive`

## 🎮 Interactive Mode Options

1. **Quick Test** - 2 combinations (1 minute)
2. **Conservative** - 64 combinations (5-10 minutes)
3. **Moderate** - 216 combinations (30-60 minutes)
4. **Thorough** - 2304 combinations (2-4 hours)
5. **Aggressive** - 12,500 combinations (overnight)
6. **Custom** - Define your own ranges

## ❓ Help & Troubleshooting

```bash
# Check available data
python src/check_date_range.py

# Activate virtual environment
source source-venv.sh

# View this guide
cat doc/enhanced-backtesting-user-guide.md
```

## 📈 Key Metrics to Watch

- **Total Return** > 0% (profitable)
- **Sharpe Ratio** > 1.0 (good risk-adjusted returns)
- **Max Drawdown** < 20% (acceptable risk)
- **Win Rate** > 50% (more wins than losses)