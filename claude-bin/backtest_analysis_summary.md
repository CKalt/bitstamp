# 30-Day Backtest Analysis: Production vs Test Systems

## Executive Summary
Both systems show the proximity threshold is working, but **HOURLY bars (production) perform better than 1-minute bars** for this strategy over 30 days.

## Results Comparison

### 🎯 PRODUCTION SYSTEM (gg btc)
- **Configuration**: Hourly bars, 0.3% proximity threshold
- **P&L**: -$483 (-4.83%)
- **Trades**: 19 (blocked 47 bad signals - 71% block rate)
- **Max Drawdown**: 7.95%
- **Win Rate**: 33.3%

### 🧪 TEST SYSTEM (gg tst)  
- **Configuration**: 1-minute bars, 0.3% proximity threshold
- **P&L**: -$2,218 (-22.18%)
- **Trades**: 63 (blocked 11,032 bad signals - 99.4% block rate\!)
- **Max Drawdown**: 25.11%
- **Win Rate**: 12.9%

### ⚠️ BASELINE (No Threshold)
- **P&L**: -$9,729 (-97.29%\!) 
- **Trades**: 1,467 (excessive flipping)
- **Win Rate**: 5.5%

## Key Findings

### 1. Proximity Threshold Impact (0.3%)
✅ **Massive improvement over no threshold:**
- Test system: +$7,511 improvement
- Reduced trades by 96% (1,404 fewer trades)
- Successfully blocked 11,032 bad signals on 1-minute bars

### 2. Hourly vs 1-Minute Bars
📊 **Hourly bars outperform by $1,734:**
- Production (hourly): -$483 loss
- Test (1-minute): -$2,218 loss
- Production has 3x better win rate (33% vs 13%)
- Production has lower drawdown (8% vs 25%)

### 3. Why Hourly Performs Better
1. **Less noise**: Hourly bars filter out minute-to-minute volatility
2. **Stronger signals**: MA crossovers on hourly are more meaningful
3. **Lower fees**: Fewer trades = less fees (19 vs 63 trades)
4. **Better win rate**: 33% vs 13%

## Recommendations

1. **Keep the 0.3% proximity threshold** - It's preventing massive losses
2. **Production system (hourly) is the better configuration** for this MA strategy
3. **Consider testing 0.4-0.5% threshold** on hourly to potentially reduce losses further
4. **1-minute bars might work better with different strategies** (momentum, scalping)

## Bottom Line
The proximity threshold is working excellently - preventing 96-99% of bad trades. The production system with hourly bars is the superior configuration for this MA crossover strategy.
