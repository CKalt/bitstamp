# TDR Trading System Enhancement Plan

## Proposed Enhancement: Dynamic Position Sizing

### Current Limitation
The system currently operates with binary positions:
- **100% LONG**: All capital in BTC
- **100% SHORT**: All capital in USD

This "all-or-nothing" approach has drawbacks:
1. High risk during uncertain market conditions
2. No ability to scale in/out of positions
3. Missed opportunities when confidence is moderate (e.g., 60-79%)
4. Large slippage impact on full position reversals

### Proposed Position Sizing Strategy

#### 1. Confidence-Based Sizing
Map regime confidence to position size:
```
90-100% confidence → 100% position
80-89% confidence → 80% position  
70-79% confidence → 60% position
60-69% confidence → 40% position
50-59% confidence → 20% position
<50% confidence → 0% (stay in USD)
```

#### 2. Regime-Specific Allocations
Different position sizes by market regime:
- **TRENDING**: 80-100% (follow strong trends)
- **RANGING**: 40-60% (smaller bets on mean reversion)
- **VOLATILE**: 20-40% (protect capital in chaos)

#### 3. Gradual Position Transitions
Instead of instant reversals:
- Scale out of losing positions in steps
- Scale into new positions gradually
- Reduce slippage and whipsaw damage

### Implementation Approach

#### Phase 1: Backtesting Infrastructure
1. Modify backtester to support partial positions
2. Test various position sizing algorithms:
   - Linear scaling (confidence % = position %)
   - Exponential scaling (accelerate near high confidence)
   - Kelly Criterion (optimal sizing based on win rate)
   - Risk parity (size based on volatility)

#### Phase 2: Strategy Updates
1. Update `AdaptiveMultiStrategy` to output position size
2. Modify order execution to handle partial amounts
3. Add position size to signal generation

#### Phase 3: Risk Management
1. Maximum position limits by volatility
2. Drawdown-based position reduction
3. Time-based position decay (reduce old positions)

### Backtesting Metrics to Track

1. **Risk-Adjusted Returns**
   - Sharpe Ratio improvement
   - Maximum drawdown reduction
   - Volatility of returns

2. **Trading Efficiency**
   - Reduced slippage costs
   - Lower turnover
   - Better entry/exit prices

3. **Confidence Calibration**
   - Accuracy of confidence scores
   - Optimal confidence thresholds
   - Position size vs outcome correlation

### Example Scenarios

#### Current System (Binary)
- BTC at $100k, confident LONG → Buy 1.5 BTC
- Price drops to $95k, signal flips → Sell ALL 1.5 BTC
- Loss: $7,500 + slippage on full position

#### Proposed System (Graduated)
- BTC at $100k, 80% confident LONG → Buy 1.2 BTC (80%)
- Confidence drops to 60% → Reduce to 0.9 BTC (60%)
- Price drops, confidence at 40% → Reduce to 0.6 BTC (40%)
- Signal flips at 70% confidence SHORT → Move to 40% USD, 60% BTC
- Gradual transition reduces losses and slippage

### Benefits

1. **Risk Reduction**
   - Lower drawdowns during uncertainty
   - Protected capital in volatile markets
   - Smoother equity curve

2. **Improved Returns**
   - Capture partial moves with lower risk
   - Compound gains more consistently
   - Reduce whipsaw losses

3. **Psychological Benefits**
   - Less stressful than all-or-nothing
   - More nuanced market expression
   - Better sleep at night

### Potential Challenges

1. **Complexity**
   - More parameters to optimize
   - Harder to debug issues
   - More ways to go wrong

2. **Tax Implications**
   - More frequent trades
   - Partial position tax lots
   - Wash sale considerations

3. **Exchange Fees**
   - More transactions
   - Need to model fee impact
   - Minimum trade sizes

### Next Steps

1. **Research Phase**
   - Review academic literature on position sizing
   - Study successful fund approaches
   - Analyze historical BTC volatility regimes

2. **Prototype Phase**
   - Create position sizing module
   - Backtest on historical data
   - Compare with current binary approach

3. **Decision Phase**
   - If backtests show >20% Sharpe improvement → Proceed
   - If drawdowns reduce >30% → Strong proceed
   - If complexity outweighs benefits → Keep binary

### Alternative Simpler Approach

If full dynamic sizing is too complex, consider a simple three-state system:
- **FULL LONG**: 100% BTC (high confidence bull)
- **NEUTRAL**: 50% BTC, 50% USD (uncertain)
- **FULL SHORT**: 100% USD (high confidence bear)

This provides some flexibility while keeping it manageable.

## Summary

Dynamic position sizing could significantly improve risk-adjusted returns by:
- Reducing drawdowns during uncertain periods
- Capturing gains with appropriate risk
- Smoothing the equity curve

The key is finding the right balance between sophistication and simplicity. Start with backtesting to quantify potential benefits before committing to implementation.