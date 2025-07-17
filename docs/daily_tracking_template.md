# Daily Trading Tracker Template

## Date: [YYYY-MM-DD]

### Morning Check (5 min)
```
status long
whipsaw_stats  
show_trade_sequence 24
```

**Observations**:
- [ ] Position: LONG / SHORT at $______
- [ ] P&L since yesterday: $______
- [ ] Overnight activity: ______
- [ ] Current regime: TRENDING / RANGING / VOLATILE

### Market Conditions
- [ ] Major news/events: ______
- [ ] Volume: Normal / High / Low
- [ ] Volatility: Normal / High / Low
- [ ] Other pairs correlation: ______

### Trades Today
| Time | Type | Price | Size | Reason | Result |
|------|------|-------|------|--------|--------|
| | | | | | |

### Parameter Observations
- [ ] Pivot protection triggered? Y/N - Good/Bad?
- [ ] Whipsaws detected? Y/N - Justified?
- [ ] Regime detection accurate? Y/N
- [ ] Trade timing optimal? Y/N

### End of Day Review (10 min)
```
trades 20
performance_summary (when available)
```

**Today's Performance**:
- Trades: ___
- P&L: $____
- Best decision: ______
- Worst decision: ______
- Tomorrow's focus: ______

### Ideas for Improvement
- [ ] Parameter to adjust: ______
- [ ] Feature to build: ______
- [ ] Backtest to run: ______

### Weekly Rollup Items
- [ ] Patterns noticed: ______
- [ ] Parameter change candidates: ______
- [ ] Multi-pair opportunities: ______

---

## Quick Commands Reference

**Status Checks**:
- `status` - Quick position view
- `status long` - Full details with pivot levels
- `whipsaw_stats` - Whipsaw analysis
- `show_trade_sequence` - See trade patterns

**Performance**:
- `trades [n]` - Recent trades
- `show_diagnostics` - System events
- `position_history` - Track position changes

**Analysis** (coming soon):
- `performance_report [days]`
- `correlation_matrix`
- `regime_analysis`

**Remember**: Document the WHY, not just the WHAT. Your future self will thank you!