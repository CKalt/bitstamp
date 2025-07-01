# AdaptiveMultiStrategy Backtesting Debug Report

## Summary
The AdaptiveMultiStrategy backtester was generating zero trades across all 2,304 parameter combinations due to several critical mismatches between the backtester and strategy implementation.

## Root Causes Identified

### 1. **Column Name Mismatch**
- **Issue**: The backtester provides a DataFrame with column `'price'`, but the strategy expects `'close'`
- **Location**: All signal generation methods in `tdr_core/strategies.py` use `df['close']`
- **Impact**: Causes KeyError when strategy tries to access non-existent column

### 2. **Return Format Mismatch - detect_market_regime()**
- **Issue**: Method returns a tuple `(regime, confidence, metrics)` but backtester expects a dictionary
- **Location**: `bktst_enhanced.py` lines 138-140
- **Code**:
  ```python
  # Current (incorrect):
  regime_result = strategy.detect_market_regime(data_slice)
  new_regime = regime_result['regime']  # Error: tuple has no 'regime' key
  
  # Should be:
  new_regime, confidence, metrics = strategy.detect_market_regime(data_slice)
  ```

### 3. **Signal Return Format Mismatch**
- **Issue**: Signal methods return tuples `(signal, reason)` but backtester expects just the signal
- **Location**: `bktst_enhanced.py` lines 159-163
- **Code**:
  ```python
  # Current (incorrect):
  signal = strategy.generate_trending_signal(data_slice)
  
  # Should be:
  signal, reason = strategy.generate_trending_signal(data_slice)
  ```

### 4. **Missing OHLC Data**
- **Issue**: Data resampling only creates `price`, `amount`, `volume` columns
- **Location**: `bktst_enhanced.py` lines 391-395
- **Impact**: Strategy may need full OHLC data for indicators
- **Fix**:
  ```python
  # Current:
  df_hourly = df.resample('1H').agg({
      'price': 'last',
      'amount': 'sum',
      'volume': 'sum'
  })
  
  # Should be:
  df_hourly = df.resample('1H').agg({
      'price': ['first', 'max', 'min', 'last'],
      'amount': 'sum',
      'volume': 'sum'
  })
  df_hourly.columns = ['open', 'high', 'low', 'close', 'amount', 'volume']
  ```

### 5. **Uninitialized Regime**
- **Issue**: `current_regime` starts as `None` but comparisons use strings
- **Location**: `bktst_enhanced.py` line 111
- **Impact**: First regime detection may fail
- **Fix**: Initialize as `current_regime = 'TRENDING'`

## Solution

### Option 1: Fix the Backtester (Recommended)
Apply the fixes in `bktst_enhanced_fixed.py`:
1. Add column mapping: `data_slice['close'] = data_slice['price']`
2. Unpack tuple returns from `detect_market_regime()`
3. Unpack tuple returns from signal generation methods
4. Create proper OHLC columns in data preparation
5. Initialize `current_regime` with valid string

### Option 2: Modify the Strategy
Add adapter methods to handle column mapping internally (not recommended as it breaks the strategy design)

## Testing the Fix

1. **Run the debug script**:
   ```bash
   python debug_adaptive_strategy.py
   ```

2. **Test with fixed backtester**:
   ```bash
   python bktst_enhanced_fixed.py --skip-adaptive=False
   ```

3. **Compare results**:
   - Original: 0 trades across all parameter combinations
   - Fixed: Should generate trades based on market conditions

## Additional Debug Information

If trades are still not being generated after fixes:
1. Check if the regime detection thresholds are too strict
2. Verify that the data time range is sufficient (needs at least 60 bars)
3. Check if signal confirmation requirements are preventing trades
4. Review the `min_trade_gap_minutes` parameter (might be too restrictive)

## Files Modified
- `debug_adaptive_strategy.py` - Debug analysis script
- `bktst_enhanced_fixed.py` - Fixed version of the backtester
- `adaptive_strategy_fix.patch` - Patch file for the original

## Next Steps
1. Apply the fixes to `src/bktst_enhanced.py`
2. Re-run the backtesting with various parameter combinations
3. Monitor the trade generation and regime switching behavior
4. Fine-tune parameters based on results