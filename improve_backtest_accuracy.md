# Improving Backtest Accuracy

## 1. Model Realistic Fills
```python
# Instead of:
fill_price = signal_price

# Use:
if trade_type == 'market_sell':
    fill_price = signal_price * 0.9995  # 0.05% slippage
else:  # market_buy
    fill_price = signal_price * 1.0005  # 0.05% slippage
```

## 2. Account for Multi-Part Orders
```python
# For positions > 0.9 BTC:
num_parts = math.ceil(position_size / 0.9)
avg_slippage = 0.0005 * num_parts  # More parts = more slippage
```

## 3. Add Whipsaw Filter
```python
# Prevent rapid reversals
min_time_between_flips = timedelta(hours=2)
if last_trade_time + min_time_between_flips > current_time:
    skip_signal = True
```

## 4. Realistic Fee Calculation
```python
# Bitstamp fee tiers
if monthly_volume < 20000:
    fee_rate = 0.0025  # 0.25%
elif monthly_volume < 100000:
    fee_rate = 0.0024  # 0.24%
# etc...

# Multi-part orders = multiple fees
total_fee = fee_rate * trade_value * num_parts
```

## 5. Volatility-Adjusted Pivots
```python
# Wider pivots in choppy markets
atr = calculate_atr(14)  # Average True Range
pivot_buffer = max(0.002, atr / price)  # Min 0.2%, scale with volatility
```

## 6. Time-of-Day Analysis
- Check if certain hours are more profitable
- Your trades today clustered around 16:00-20:00
- Consider time-based filters

## 7. Adaptive Trade Sizing
```python
# Reduce size after losses
if consecutive_losses > 2:
    trade_size = base_size * 0.5
```