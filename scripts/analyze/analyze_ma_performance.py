#!/usr/bin/env python3
"""
Detailed MA strategy performance analysis with drawdowns
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data.loader import parse_log_file
from indicators.technical_indicators import ensure_datetime_index

print("📊 MA STRATEGY 30-DAY PERFORMANCE ANALYSIS")
print("=" * 60)

# Load 30 days of data
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

print(f"Analyzing period: {start_date.date()} to {end_date.date()}")
print("\nLoading data...")

df = parse_log_file('btcusd.log', start_date, end_date)
df = ensure_datetime_index(df)

# Resample to hourly
hourly = df['price'].resample('1H').ohlc()
hourly['volume'] = df['amount'].resample('1H').sum()
hourly = hourly.dropna()

print(f"Loaded {len(hourly)} hourly candles")

# Calculate MAs
hourly['MA4'] = hourly['close'].rolling(window=4).mean()
hourly['MA20'] = hourly['close'].rolling(window=20).mean()

# Generate signals
hourly['signal'] = 0
hourly.loc[hourly['MA4'] > hourly['MA20'], 'signal'] = 1
hourly.loc[hourly['MA4'] < hourly['MA20'], 'signal'] = -1

# Find trades (signal changes)
hourly['position'] = hourly['signal'].fillna(0)
hourly['trade'] = hourly['position'].diff()

# Simulate trading
initial_balance = 10000
balance = initial_balance
btc_balance = 0
position = 0
entry_price = 0
trades = []
equity_curve = []
max_drawdown = 0
current_drawdown = 0
peak_equity = initial_balance

for idx in hourly.index[20:]:  # Start after MA20 warmup
    price = hourly.loc[idx, 'close']
    signal = hourly.loc[idx, 'signal']
    
    # Calculate current equity
    if position == 1:
        current_equity = btc_balance * price
    else:
        current_equity = balance
    
    equity_curve.append({
        'time': idx,
        'equity': current_equity,
        'price': price,
        'position': position
    })
    
    # Update peak and drawdown
    if current_equity > peak_equity:
        peak_equity = current_equity
        current_drawdown = 0
    else:
        current_drawdown = (peak_equity - current_equity) / peak_equity * 100
        max_drawdown = max(max_drawdown, current_drawdown)
    
    # Check for trade
    if position != signal and signal != 0:
        fee_rate = 0.0012
        
        if signal == 1 and position != 1:  # Buy
            btc_amount = (balance * (1 - fee_rate)) / price
            btc_balance = btc_amount
            balance = 0
            entry_price = price
            position = 1
            trades.append({
                'time': idx,
                'action': 'BUY',
                'price': price,
                'equity_before': current_equity
            })
            
        elif signal == -1 and position != -1:  # Sell
            if btc_balance > 0:
                usd_amount = btc_balance * price * (1 - fee_rate)
                profit = (price - entry_price) * btc_balance
                balance = usd_amount
                btc_balance = 0
                position = -1
                trades.append({
                    'time': idx,
                    'action': 'SELL',
                    'price': price,
                    'profit': profit,
                    'equity_after': usd_amount
                })

# Final equity
if position == 1 and btc_balance > 0:
    final_equity = btc_balance * hourly['close'].iloc[-1]
else:
    final_equity = balance

total_return = ((final_equity - initial_balance) / initial_balance) * 100

# Analyze trades
winning_trades = [t for t in trades if t.get('action') == 'SELL' and t.get('profit', 0) > 0]
losing_trades = [t for t in trades if t.get('action') == 'SELL' and t.get('profit', 0) <= 0]

print("\n📈 PERFORMANCE SUMMARY:")
print("-" * 40)
print(f"Initial Balance: ${initial_balance:,.2f}")
print(f"Final Equity: ${final_equity:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Max Drawdown: {max_drawdown:.2f}%")

print(f"\n📊 TRADE STATISTICS:")
print(f"Total Trades: {len([t for t in trades if t['action'] == 'SELL'])}")
print(f"Winning Trades: {len(winning_trades)}")
print(f"Losing Trades: {len(losing_trades)}")
if trades:
    win_rate = len(winning_trades) / len([t for t in trades if t['action'] == 'SELL']) * 100
    print(f"Win Rate: {win_rate:.1f}%")

# Find worst trades
if losing_trades:
    print(f"\n🔴 WORST LOSING TRADES:")
    sorted_losses = sorted(losing_trades, key=lambda x: x.get('profit', 0))
    for i, trade in enumerate(sorted_losses[:3]):
        buy_trade = [t for t in trades if t['action'] == 'BUY' and t['time'] < trade['time']][-1]
        loss_pct = (trade['profit'] / (buy_trade['price'] * btc_balance)) * 100
        print(f"{i+1}. Buy: ${buy_trade['price']:,.0f} → Sell: ${trade['price']:,.0f}")
        print(f"   Loss: ${abs(trade['profit']):.2f} ({abs(loss_pct):.1f}%)")
        print(f"   Held for: {trade['time'] - buy_trade['time']}")

# Current position analysis
if position == 1:
    current_price = hourly['close'].iloc[-1]
    current_loss = (current_price - entry_price) / entry_price * 100
    print(f"\n⚠️  CURRENT POSITION:")
    print(f"Status: LONG (still holding)")
    print(f"Entry: ${entry_price:,.2f}")
    print(f"Current: ${current_price:,.2f}")
    print(f"Unrealized Loss: {current_loss:.2f}%")

# Equity curve analysis
equity_df = pd.DataFrame(equity_curve)
print(f"\n📉 DRAWDOWN ANALYSIS:")
print(f"Maximum Drawdown: {max_drawdown:.2f}%")
print(f"Current Drawdown: {current_drawdown:.2f}%")

# Time underwater
underwater_time = sum(1 for e in equity_curve if e['equity'] < peak_equity)
total_time = len(equity_curve)
underwater_pct = (underwater_time / total_time) * 100
print(f"Time in Drawdown: {underwater_pct:.1f}% of the time")

print("\n🎯 BOTTOM LINE:")
print("-" * 40)
if total_return > 0:
    print(f"✅ Strategy was profitable: +{total_return:.2f}%")
else:
    print(f"❌ Strategy lost money: {total_return:.2f}%")
print(f"⚠️  But max drawdown was {max_drawdown:.2f}%")
print(f"💡 Risk/Reward Ratio: {abs(max_drawdown/total_return):.2f}x drawdown per unit of return")