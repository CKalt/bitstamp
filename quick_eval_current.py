#!/usr/bin/env python3
"""
Quick evaluation of current strategy performance
"""
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from data.loader import parse_log_file
from indicators.technical_indicators import add_moving_averages, generate_ma_signals

def evaluate_strategy(days_back):
    """Quick evaluation without full backtest infrastructure"""
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"\n📊 Evaluating MA 6/34 Strategy")
    print(f"Period: Last {days_back} days ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
    print("Loading data...")
    
    # Load data
    df = parse_log_file('btcusd.log', start_date=start_date, end_date=end_date)
    
    if df is None or len(df) == 0:
        print("❌ No data found")
        return
    
    print(f"✅ Loaded {len(df):,} data points")
    
    # Convert to hourly for MA calculations
    # Ensure datetime index
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
    
    df_hourly = df.resample('1H').agg({
        'price': ['first', 'max', 'min', 'last'],
        'amount': 'sum'
    })
    df_hourly.columns = ['open', 'high', 'low', 'close', 'volume']
    df_hourly.dropna(inplace=True)
    
    print(f"✅ Converted to {len(df_hourly)} hourly bars")
    
    # Add MA indicators manually since the function expects timestamp
    df_ma = df_hourly.copy()
    df_ma['Short_MA'] = df_ma['close'].rolling(window=6).mean()
    df_ma['Long_MA'] = df_ma['close'].rolling(window=34).mean()
    
    # Generate signals
    df_ma['MA_Signal'] = 0
    df_ma.loc[df_ma['Short_MA'] > df_ma['Long_MA'], 'MA_Signal'] = 1
    df_ma.loc[df_ma['Short_MA'] < df_ma['Long_MA'], 'MA_Signal'] = -1
    df_ma.dropna(inplace=True)
    df_ma = generate_ma_signals(df_ma)
    
    # Count signals
    signals = df_ma['MA_Signal'].value_counts()
    signal_changes = (df_ma['MA_Signal'].diff() != 0).sum()
    
    # Simulate simple trading
    position = 0  # Start neutral
    trades = []
    
    for i in range(1, len(df_ma)):
        prev_signal = df_ma.iloc[i-1]['MA_Signal']
        curr_signal = df_ma.iloc[i]['MA_Signal']
        
        if prev_signal != curr_signal and curr_signal != 0:
            # Signal changed
            price = df_ma.iloc[i]['close']
            if curr_signal == 1 and position <= 0:
                # BUY signal
                trades.append({
                    'time': df_ma.index[i],
                    'type': 'BUY',
                    'price': price,
                    'from_position': position
                })
                position = 1
            elif curr_signal == -1 and position >= 0:
                # SELL signal
                trades.append({
                    'time': df_ma.index[i],
                    'type': 'SELL',
                    'price': price,
                    'from_position': position
                })
                position = -1
    
    # Calculate simple P&L
    initial_balance = 10000
    balance = initial_balance
    btc = 0
    fee_rate = 0.0012
    
    for trade in trades:
        if trade['type'] == 'BUY':
            # Buy BTC with all USD
            usd_spent = balance
            fee = usd_spent * fee_rate
            btc = (usd_spent - fee) / trade['price']
            balance = 0
        else:  # SELL
            # Sell all BTC for USD
            usd_received = btc * trade['price']
            fee = usd_received * fee_rate
            balance = usd_received - fee
            btc = 0
    
    # Final value
    final_price = df_ma.iloc[-1]['close']
    if btc > 0:
        final_value = btc * final_price
    else:
        final_value = balance
    
    total_return = ((final_value - initial_balance) / initial_balance) * 100
    
    # Print results
    print(f"\n📈 RESULTS:")
    print(f"   Total Trades: {len(trades)}")
    print(f"   Signal Changes: {signal_changes}")
    print(f"   Final Position: {'LONG' if position > 0 else 'SHORT' if position < 0 else 'NEUTRAL'}")
    print(f"   Initial: ${initial_balance:,.2f}")
    print(f"   Final: ${final_value:,.2f}")
    print(f"   Return: {total_return:+.2f}%")
    
    if len(trades) > 0:
        print(f"\n📊 Trade Summary:")
        for i, trade in enumerate(trades[-5:], 1):  # Last 5 trades
            print(f"   {trade['time'].strftime('%Y-%m-%d %H:%M')} - {trade['type']} @ ${trade['price']:,.0f}")
    
    # Price range
    print(f"\n💹 Price Range:")
    print(f"   Start: ${df_hourly.iloc[0]['close']:,.0f}")
    print(f"   End: ${df_hourly.iloc[-1]['close']:,.0f}")
    print(f"   High: ${df_hourly['high'].max():,.0f}")
    print(f"   Low: ${df_hourly['low'].min():,.0f}")
    
    return {
        'days': days_back,
        'trades': len(trades),
        'return': total_return,
        'final_position': position
    }


def main():
    print("Current Live Strategy: MA 6/34")
    print("Original Performance: +8.77% return, 0.600 Sharpe")
    
    results = []
    for days in [30, 60]:
        result = evaluate_strategy(days)
        if result:
            results.append(result)
    
    if results:
        print(f"\n{'='*50}")
        print("COMPARISON SUMMARY")
        print(f"{'='*50}")
        print(f"{'Period':<15} {'Trades':>8} {'Return':>10}")
        print(f"{'-'*33}")
        for r in results:
            print(f"Last {r['days']} days    {r['trades']:>8} {r['return']:>9.2f}%")


if __name__ == "__main__":
    main()