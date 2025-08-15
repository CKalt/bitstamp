#!/usr/bin/env python3
"""
Test Adaptive RSI strategy on recent data
Compare with MA crossover strategy
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import parse_log_file

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def backtest_adaptive_rsi(df, rsi_period=14, base_oversold=30, base_overbought=70, 
                          adaptive=True, initial_usd=10000):
    """
    Backtest Adaptive RSI strategy
    """
    # Calculate RSI
    df['RSI'] = calculate_rsi(df['close'], rsi_period)
    
    # Calculate volatility for adaptive thresholds
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Initialize
    balance_usd = initial_usd
    balance_btc = 0
    position = -1  # Start SHORT
    trades = []
    entry_price = 0
    
    for i in range(rsi_period + 20, len(df)):  # Need data for RSI and volatility
        current_price = df.iloc[i]['close']
        rsi = df.iloc[i]['RSI']
        volatility = df.iloc[i]['volatility']
        timestamp = df.index[i]
        
        if pd.isna(rsi):
            continue
        
        # Adapt thresholds based on volatility
        if adaptive and not pd.isna(volatility):
            if volatility > 0.03:  # High volatility
                adjustment = min(10, volatility * 200)
                oversold = max(20, base_oversold - adjustment)
                overbought = min(80, base_overbought + adjustment)
            elif volatility < 0.01:  # Low volatility
                oversold = min(40, base_oversold + 5)
                overbought = max(60, base_overbought - 5)
            else:  # Normal volatility
                oversold = base_oversold
                overbought = base_overbought
        else:
            oversold = base_oversold
            overbought = base_overbought
        
        # Trading logic
        if position == -1 and rsi < oversold:  # Oversold - BUY
            fee = balance_usd * 0.0025
            balance_btc = (balance_usd - fee) / current_price
            balance_usd = 0
            position = 1
            entry_price = current_price
            trades.append({
                'time': timestamp,
                'type': 'BUY',
                'price': current_price,
                'rsi': rsi,
                'threshold': oversold
            })
        elif position == 1 and rsi > overbought:  # Overbought - SELL
            gross_usd = balance_btc * current_price
            fee = gross_usd * 0.0025
            balance_usd = gross_usd - fee
            balance_btc = 0
            position = -1
            
            pnl = (current_price - entry_price) / entry_price * 100
            trades.append({
                'time': timestamp,
                'type': 'SELL',
                'price': current_price,
                'rsi': rsi,
                'threshold': overbought,
                'pnl_pct': pnl
            })
    
    # Calculate final value
    final_price = df.iloc[-1]['close']
    if position == 1:
        final_value = balance_btc * final_price
    else:
        final_value = balance_usd
    
    pnl = final_value - initial_usd
    pnl_percent = (pnl / initial_usd) * 100
    
    return {
        'strategy': 'Adaptive RSI' if adaptive else 'Fixed RSI',
        'pnl': pnl,
        'pnl_percent': pnl_percent,
        'num_trades': len(trades),
        'final_position': 'LONG' if position == 1 else 'SHORT',
        'trades': trades
    }

def backtest_ma_crossover(df, short_window=4, long_window=20, proximity_threshold=0.003, initial_usd=10000):
    """
    Backtest MA crossover for comparison
    """
    # Calculate MAs
    df['MA_short'] = df['close'].rolling(short_window).mean()
    df['MA_long'] = df['close'].rolling(long_window).mean()
    
    # Initialize
    balance_usd = initial_usd
    balance_btc = 0
    position = -1
    trades = []
    signals_blocked = 0
    
    for i in range(long_window, len(df)):
        current_price = df.iloc[i]['close']
        ma_short = df.iloc[i]['MA_short']
        ma_long = df.iloc[i]['MA_long']
        timestamp = df.index[i]
        
        # Calculate proximity
        proximity = abs(ma_short - ma_long) / ma_long if ma_long > 0 else 0
        
        # Determine signal
        signal = 1 if ma_short > ma_long else -1
        
        # Trading logic
        if signal == 1 and position == -1:  # Buy signal
            if proximity >= proximity_threshold:
                fee = balance_usd * 0.0025
                balance_btc = (balance_usd - fee) / current_price
                balance_usd = 0
                position = 1
                trades.append({'time': timestamp, 'type': 'BUY', 'price': current_price})
            else:
                signals_blocked += 1
        elif signal == -1 and position == 1:  # Sell signal
            if proximity >= proximity_threshold:
                gross_usd = balance_btc * current_price
                fee = gross_usd * 0.0025
                balance_usd = gross_usd - fee
                balance_btc = 0
                position = -1
                trades.append({'time': timestamp, 'type': 'SELL', 'price': current_price})
            else:
                signals_blocked += 1
    
    # Calculate final value
    final_price = df.iloc[-1]['close']
    if position == 1:
        final_value = balance_btc * final_price
    else:
        final_value = balance_usd
    
    pnl = final_value - initial_usd
    pnl_percent = (pnl / initial_usd) * 100
    
    return {
        'strategy': f'MA{short_window}/{long_window} + {proximity_threshold*100:.1f}%',
        'pnl': pnl,
        'pnl_percent': pnl_percent,
        'num_trades': len(trades),
        'signals_blocked': signals_blocked,
        'final_position': 'LONG' if position == 1 else 'SHORT'
    }

def main():
    print("=" * 80)
    print("ADAPTIVE RSI vs MA CROSSOVER - Strategy Comparison")
    print("=" * 80)
    
    # Load 30 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"\nTesting period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    df = parse_log_file('btcusd.log', start_date=start_date, end_date=end_date)
    df.index = pd.to_datetime(df.index, unit='s')
    if 'price' in df.columns:
        df['close'] = df['price']
    
    # Test on both hourly and 1-minute bars
    print("\nPreparing test data...")
    
    # Hourly bars
    df_hourly = df.resample('1h').agg({
        'close': 'last',
        'amount': 'sum'
    }).dropna()
    print(f"  • Hourly bars: {len(df_hourly)} candles")
    
    # 1-minute bars
    df_1min = df.resample('1min').agg({
        'close': 'last',
        'amount': 'sum'
    }).dropna()
    print(f"  • 1-minute bars: {len(df_1min)} candles")
    
    # Run tests
    results = []
    
    print("\nRunning backtests...")
    print("-" * 40)
    
    # Test RSI strategies on hourly
    print("Testing Adaptive RSI (hourly)...")
    rsi_hourly = backtest_adaptive_rsi(df_hourly.copy(), adaptive=True)
    results.append({**rsi_hourly, 'timeframe': 'Hourly'})
    
    print("Testing Fixed RSI (hourly)...")
    rsi_fixed_hourly = backtest_adaptive_rsi(df_hourly.copy(), adaptive=False)
    results.append({**rsi_fixed_hourly, 'timeframe': 'Hourly'})
    
    # Test RSI on 1-minute
    print("Testing Adaptive RSI (1-min)...")
    rsi_1min = backtest_adaptive_rsi(df_1min.copy(), adaptive=True)
    results.append({**rsi_1min, 'timeframe': '1-min'})
    
    # Test MA crossover for comparison
    print("Testing MA4/MA20 (hourly)...")
    ma_hourly = backtest_ma_crossover(df_hourly.copy())
    results.append({**ma_hourly, 'timeframe': 'Hourly'})
    
    print("Testing MA4/MA20 (1-min)...")
    ma_1min = backtest_ma_crossover(df_1min.copy())
    results.append({**ma_1min, 'timeframe': '1-min'})
    
    # Display results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(f"\n{'Strategy':<25} {'Timeframe':<10} {'P&L':<12} {'Return':<10} {'Trades':<8}")
    print("-" * 75)
    
    for r in sorted(results, key=lambda x: x['pnl'], reverse=True):
        status = "✅" if r['pnl'] > 0 else "❌"
        print(f"{r['strategy']:<25} {r['timeframe']:<10} "
              f"{status} ${r['pnl']:>9,.2f}  {r['pnl_percent']:>7.2f}%  "
              f"{r['num_trades']:>6}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    best = max(results, key=lambda x: x['pnl'])
    worst = min(results, key=lambda x: x['pnl'])
    
    print(f"\n🏆 Best Strategy: {best['strategy']} ({best['timeframe']})")
    print(f"   P&L: ${best['pnl']:,.2f} ({best['pnl_percent']:.2f}%)")
    print(f"   Trades: {best['num_trades']}")
    
    print(f"\n❌ Worst Strategy: {worst['strategy']} ({worst['timeframe']})")
    print(f"   P&L: ${worst['pnl']:,.2f} ({worst['pnl_percent']:.2f}%)")
    
    # Compare RSI vs MA
    rsi_avg = np.mean([r['pnl'] for r in results if 'RSI' in r['strategy']])
    ma_avg = np.mean([r['pnl'] for r in results if 'MA' in r['strategy']])
    
    print(f"\n📊 Strategy Type Comparison:")
    print(f"   RSI strategies avg: ${rsi_avg:,.2f}")
    print(f"   MA strategies avg: ${ma_avg:,.2f}")
    
    if rsi_avg > ma_avg:
        improvement = rsi_avg - ma_avg
        print(f"\n✅ RSI strategies outperform MA by ${improvement:,.2f} on average")
    else:
        print(f"\n❌ MA strategies still perform better in this period")
    
    # Show some RSI trades if profitable
    if rsi_1min['pnl'] > 0 and rsi_1min['trades']:
        print(f"\n📈 Sample Adaptive RSI trades (1-min):")
        for trade in rsi_1min['trades'][:5]:
            if trade['type'] == 'BUY':
                print(f"   BUY @ ${trade['price']:,.0f} (RSI: {trade['rsi']:.1f} < {trade['threshold']:.0f})")
            else:
                print(f"   SELL @ ${trade['price']:,.0f} (RSI: {trade['rsi']:.1f} > {trade['threshold']:.0f}) "
                      f"P&L: {trade.get('pnl_pct', 0):.2f}%")

if __name__ == '__main__':
    main()