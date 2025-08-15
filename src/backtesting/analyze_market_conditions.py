#!/usr/bin/env python3
"""
Analyze why the strategy is losing money and test alternatives
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import parse_log_file

def calculate_ma(data, window):
    """Calculate simple moving average"""
    return data.rolling(window=window).mean()

def analyze_market_conditions(df):
    """Analyze market conditions over the period"""
    df['MA4'] = calculate_ma(df['close'], 4)
    df['MA20'] = calculate_ma(df['close'], 20)
    
    # Calculate market statistics
    start_price = df.iloc[20]['close']  # After MA20 available
    end_price = df.iloc[-1]['close']
    buy_hold_return = ((end_price - start_price) / start_price) * 100
    
    # Count crossovers
    df['signal'] = (df['MA4'] > df['MA20']).astype(int)
    df['crossover'] = df['signal'].diff()
    bullish_crosses = (df['crossover'] == 1).sum()
    bearish_crosses = (df['crossover'] == -1).sum()
    
    # Calculate whipsaw frequency
    df['proximity'] = abs(df['MA4'] - df['MA20']) / df['MA20']
    avg_proximity = df['proximity'].mean() * 100
    low_proximity_pct = (df['proximity'] < 0.003).sum() / len(df) * 100
    
    # Volatility
    df['returns'] = df['close'].pct_change()
    volatility = df['returns'].std() * np.sqrt(len(df)) * 100  # Annualized
    
    # Trend analysis
    df['trend'] = calculate_ma(df['close'], 100)
    if len(df) > 200:
        trend_direction = "UP" if df.iloc[-1]['trend'] > df.iloc[-100]['trend'] else "DOWN"
    else:
        trend_direction = "UNKNOWN"
    
    return {
        'start_price': start_price,
        'end_price': end_price,
        'buy_hold_return': buy_hold_return,
        'bullish_crosses': bullish_crosses,
        'bearish_crosses': bearish_crosses,
        'total_crosses': bullish_crosses + bearish_crosses,
        'avg_proximity': avg_proximity,
        'low_proximity_pct': low_proximity_pct,
        'volatility': volatility,
        'trend': trend_direction
    }

def test_alternative_strategies(df):
    """Test alternative strategies"""
    results = []
    
    # 1. Buy and Hold
    start_price = df.iloc[0]['close']
    end_price = df.iloc[-1]['close']
    buy_hold_pnl = ((end_price - start_price) / start_price) * 10000
    results.append({
        'strategy': 'Buy and Hold',
        'pnl': buy_hold_pnl,
        'pnl_pct': (buy_hold_pnl / 10000) * 100
    })
    
    # 2. Inverse strategy (do opposite of signals)
    df['MA4'] = calculate_ma(df['close'], 4)
    df['MA20'] = calculate_ma(df['close'], 20)
    
    balance_usd = 10000
    balance_btc = 0
    position = -1
    
    for i in range(20, len(df)):
        ma4 = df.iloc[i]['MA4']
        ma20 = df.iloc[i]['MA20']
        current_price = df.iloc[i]['close']
        proximity = abs(ma4 - ma20) / ma20 if ma20 > 0 else 0
        
        # INVERSE: Sell when MA4 > MA20, Buy when MA4 < MA20
        if ma4 < ma20 and position == -1 and proximity >= 0.003:  # Inverse buy
            fee = balance_usd * 0.0025
            balance_btc = (balance_usd - fee) / current_price
            balance_usd = 0
            position = 1
        elif ma4 > ma20 and position == 1 and proximity >= 0.003:  # Inverse sell
            gross_usd = balance_btc * current_price
            fee = gross_usd * 0.0025
            balance_usd = gross_usd - fee
            balance_btc = 0
            position = -1
    
    final_value = balance_btc * df.iloc[-1]['close'] if position == 1 else balance_usd
    inverse_pnl = final_value - 10000
    results.append({
        'strategy': 'Inverse MA Cross (Contrarian)',
        'pnl': inverse_pnl,
        'pnl_pct': (inverse_pnl / 10000) * 100
    })
    
    # 3. Wider MAs (MA10/MA50)
    df['MA10'] = calculate_ma(df['close'], 10)
    df['MA50'] = calculate_ma(df['close'], 50)
    
    balance_usd = 10000
    balance_btc = 0
    position = -1
    
    for i in range(50, len(df)):
        ma10 = df.iloc[i]['MA10']
        ma50 = df.iloc[i]['MA50']
        current_price = df.iloc[i]['close']
        proximity = abs(ma10 - ma50) / ma50 if ma50 > 0 else 0
        
        if ma10 > ma50 and position == -1 and proximity >= 0.003:
            fee = balance_usd * 0.0025
            balance_btc = (balance_usd - fee) / current_price
            balance_usd = 0
            position = 1
        elif ma10 < ma50 and position == 1 and proximity >= 0.003:
            gross_usd = balance_btc * current_price
            fee = gross_usd * 0.0025
            balance_usd = gross_usd - fee
            balance_btc = 0
            position = -1
    
    final_value = balance_btc * df.iloc[-1]['close'] if position == 1 else balance_usd
    wider_pnl = final_value - 10000
    results.append({
        'strategy': 'Wider MAs (MA10/MA50)',
        'pnl': wider_pnl,
        'pnl_pct': (wider_pnl / 10000) * 100
    })
    
    return results

def main():
    print("=" * 80)
    print("MARKET ANALYSIS - Why is the strategy losing money?")
    print("=" * 80)
    
    # Load 30 days of hourly data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"\nAnalyzing period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    df = parse_log_file('btcusd.log', start_date=start_date, end_date=end_date)
    df.index = pd.to_datetime(df.index, unit='s')
    if 'price' in df.columns:
        df['close'] = df['price']
    
    # Create hourly bars
    df_hourly = df.resample('1h').agg({
        'close': 'last',
        'amount': 'sum'
    }).dropna()
    
    print(f"Analyzing {len(df_hourly)} hourly candles")
    
    # Analyze market conditions
    market = analyze_market_conditions(df_hourly)
    
    print("\n" + "=" * 80)
    print("MARKET CONDITIONS")
    print("=" * 80)
    
    print(f"\n📊 Price Movement:")
    print(f"   Start: ${market['start_price']:,.0f}")
    print(f"   End:   ${market['end_price']:,.0f}")
    print(f"   Buy & Hold Return: {market['buy_hold_return']:.2f}%")
    
    print(f"\n🔄 MA Crossover Activity:")
    print(f"   Bullish crosses: {market['bullish_crosses']}")
    print(f"   Bearish crosses: {market['bearish_crosses']}")
    print(f"   Total crosses: {market['total_crosses']}")
    print(f"   Crosses per day: {market['total_crosses'] / 30:.1f}")
    
    print(f"\n📏 MA Proximity:")
    print(f"   Average proximity: {market['avg_proximity']:.2f}%")
    print(f"   Time with MAs < 0.3% apart: {market['low_proximity_pct']:.1f}%")
    
    print(f"\n📈 Market Character:")
    print(f"   Volatility: {market['volatility']:.1f}%")
    print(f"   Trend: {market['trend']}")
    
    # Diagnose problems
    print("\n" + "=" * 80)
    print("DIAGNOSIS - Why MA4/MA20 is losing money:")
    print("=" * 80)
    
    problems = []
    
    if market['total_crosses'] > 60:
        problems.append(f"❌ Too many crossovers ({market['total_crosses']} in 30 days) = whipsawing")
    
    if market['low_proximity_pct'] > 50:
        problems.append(f"❌ MAs too close together {market['low_proximity_pct']:.0f}% of the time")
    
    if abs(market['buy_hold_return']) < 5:
        problems.append(f"❌ Choppy/sideways market ({market['buy_hold_return']:.1f}% movement)")
    
    if market['volatility'] > 100:
        problems.append(f"❌ High volatility ({market['volatility']:.0f}%) causes false signals")
    
    if not problems:
        problems.append("⚠️  Strategy parameters may not fit current market")
    
    for problem in problems:
        print(f"   {problem}")
    
    # Test alternatives
    print("\n" + "=" * 80)
    print("ALTERNATIVE STRATEGIES TEST")
    print("=" * 80)
    
    alternatives = test_alternative_strategies(df_hourly)
    
    print(f"\n{'Strategy':<30} {'P&L':<15} {'Return':<10}")
    print("-" * 60)
    
    # Add current strategy for comparison
    print(f"{'Current (MA4/MA20 + 0.3%)':<30} ${-483:>12,.2f}  {-4.83:>8.2f}%")
    
    for alt in sorted(alternatives, key=lambda x: x['pnl'], reverse=True):
        print(f"{alt['strategy']:<30} ${alt['pnl']:>12,.2f}  {alt['pnl_pct']:>8.2f}%")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n1. The MA4/MA20 strategy is too fast for current market conditions")
    print("2. Consider these options:")
    print("   a) Use wider MAs (e.g., MA10/MA50) for fewer, stronger signals")
    print("   b) Increase proximity threshold to 0.5% or higher")
    print("   c) Add additional filters (volume, momentum, etc.)")
    print("   d) Wait for trending market conditions")
    print("   e) Consider mean reversion instead of trend following")
    
    if market['buy_hold_return'] > 0:
        print(f"\n⚠️  Note: Simply holding BTC would have made {market['buy_hold_return']:.2f}%")

if __name__ == '__main__':
    main()