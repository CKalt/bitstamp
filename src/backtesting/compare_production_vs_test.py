#!/usr/bin/env python3
"""
Compare production (gg btc) vs test (gg tst) configurations
Production: Hourly bars, 0.3% proximity threshold, live trading
Test: 1-minute bars, 0.3% proximity threshold, paper trading
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import parse_log_file

def calculate_ma(data, window):
    """Calculate simple moving average"""
    return data.rolling(window=window).mean()

def run_backtest(df, proximity_threshold=0.3, initial_usd=10000, description=""):
    """
    Run backtest with specific configuration
    Uses MA4 and MA20 crossover strategy
    """
    # Calculate moving averages
    df['MA4'] = calculate_ma(df['close'], 4)
    df['MA20'] = calculate_ma(df['close'], 20)
    
    # Initialize
    balance_usd = initial_usd
    balance_btc = 0
    position = -1  # Start SHORT (holding USD)
    trades = []
    signals_blocked = 0
    signals_total = 0
    max_drawdown = 0
    peak_value = initial_usd
    
    # Track daily values for analysis
    daily_values = []
    current_day = None
    
    # Skip first 20 rows (need MA20)
    for i in range(20, len(df)):
        current_price = df.iloc[i]['close']
        ma4 = df.iloc[i]['MA4']
        ma20 = df.iloc[i]['MA20']
        timestamp = df.index[i]
        
        # Track daily values
        day = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        if day != current_day:
            current_day = day
            if position == 1:
                current_value = balance_btc * current_price
            else:
                current_value = balance_usd
            daily_values.append({
                'date': day,
                'value': current_value,
                'price': current_price
            })
            # Update max drawdown
            if current_value > peak_value:
                peak_value = current_value
            drawdown = (peak_value - current_value) / peak_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate proximity
        if ma20 > 0:
            proximity = abs(ma4 - ma20) / ma20
        else:
            proximity = 0
        
        # Determine signal (1 = BUY signal, -1 = SELL signal)
        if ma4 > ma20:
            signal = 1  # MA4 above MA20 - bullish
        else:
            signal = -1  # MA4 below MA20 - bearish
            
        # Check if we should trade
        should_trade = False
        trade_type = None
        
        if signal == 1 and position == -1:  # Buy signal and we're SHORT
            signals_total += 1
            if proximity >= proximity_threshold:
                should_trade = True
                trade_type = 'BUY'
            else:
                signals_blocked += 1
                
        elif signal == -1 and position == 1:  # Sell signal and we're LONG  
            signals_total += 1
            if proximity >= proximity_threshold:
                should_trade = True
                trade_type = 'SELL'
            else:
                signals_blocked += 1
        
        # Execute trade if conditions met
        if should_trade:
            if trade_type == 'BUY':
                # Buy BTC with USD
                fee = balance_usd * 0.0025  # 0.25% fee
                balance_btc = (balance_usd - fee) / current_price
                balance_usd = 0
                position = 1
                trades.append({
                    'time': timestamp,
                    'type': 'BUY',
                    'price': current_price,
                    'btc_amount': balance_btc,
                    'proximity': proximity * 100  # Convert to percentage
                })
            elif trade_type == 'SELL':
                # Sell BTC for USD
                gross_usd = balance_btc * current_price
                fee = gross_usd * 0.0025  # 0.25% fee
                balance_usd = gross_usd - fee
                balance_btc = 0
                position = -1
                trades.append({
                    'time': timestamp,
                    'type': 'SELL',
                    'price': current_price,
                    'usd_amount': balance_usd,
                    'proximity': proximity * 100
                })
    
    # Calculate final value
    final_price = df.iloc[-1]['close']
    if position == 1:
        final_value = balance_btc * final_price
    else:
        final_value = balance_usd
    
    pnl = final_value - initial_usd
    pnl_percent = (pnl / initial_usd) * 100
    
    # Calculate trade statistics
    winning_trades = 0
    losing_trades = 0
    total_profit = 0
    total_loss = 0
    
    for i in range(0, len(trades) - 1, 2):
        if i + 1 < len(trades):
            buy_trade = trades[i]
            sell_trade = trades[i + 1]
            if sell_trade['type'] == 'SELL':
                profit = sell_trade['usd_amount'] - (initial_usd if i == 0 else trades[i-1]['usd_amount'] if i > 0 else initial_usd)
                if profit > 0:
                    winning_trades += 1
                    total_profit += profit
                else:
                    losing_trades += 1
                    total_loss += abs(profit)
    
    win_rate = (winning_trades / max(1, winning_trades + losing_trades)) * 100
    avg_win = total_profit / max(1, winning_trades)
    avg_loss = total_loss / max(1, losing_trades)
    profit_factor = total_profit / max(1, total_loss)
    
    return {
        'description': description,
        'proximity_threshold': proximity_threshold,
        'final_value': final_value,
        'pnl': pnl,
        'pnl_percent': pnl_percent,
        'num_trades': len(trades),
        'signals_blocked': signals_blocked,
        'signals_total': signals_total,
        'block_rate': (signals_blocked / max(1, signals_total)) * 100,
        'final_position': 'LONG' if position == 1 else 'SHORT',
        'trades': trades,
        'max_drawdown': max_drawdown * 100,
        'win_rate': win_rate,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'daily_values': daily_values
    }

def main():
    print("=" * 80)
    print("PRODUCTION vs TEST SYSTEM COMPARISON - 30 Day Backtest")
    print("=" * 80)
    
    # Load 30 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"\nLoading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    df = parse_log_file('btcusd.log', start_date=start_date, end_date=end_date)
    
    if df.empty:
        print("Error: No data loaded")
        return
    
    # Convert to proper format
    df.index = pd.to_datetime(df.index, unit='s')
    if 'price' in df.columns:
        df['close'] = df['price']
    
    print(f"Loaded {len(df)} trades")
    
    # Test both configurations
    print("\nPreparing test data...")
    
    # 1. PRODUCTION CONFIG: Hourly bars
    print("  • Creating hourly bars for production config...")
    df_hourly = df.resample('1h').agg({
        'close': 'last',
        'amount': 'sum'
    }).dropna()
    print(f"    → {len(df_hourly)} hourly candles")
    
    # 2. TEST CONFIG: 1-minute bars  
    print("  • Creating 1-minute bars for test config...")
    df_1min = df.resample('1min').agg({
        'close': 'last',
        'amount': 'sum'
    }).dropna()
    print(f"    → {len(df_1min)} minute candles")
    
    print("\nRunning backtests...")
    print("-" * 80)
    
    # Run production backtest (hourly bars, 0.3% threshold)
    print("Testing PRODUCTION config (hourly bars, 0.3% threshold)...")
    prod_result = run_backtest(
        df_hourly, 
        proximity_threshold=0.003,
        description="PRODUCTION (gg btc): Hourly bars"
    )
    
    # Run test backtest (1-min bars, 0.3% threshold)
    print("Testing TEST config (1-min bars, 0.3% threshold)...")
    test_result = run_backtest(
        df_1min,
        proximity_threshold=0.003,
        description="TEST (gg tst): 1-minute bars"
    )
    
    # Also test without threshold for comparison
    print("Testing baseline (1-min bars, NO threshold)...")
    baseline_result = run_backtest(
        df_1min,
        proximity_threshold=0.0,
        description="BASELINE: No threshold"
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 80)
    
    results = [prod_result, test_result, baseline_result]
    
    print(f"\n{'Configuration':<35} {'P&L':<12} {'P&L %':<10} {'Trades':<10} {'Blocked':<10} {'Win Rate':<10}")
    print("-" * 80)
    
    for r in results:
        config = r['description']
        if len(config) > 35:
            config = config[:32] + "..."
        print(f"{config:<35} "
              f"${r['pnl']:>10,.2f}  "
              f"{r['pnl_percent']:>8.2f}%  "
              f"{r['num_trades']:>8}  "
              f"{r['signals_blocked']:>9}  "
              f"{r['win_rate']:>8.1f}%")
    
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    
    # Production vs Test comparison
    print(f"\n📊 PRODUCTION (Hourly bars, 0.3% threshold):")
    print(f"   Final P&L:        ${prod_result['pnl']:,.2f} ({prod_result['pnl_percent']:.2f}%)")
    print(f"   Total trades:     {prod_result['num_trades']}")
    print(f"   Signals blocked:  {prod_result['signals_blocked']} ({prod_result['block_rate']:.1f}%)")
    print(f"   Max drawdown:     {prod_result['max_drawdown']:.2f}%")
    print(f"   Win rate:         {prod_result['win_rate']:.1f}%")
    print(f"   Profit factor:    {prod_result['profit_factor']:.2f}")
    
    print(f"\n🧪 TEST (1-minute bars, 0.3% threshold):")
    print(f"   Final P&L:        ${test_result['pnl']:,.2f} ({test_result['pnl_percent']:.2f}%)")
    print(f"   Total trades:     {test_result['num_trades']}")
    print(f"   Signals blocked:  {test_result['signals_blocked']} ({test_result['block_rate']:.1f}%)")
    print(f"   Max drawdown:     {test_result['max_drawdown']:.2f}%")
    print(f"   Win rate:         {test_result['win_rate']:.1f}%")
    print(f"   Profit factor:    {test_result['profit_factor']:.2f}")
    
    print(f"\n⚠️  BASELINE (1-minute bars, NO threshold):")
    print(f"   Final P&L:        ${baseline_result['pnl']:,.2f} ({baseline_result['pnl_percent']:.2f}%)")
    print(f"   Total trades:     {baseline_result['num_trades']}")
    print(f"   Win rate:         {baseline_result['win_rate']:.1f}%")
    
    # Calculate improvements
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    test_vs_baseline = test_result['pnl'] - baseline_result['pnl']
    test_vs_prod = test_result['pnl'] - prod_result['pnl']
    
    print(f"\n1️⃣ Impact of proximity threshold (0.3%):")
    print(f"   Test vs Baseline improvement: ${test_vs_baseline:,.2f}")
    print(f"   Trade reduction: {baseline_result['num_trades'] - test_result['num_trades']} fewer trades")
    print(f"   Success rate: Prevented {test_result['signals_blocked']} bad trades")
    
    print(f"\n2️⃣ 1-minute vs Hourly bars (both with 0.3% threshold):")
    if test_vs_prod > 0:
        print(f"   Test system outperforms by: ${test_vs_prod:,.2f}")
    else:
        print(f"   Production system outperforms by: ${-test_vs_prod:,.2f}")
    print(f"   Test system trades: {test_result['num_trades']}")  
    print(f"   Production trades: {prod_result['num_trades']}")
    
    print(f"\n3️⃣ Risk metrics:")
    print(f"   Production max drawdown: {prod_result['max_drawdown']:.2f}%")
    print(f"   Test max drawdown: {test_result['max_drawdown']:.2f}%")
    print(f"   Baseline max drawdown: {baseline_result['max_drawdown']:.2f}%")
    
    # Show sample trades
    print("\n" + "=" * 80)
    print("SAMPLE TRADES")
    print("=" * 80)
    
    if prod_result['trades']:
        print("\n📊 Production - Last 3 trades:")
        for trade in prod_result['trades'][-3:]:
            print(f"   {trade['time']} - {trade['type']} @ ${trade['price']:,.0f} (proximity: {trade['proximity']:.2f}%)")
    
    if test_result['trades']:
        print("\n🧪 Test - Last 3 trades:")
        for trade in test_result['trades'][-3:]:
            print(f"   {trade['time']} - {trade['type']} @ ${trade['price']:,.0f} (proximity: {trade['proximity']:.2f}%)")

if __name__ == '__main__':
    main()