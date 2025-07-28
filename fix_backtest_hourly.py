#!/usr/bin/env python3
"""
CRITICAL FIX: Ensure backtest uses hourly bars like the live system
"""
import pandas as pd
from datetime import datetime, timedelta

def load_and_resample_data(log_file, start_date=None, end_date=None):
    """
    Load data and resample to hourly bars - MATCHING LIVE SYSTEM
    """
    print(f"Loading data from {log_file}...")
    
    # Import the data loader
    import sys
    sys.path.append('src')
    from data.loader import parse_log_file
    
    # Load raw tick data
    df = parse_log_file(log_file, start_date=start_date, end_date=end_date)
    
    if df is None or len(df) == 0:
        raise ValueError("No data loaded")
    
    # Ensure datetime index
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
    
    print(f"Loaded {len(df):,} ticks from {df.index[0]} to {df.index[-1]}")
    
    # CRITICAL: Resample to hourly bars like the live system does
    print("Resampling to hourly bars (matching live system)...")
    df_hourly = df.resample('1H').agg({
        'price': ['first', 'max', 'min', 'last'],
        'amount': 'sum'
    })
    
    # Flatten columns
    df_hourly.columns = ['open', 'high', 'low', 'close', 'volume']
    df_hourly = df_hourly.dropna()
    
    print(f"Resampled to {len(df_hourly):,} hourly bars")
    print(f"Date range: {df_hourly.index[0]} to {df_hourly.index[-1]}")
    
    return df_hourly


def test_ma_signals(df_hourly, ma_short=6, ma_long=34):
    """
    Test MA signals on hourly data
    """
    # Add MAs
    df_hourly['MA_short'] = df_hourly['close'].rolling(window=ma_short).mean()
    df_hourly['MA_long'] = df_hourly['close'].rolling(window=ma_long).mean()
    
    # Generate signals
    df_hourly['signal'] = 0
    df_hourly.loc[df_hourly['MA_short'] > df_hourly['MA_long'], 'signal'] = 1
    df_hourly.loc[df_hourly['MA_short'] < df_hourly['MA_long'], 'signal'] = -1
    
    # Count signal changes (trades)
    df_hourly['signal_change'] = df_hourly['signal'].diff() != 0
    trade_count = df_hourly['signal_change'].sum() - 1  # Subtract initial NaN
    
    print(f"\nMA {ma_short}/{ma_long} Analysis:")
    print(f"Total hourly bars: {len(df_hourly)}")
    print(f"Signal changes (trades): {trade_count}")
    print(f"Average hours between trades: {len(df_hourly) / max(trade_count, 1):.1f}")
    
    # Show recent signals
    print("\nLast 10 hourly bars:")
    print(df_hourly[['close', 'MA_short', 'MA_long', 'signal']].tail(10))
    
    return df_hourly


if __name__ == "__main__":
    # Test with 7 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print("="*60)
    print("TESTING HOURLY RESAMPLING")
    print("="*60)
    
    # Load and resample
    df_hourly = load_and_resample_data('btcusd.log', start_date, end_date)
    
    # Test MA signals
    test_ma_signals(df_hourly, ma_short=6, ma_long=34)
    
    print("\n" + "="*60)
    print("COMPARISON:")
    print("- Tick data: ~250,000 points per week")
    print(f"- Hourly data: {len(df_hourly)} bars per week")
    print("- This matches how the live system processes data!")
    print("="*60)