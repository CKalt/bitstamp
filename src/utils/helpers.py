# src/utils/helpers.py

import pandas as pd

def ensure_datetime_index(df):
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.set_index('datetime', inplace=True)
    return df

def print_strategy_results(strategies):
    print("\nDetailed Strategy Results:")
    for name, result in strategies.items():
        print(f"{name}:")
        for key, value in result.items():
            print(f"{key}: {value}")
        print()
