# src/strategies/ramm_strategy.py

from indicators.technical_indicators import (
    ensure_datetime_index,
    add_moving_averages,
    calculate_rsi,
    calculate_market_conditions
)

def calculate_ramm_signals(df,
                           ma_short=10, ma_long=50,  # MA parameters
                           rsi_period=14, rsi_ob=70, rsi_os=30,  # RSI parameters
                           regime_lookback=20):
    """
    Generate RAMM strategy signals combining MA Crossover and RSI based on market regime
    """
    df = ensure_datetime_index(df)

    # Calculate market regime
    df_regime = calculate_market_conditions(df.copy(), regime_lookback)
    df['regime'] = df_regime['regime']

    # Calculate MA signals
    df = add_moving_averages(df, ma_short, ma_long)
    df['MA_Signal'] = 0
    df.loc[df['Short_MA'] > df['Long_MA'], 'MA_Signal'] = 1
    df.loc[df['Short_MA'] < df['Long_MA'], 'MA_Signal'] = -1

    # Calculate RSI signals
    df = calculate_rsi(df, rsi_period)
    df['RSI_Signal'] = 0
    df.loc[df['RSI'] < rsi_os, 'RSI_Signal'] = 1
    df.loc[df['RSI'] > rsi_ob, 'RSI_Signal'] = -1

    # Generate RAMM signals based on regime
    df['RAMM_Signal'] = 0

    # Trending regime: use MA Crossover
    df.loc[df['regime'] == 1, 'RAMM_Signal'] = df.loc[df['regime'] == 1, 'MA_Signal']

    # Mean-reverting regime: use RSI
    df.loc[df['regime'] == -1, 'RAMM_Signal'] = df.loc[df['regime'] == -1, 'RSI_Signal']

    # Mixed regime: combine signals (only take trades when both agree)
    mixed_mask = df['regime'] == 0
    df.loc[mixed_mask & (df['MA_Signal'] == 1) & (df['RSI_Signal'] == 1), 'RAMM_Signal'] = 1
    df.loc[mixed_mask & (df['MA_Signal'] == -1) & (df['RSI_Signal'] == -1), 'RAMM_Signal'] = -1

    return df
