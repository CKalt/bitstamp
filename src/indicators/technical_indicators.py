# src/indicators/technical_indicators.py

import pandas as pd
import numpy as np

def ensure_datetime_index(df):
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.set_index('datetime', inplace=True)
    return df

def calculate_market_conditions(df, lookback_short=5, lookback_long=20):
    """
    Calculate market conditions including regime, volatility state, and volume profile
    """
    df = ensure_datetime_index(df)
    
    # Price movement and volatility
    df['returns'] = df['price'].pct_change()
    df['volatility_short'] = df['returns'].rolling(lookback_short).std()
    df['volatility_long'] = df['returns'].rolling(lookback_long).std()
    df['volatility_ratio'] = df['volatility_short'] / df['volatility_long']
    
    # Volume profile
    df['volume'] = df['price'] * df['amount']
    df['volume_sma'] = df['volume'].rolling(lookback_long).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Trend strength
    df['price_sma_short'] = df['price'].rolling(lookback_short).mean()
    df['price_sma_long'] = df['price'].rolling(lookback_long).mean()
    df['trend_strength'] = (df['price_sma_short'] - df['price_sma_long']) / df['price_sma_long']
    
    # Regime classification
    df['regime'] = 0  # Default to neutral regime
    
    # Trending regime conditions
    trending_conditions = (
        (df['trend_strength'].abs() > df['trend_strength'].rolling(lookback_long).std()) &
        (df['volume_ratio'] > 1.0)
    )
    df.loc[trending_conditions, 'regime'] = 1
    
    # Volatile regime conditions
    volatile_conditions = (
        (df['volatility_ratio'] > 1.2) &
        (df['volume_ratio'] > 1.5)
    )
    df.loc[volatile_conditions, 'regime'] = 2
    
    # Range-bound regime conditions
    range_bound_conditions = (
        (df['volatility_ratio'] < 0.8) &
        (df['trend_strength'].abs() < df['trend_strength'].rolling(lookback_long).std() * 0.5)
    )
    df.loc[range_bound_conditions, 'regime'] = -1
    
    return df

def calculate_adaptive_vwma(df, base_window=10):
    """
    Calculate VWMA with adaptive parameters based on market conditions
    """
    df = ensure_datetime_index(df)
    
    # Calculate market conditions
    df = calculate_market_conditions(df)
    
    # Adaptive VWMA window based on regime
    df['adaptive_window'] = base_window
    df.loc[df['regime'] == 1, 'adaptive_window'] = base_window * 0.5  # Shorter in trending
    df.loc[df['regime'] == 2, 'adaptive_window'] = base_window * 0.3  # Shortest in volatile
    df.loc[df['regime'] == -1, 'adaptive_window'] = base_window * 1.5  # Longer in range-bound
    
    # Calculate adaptive VWMA
    df['vol_price'] = df['price'] * df['volume']
    
    df['VWMA'] = np.nan
    for i in range(len(df)):
        window_size = int(df['adaptive_window'].iloc[i])
        if i >= window_size:
            vol_price_sum = df['vol_price'].iloc[i-window_size+1:i+1].sum()
            volume_sum = df['volume'].iloc[i-window_size+1:i+1].sum()
            df['VWMA'].iloc[i] = vol_price_sum / volume_sum if volume_sum != 0 else np.nan
        else:
            df['VWMA'].iloc[i] = np.nan

    # Calculate additional signals
    df['VWMA_slope'] = df['VWMA'].pct_change(periods=3)
    df['price_to_vwma'] = df['price'] / df['VWMA'] - 1
    
    return df

def generate_adaptive_vwma_signals(df, vol_scale=1.0):
    """
    Generate trading signals with regime-based adaptivity
    """
    df['Adaptive_VWMA_Signal'] = 0
    
    # Base volume threshold varies by regime
    df['vol_threshold'] = 1.1  # Default
    df.loc[df['regime'] == 1, 'vol_threshold'] = 1.0  # Lower in trending
    df.loc[df['regime'] == 2, 'vol_threshold'] = 1.3  # Higher in volatile
    df.loc[df['regime'] == -1, 'vol_threshold'] = 1.2  # Moderate in range-bound
    
    # Adjust thresholds by scale parameter
    df['vol_threshold'] = df['vol_threshold'] * vol_scale
    
    # Generate signals based on regime
    for regime in [-1, 0, 1, 2]:
        regime_mask = df['regime'] == regime
        
        if regime == 1:  # Trending regime
            # More sensitive to crossovers, strong volume confirmation
            long_conditions = regime_mask & (
                (df['price'] > df['VWMA']) &
                (df['VWMA_slope'] > 0) &
                (df['volume_ratio'] > df['vol_threshold'])
            )
            short_conditions = regime_mask & (
                (df['price'] < df['VWMA']) &
                (df['VWMA_slope'] < 0) &
                (df['volume_ratio'] > df['vol_threshold'])
            )
            
        elif regime == 2:  # Volatile regime
            # Quick reversals, very strict volume confirmation
            long_conditions = regime_mask & (
                (df['price_to_vwma'] < -0.02) &  # Oversold
                (df['VWMA_slope'].shift(1) < 0) & (df['VWMA_slope'] > 0) &  # Slope reversal
                (df['volume_ratio'] > df['vol_threshold'])
            )
            short_conditions = regime_mask & (
                (df['price_to_vwma'] > 0.02) &  # Overbought
                (df['VWMA_slope'].shift(1) > 0) & (df['VWMA_slope'] < 0) &  # Slope reversal
                (df['volume_ratio'] > df['vol_threshold'])
            )
            
        elif regime == -1:  # Range-bound regime
            # Mean reversion signals
            long_conditions = regime_mask & (
                (df['price_to_vwma'] < -0.01) &
                (df['volume_ratio'] > df['vol_threshold']) &
                (df['volatility_ratio'] < 1.0)
            )
            short_conditions = regime_mask & (
                (df['price_to_vwma'] > 0.01) &
                (df['volume_ratio'] > df['vol_threshold']) &
                (df['volatility_ratio'] < 1.0)
            )
            
        else:  # Neutral regime
            # Conservative signals
            long_conditions = regime_mask & (
                (df['price'] > df['VWMA']) &
                (df['volume_ratio'] > df['vol_threshold']) &
                (df['VWMA_slope'] > 0)
            )
            short_conditions = regime_mask & (
                (df['price'] < df['VWMA']) &
                (df['volume_ratio'] > df['vol_threshold']) &
                (df['VWMA_slope'] < 0)
            )
        
        df.loc[long_conditions, 'Adaptive_VWMA_Signal'] = 1
        df.loc[short_conditions, 'Adaptive_VWMA_Signal'] = -1
    
    return df


def add_moving_averages(df, short_window, long_window):
    # Removed the call to ensure_datetime_index(df)
    df[f'MA_Short_{short_window}'] = df['price'].rolling(window=short_window).mean()
    df[f'MA_Long_{long_window}'] = df['price'].rolling(window=long_window).mean()
    return df

def generate_ma_signals(df):
    df['MA_Signal'] = 0
    df.loc[df['Short_MA'] > df['Long_MA'], 'MA_Signal'] = 1
    df.loc[df['Short_MA'] < df['Long_MA'], 'MA_Signal'] = -1
    return df

def calculate_rsi(df, window=14):
    df = ensure_datetime_index(df)
    delta = df['price'].diff()
    gain = (delta.clip(lower=0)).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def generate_rsi_signals(df, overbought=70, oversold=30):
    df['RSI_Signal'] = 0
    df.loc[df['RSI'] < oversold, 'RSI_Signal'] = 1
    df.loc[df['RSI'] > overbought, 'RSI_Signal'] = -1
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    df = ensure_datetime_index(df)
    df['BB_MA'] = df['price'].rolling(window=window).mean()
    df['BB_STD'] = df['price'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_MA'] + (df['BB_STD'] * num_std)
    df['BB_Lower'] = df['BB_MA'] - (df['BB_STD'] * num_std)
    return df

def generate_bollinger_band_signals(df):
    df['BB_Signal'] = 0
    df.loc[df['price'] < df['BB_Lower'], 'BB_Signal'] = 1  # Buy signal
    df.loc[df['price'] > df['BB_Upper'], 'BB_Signal'] = -1  # Sell signal
    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    df = ensure_datetime_index(df)
    df['MACD_Fast'] = df['price'].ewm(span=fast, adjust=False).mean()
    df['MACD_Slow'] = df['price'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['MACD_Fast'] - df['MACD_Slow']
    df['MACD_Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    return df

def generate_macd_signals(df):
    df['MACD_Signal'] = 0
    df.loc[df['MACD'] > df['MACD_Signal_Line'], 'MACD_Signal'] = 1
    df.loc[df['MACD'] < df['MACD_Signal_Line'], 'MACD_Signal'] = -1
    return df
