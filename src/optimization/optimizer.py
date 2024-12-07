# src/optimization/optimizer.py

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from itertools import product

from backtesting.backtester import backtest
from indicators.technical_indicators import (
    calculate_adaptive_vwma,
    generate_adaptive_vwma_signals,
    add_moving_averages,
    generate_ma_signals,
    calculate_rsi,
    generate_rsi_signals,
    calculate_bollinger_bands,
    generate_bollinger_band_signals,
    calculate_macd,
    generate_macd_signals,
    ensure_datetime_index
)
from strategies.ramm_strategy import calculate_ramm_signals

def optimize_adaptive_vwma_parameters(df,
                                      base_window_range=range(5, 21, 3),
                                      vol_scale_range=np.arange(0.8, 1.4, 0.1)):
    """
    Optimize Adaptive VWMA parameters
    """
    results = []
    total_combinations = len(base_window_range) * len(vol_scale_range)

    print(f"Testing {total_combinations} parameter combinations...")

    with tqdm(total=total_combinations, desc="Optimizing Adaptive VWMA Parameters") as pbar:
        for base_window in base_window_range:
            for vol_scale in vol_scale_range:
                df_test = df.copy()

                # Calculate adaptive VWMA
                df_test = calculate_adaptive_vwma(df_test, base_window=base_window)

                # Generate signals
                df_test = generate_adaptive_vwma_signals(df_test, vol_scale=vol_scale)

                # Run backtest
                metrics = backtest(df_test, 'Adaptive_VWMA')
                average_trades_per_day = metrics['Average_Trades_Per_Day']

                if 1 <= average_trades_per_day <= 4 and metrics['Total_Return'] > 0:
                    results.append({
                        'Strategy': 'Adaptive_VWMA',
                        'Base_Window': base_window,
                        'Volume_Scale': vol_scale,
                        **metrics
                    })

                pbar.update(1)

    return pd.DataFrame(results)

def optimize_ramm_parameters(df, max_iterations=50):
    """
    Optimize RAMM strategy parameters using grid search
    """
    results = []

    # Define parameter ranges
    ma_short_range = range(4, 21, 2)
    ma_long_range = range(20, 51, 5)
    rsi_period_range = range(10, 21, 2)
    rsi_ob_range = range(65, 81, 5)
    rsi_os_range = range(20, 36, 5)
    regime_lookback_range = range(15, 31, 5)

    # Create parameter combinations
    param_combinations = list(product(
        ma_short_range, ma_long_range,
        rsi_period_range, rsi_ob_range, rsi_os_range,
        regime_lookback_range
    ))

    # Limit combinations if needed
    if len(param_combinations) > max_iterations:
        param_combinations = random.sample(param_combinations, max_iterations)

    for params in tqdm(param_combinations, desc="Optimizing RAMM Parameters"):
        ma_short, ma_long, rsi_period, rsi_ob, rsi_os, regime_lookback = params

        if ma_short >= ma_long or rsi_os >= rsi_ob:
            continue

        df_test = df.copy()
        df_test = calculate_ramm_signals(
            df_test, ma_short, ma_long,
            rsi_period, rsi_ob, rsi_os,
            regime_lookback
        )

        metrics = backtest(df_test, 'RAMM')
        if 1 <= metrics['Average_Trades_Per_Day'] <= 4 and metrics['Total_Return'] > 0:
            results.append({
                'Strategy': 'RAMM',
                'MA_Short': ma_short,
                'MA_Long': ma_long,
                'RSI_Period': rsi_period,
                'RSI_Overbought': rsi_ob,
                'RSI_Oversold': rsi_os,
                'Regime_Lookback': regime_lookback,
                **metrics
            })

    return pd.DataFrame(results)

def optimize_ma_parameters(df, short_range, long_range):
    results = []
    for short_window in short_range:
        for long_window in long_range:
            if short_window >= long_window:
                continue
            df_test = add_moving_averages(df.copy(), short_window, long_window)
            df_test = generate_ma_signals(df_test)
            metrics = backtest(df_test, 'MA')
            average_trades_per_day = metrics['Average_Trades_Per_Day']
            if 1 <= average_trades_per_day <= 4 and metrics['Total_Return'] > 0:
                results.append({
                    'Strategy': 'MA',
                    'Short_Window': short_window,
                    'Long_Window': long_window,
                    **metrics
                })
    return pd.DataFrame(results)

def optimize_rsi_parameters(df, window_range, overbought_range, oversold_range):
    results = []
    for window in window_range:
        for overbought in overbought_range:
            for oversold in oversold_range:
                if oversold >= overbought:
                    continue
                df_test = calculate_rsi(df.copy(), window)
                df_test = generate_rsi_signals(df_test, overbought, oversold)
                metrics = backtest(df_test, 'RSI')
                average_trades_per_day = metrics['Average_Trades_Per_Day']
                if 1 <= average_trades_per_day <= 4 and metrics['Total_Return'] > 0:
                    results.append({
                        'Strategy': 'RSI',
                        'RSI_Window': window,
                        'Overbought': overbought,
                        'Oversold': oversold,
                        **metrics
                    })
    return pd.DataFrame(results)

def optimize_hft_parameters(df, strategy, **kwargs):
    results = []
    param_grid = kwargs['param_grid']
    for params in param_grid:
        df_test = df.copy()
        if strategy == 'BB':
            df_test = calculate_bollinger_bands(
                df_test, window=params['window'], num_std=params['num_std'])
            df_test = generate_bollinger_band_signals(df_test)
        elif strategy == 'MACD':
            df_test = calculate_macd(
                df_test, fast=params['fast'], slow=params['slow'], signal=params['signal'])
            df_test = generate_macd_signals(df_test)
        else:
            continue  # Skip if strategy not recognized
        metrics = backtest(df_test, strategy)
        average_trades_per_day = metrics['Average_Trades_Per_Day']
        if 1 <= average_trades_per_day <= 4 and metrics['Total_Return'] > 0:
            results.append({
                'Strategy': strategy,
                **params,
                **metrics
            })
    return pd.DataFrame(results)

def optimize_ma_frequency(df, base_parameters):
    """
    Optimize trading frequency for a given MA strategy configuration.
    
    Args:
        df: DataFrame with price data
        base_parameters: Dict containing 'Short_Window' and 'Long_Window'
        
    Returns:
        pd.DataFrame: Results of frequency optimization
    """
    frequencies = ['5min', '15min', '30min', '1H', '2H', '4H', '6H', '12H', '1D']
    results = []
    
    print(f"Testing {len(frequencies)} different frequencies...")
    
    with tqdm(total=len(frequencies), desc="Optimizing Trading Frequency") as pbar:
        for freq in frequencies:
            # Resample data to current frequency
            df_resampled = df.resample(freq).agg({
                'price': 'last',
                'amount': 'sum',
                'volume': 'sum'
            }).dropna()
            
            # Run strategy with base parameters
            df_strategy = add_moving_averages(
                df_resampled.copy(),
                short_window=base_parameters['Short_Window'],
                long_window=base_parameters['Long_Window'],
                price_col='price'  # Specify the price column name
            )
            df_strategy = generate_ma_signals(df_strategy)
            
            # Run backtest
            metrics = backtest(df_strategy, 'MA')
            average_trades_per_day = metrics['Average_Trades_Per_Day']
            
            if 1 <= average_trades_per_day <= 4 and metrics['Total_Return'] > 0:
                results.append({
                    'Frequency': freq,
                    'Strategy': 'MA',
                    'Short_Window': base_parameters['Short_Window'],
                    'Long_Window': base_parameters['Long_Window'],
                    **metrics
                })
            
            pbar.update(1)
    
    return pd.DataFrame(results)