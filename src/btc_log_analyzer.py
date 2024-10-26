#!/usr/bin/env python
# src/btc_log_analyzer.py

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import argparse
import random
from tqdm import tqdm
from itertools import product
from multiprocessing import Pool
from scipy import stats


def create_metadata_file(log_file_path, metadata_file_path):
    print("Creating metadata file...")
    metadata = {}
    total_lines = 0
    last_timestamp = None

    with open(log_file_path, 'r') as file:
        for line in file:
            total_lines += 1
            if total_lines % 1000000 == 0:
                print(f"Processed {total_lines} lines...")
            try:
                json_data = json.loads(line)
                if json_data['event'] == 'trade':
                    timestamp = int(json_data['data']['timestamp'])
                    date = datetime.fromtimestamp(timestamp).date()
                    if date not in metadata:
                        metadata[str(date)] = {
                            'start_line': total_lines, 'timestamp': timestamp}
                    last_timestamp = timestamp
            except json.JSONDecodeError:
                continue

    metadata['total_lines'] = total_lines
    metadata['last_timestamp'] = last_timestamp

    with open(metadata_file_path, 'w') as file:
        json.dump(metadata, file)

    print(f"Metadata file created: {metadata_file_path}")


def get_start_line_from_metadata(metadata_file_path, start_date):
    with open(metadata_file_path, 'r') as file:
        metadata = json.load(file)

    start_date_str = str(start_date.date())
    if start_date_str in metadata:
        return metadata[start_date_str]['start_line']
    else:
        # If exact date not found, find the nearest date
        dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in metadata.keys()
                 if date != 'total_lines' and date != 'last_timestamp']
        nearest_date = min(dates, key=lambda x: abs(x - start_date.date()))
        return metadata[str(nearest_date)]['start_line']


def parse_log_file(file_path, start_date=None, end_date=None):
    metadata_file_path = f"{file_path}.metadata"
    if not os.path.exists(metadata_file_path):
        create_metadata_file(file_path, metadata_file_path)

    data = []
    with open(metadata_file_path, 'r') as file:
        metadata = json.load(file)

    total_lines = metadata['total_lines']
    print(f"Total lines in log file: {total_lines}")

    start_line = 1
    if start_date:
        start_line = get_start_line_from_metadata(metadata_file_path, start_date)
        print(f"Starting from line {start_line} based on start date {start_date}")

    last_date = None
    skipped_count = start_line - 1
    processed_count = 0
    end_reached = False
    progress_interval = total_lines // 10  # Only show 10 progress updates

    with open(file_path, 'r') as file:
        for i, line in enumerate(file, 1):
            if i < start_line:
                continue

            if i % progress_interval == 0:  # Show progress every 10%
                print(f"Progress: {i/total_lines*100:.1f}% - Last date: {last_date}")

            try:
                json_data = json.loads(line)
                if json_data['event'] == 'trade':
                    trade_data = json_data['data']
                    timestamp = int(trade_data['timestamp'])
                    trade_date = datetime.fromtimestamp(timestamp)
                    last_date = trade_date.strftime('%Y-%m-%d %H:%M:%S')

                    if end_date and trade_date > end_date:
                        end_reached = True
                        break

                    if start_date and trade_date < start_date:
                        skipped_count += 1
                        continue

                    processed_count += 1
                    data.append({
                        'timestamp': timestamp,
                        'price': float(trade_data['price']),
                        'amount': float(trade_data['amount']),
                        'type': int(trade_data['type'])
                    })
            except json.JSONDecodeError:
                continue

    print(f"\nFinished processing log file. Last date processed: {last_date}")
    print(f"Total entries skipped: {skipped_count}")
    print(f"Total entries processed: {processed_count}")
    if end_reached:
        print(f"Reached end date: {end_date}")
    print("Creating DataFrame...")
    df = pd.DataFrame(data)

    # Optimize data types
    df['price'] = pd.to_numeric(df['price'], downcast='float')
    df['amount'] = pd.to_numeric(df['amount'], downcast='float')
    df['type'] = df['type'].astype('int8')

    return df


def analyze_data(df):
    print("Converting timestamp to datetime...")
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    print("Calculating basic statistics...")
    print(df.describe())
    print("Plotting price over time...")
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['price'])
    plt.title('Bitcoin Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.savefig('btc_price_over_time.png')
    plt.close()
    print("Calculating and plotting hourly trading volume...")
    df['volume'] = df['price'] * df['amount']
    df.set_index('datetime', inplace=True)
    hourly_volume = df['volume'].resample('H').sum()
    plt.figure(figsize=(12, 6))
    plt.bar(hourly_volume.index, hourly_volume.values)
    plt.title('Hourly Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume (USD)')
    plt.savefig('btc_hourly_volume.png')
    plt.close()
    df.reset_index(inplace=True)


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
    
    # Rolling VWMA with adaptive window
    df['VWMA'] = df.apply(
        lambda x: (df['vol_price'].rolling(int(x['adaptive_window'])).sum() / 
                   df['volume'].rolling(int(x['adaptive_window'])).sum())[x.name],
        axis=1
    )
    
    # Calculate additional signals
    df['VWMA_slope'] = df['VWMA'].pct_change(periods=3)  # Momentum indicator
    df['price_to_vwma'] = df['price'] / df['VWMA'] - 1  # Relative price position
    
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
    df = ensure_datetime_index(df)
    df['Short_MA'] = df['price'].rolling(window=short_window).mean()
    df['Long_MA'] = df['price'].rolling(window=long_window).mean()
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


def calculate_ramm_signals(df,
                           ma_short=10, ma_long=50,  # MA parameters
                           rsi_period=14, rsi_ob=70, rsi_os=30,  # RSI parameters
                           regime_lookback=20):
    """
    Generate RAMM strategy signals combining MA Crossover and RSI based on market regime
    """
    df = ensure_datetime_index(df)

    # Calculate market regime
    df['regime'] = calculate_market_conditions(df, regime_lookback)['regime']

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
    df.loc[df['regime'] == -1,
           'RAMM_Signal'] = df.loc[df['regime'] == -1, 'RSI_Signal']

    # Mixed regime: combine signals (only take trades when both agree)
    mixed_mask = df['regime'] == 0
    df.loc[mixed_mask & (df['MA_Signal'] == 1) & (
        df['RSI_Signal'] == 1), 'RAMM_Signal'] = 1
    df.loc[mixed_mask & (df['MA_Signal'] == -1) &
           (df['RSI_Signal'] == -1), 'RAMM_Signal'] = -1

    return df


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
        metrics = backtest(df_test, strategy)
        average_trades_per_day = metrics['Average_Trades_Per_Day']
        if 1 <= average_trades_per_day <= 4 and metrics['Total_Return'] > 0:
            results.append({
                'Strategy': strategy,
                **params,
                **metrics
            })
    return pd.DataFrame(results)


def backtest(df, strategy, initial_balance=10000, position_size=0.1, transaction_cost=0.001, max_trades_per_day=10):
    df = ensure_datetime_index(df)
    df['Position'] = df[f'{strategy}_Signal'].shift(1).fillna(0)
    df['Returns'] = df['price'].pct_change().fillna(0)
    df['Strategy_Returns'] = df['Position'] * df['Returns']

    # Calculate transaction costs
    df['Trade'] = df['Position'].diff().abs()
    df['Transaction_Costs'] = df['Trade'] * transaction_cost

    # Limit trades per day
    df['Daily_Trades'] = df['Trade'].groupby(df.index.date).cumsum()
    df.loc[df['Daily_Trades'] > max_trades_per_day, 'Strategy_Returns'] = 0

    df['Strategy_Returns'] -= df['Transaction_Costs']
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Balance'] = initial_balance * df['Cumulative_Returns']

    total_trades = df['Trade'].sum()

    # Calculate average trades per day
    trading_days = (df.index[-1].date() - df.index[0].date()).days + 1
    average_trades_per_day = total_trades / trading_days if trading_days > 0 else 0

    # Profit factor and Sharpe ratio
    positive_returns = df.loc[df['Strategy_Returns']
                              > 0, 'Strategy_Returns'].sum()
    negative_returns = -df.loc[df['Strategy_Returns']
                               < 0, 'Strategy_Returns'].sum()
    profit_factor = positive_returns / \
        negative_returns if negative_returns != 0 else np.inf

    sharpe_ratio = df['Strategy_Returns'].mean() / df['Strategy_Returns'].std() * \
        np.sqrt(252) if df['Strategy_Returns'].std() != 0 else 0

    final_balance = df['Balance'].iloc[-1]
    total_return = (final_balance - initial_balance) / initial_balance * 100

    return {
        'Final_Balance': final_balance,
        'Total_Return': total_return,
        'Total_Trades': total_trades,
        'Average_Trades_Per_Day': average_trades_per_day,
        'Profit_Factor': profit_factor,
        'Sharpe_Ratio': sharpe_ratio
    }


def generate_trade_list(df, strategy):
    trades = []
    position = 0
    entry_price = 0
    entry_time = None

    for index, row in df.iterrows():
        signal = row[f'{strategy}_Signal']

        if signal == 1 and position == 0:  # Enter long position
            position = 1
            entry_price = row['price']
            entry_time = index
        elif signal == -1 and position == 1:  # Exit long position
            exit_price = row['price']
            profit = (exit_price - entry_price) / entry_price
            trades.append({
                'Entry Time': entry_time,
                'Exit Time': index,
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Profit (%)': profit * 100
            })
            position = 0

    return pd.DataFrame(trades)


def run_trading_system(df, high_frequency='1H', low_frequency='15T', max_iterations=50):
    print("Starting run_trading_system function...")
    df = ensure_datetime_index(df)
    print(f"DataFrame shape after ensuring datetime index: {df.shape}")

    # Resample data to higher timeframe
    print(f"Resampling data to higher timeframe ({high_frequency})...")
    df['volume'] = df['price'] * df['amount']
    df_high = df.resample(high_frequency).agg({
        'price': 'last',
        'amount': 'sum',
        'volume': 'sum'
    }).dropna()
    df_high['timestamp'] = df_high.index.view('int64') // 10**9
    print(f"Higher timeframe DataFrame shape: {df_high.shape}")

    # Resample data to lower timeframe
    print(f"Resampling data to lower timeframe ({low_frequency})...")
    df_low = df.resample(low_frequency).agg({
        'price': 'last',
        'amount': 'sum',
        'volume': 'sum'
    }).dropna()
    df_low['timestamp'] = df_low.index.view('int64') // 10**9
    print(f"Lower timeframe DataFrame shape: {df_low.shape}")

    strategies = {}
    all_results_list = []

    # Run MA Crossover Strategy
    print("Running MA Crossover Strategy...")
    ma_results = optimize_ma_parameters(
        df_high, range(4, 25, 2), range(26, 51, 2))
    if not ma_results.empty:
        best_ma = ma_results.loc[ma_results['Total_Return'].idxmax()]
        print("Best MA Crossover parameters:")
        print(best_ma)
        strategies['MA Crossover'] = best_ma.to_dict()
        all_results_list.append(ma_results)

        print("Generating trade list for best MA strategy...")
        best_ma_df = add_moving_averages(
            df_high.copy(), best_ma['Short_Window'], best_ma['Long_Window'])
        best_ma_df = generate_ma_signals(best_ma_df)
        ma_trades = generate_trade_list(best_ma_df, 'MA')
        ma_trades.to_csv('ma_trades.csv', index=False)
        print("MA Crossover trades saved to 'ma_trades.csv'")
    else:
        print("No MA Crossover strategies met the criteria.")

    # Run RSI Strategy
    print("Running RSI Strategy...")
    rsi_results = optimize_rsi_parameters(df_high, range(
        10, 21, 2), range(65, 81, 5), range(20, 36, 5))
    if not rsi_results.empty:
        best_rsi = rsi_results.loc[rsi_results['Total_Return'].idxmax()]
        print("Best RSI parameters:")
        print(best_rsi)
        strategies['RSI'] = best_rsi.to_dict()
        all_results_list.append(rsi_results)

        print("Generating trade list for best RSI strategy...")
        best_rsi_df = calculate_rsi(df_high.copy(), best_rsi['RSI_Window'])
        best_rsi_df = generate_rsi_signals(
            best_rsi_df, best_rsi['Overbought'], best_rsi['Oversold'])
        rsi_trades = generate_trade_list(best_rsi_df, 'RSI')
        rsi_trades.to_csv('rsi_trades.csv', index=False)
        print("RSI trades saved to 'rsi_trades.csv'")
    else:
        print("No RSI strategies met the criteria.")

    # Run Bollinger Bands Strategy
    print("Running Bollinger Bands Strategy...")
    bb_param_grid = [{'window': w, 'num_std': s}
                     for w in range(10, 31, 5) for s in [1.5, 2, 2.5]]
    bb_results = optimize_hft_parameters(
        df_low, 'BB', param_grid=bb_param_grid)
    if not bb_results.empty:
        best_bb = bb_results.loc[bb_results['Total_Return'].idxmax()]
        print("Best Bollinger Bands parameters:")
        print(best_bb)
        strategies['Bollinger Bands'] = best_bb.to_dict()
        all_results_list.append(bb_results)

        print("Generating trade list for best Bollinger Bands strategy...")
        best_bb_df = calculate_bollinger_bands(
            df_low.copy(), window=best_bb['window'], num_std=best_bb['num_std'])
        best_bb_df = generate_bollinger_band_signals(best_bb_df)
        bb_trades = generate_trade_list(best_bb_df, 'BB')
        bb_trades.to_csv('bb_trades.csv', index=False)
        print("Bollinger Bands trades saved to 'bb_trades.csv'")
    else:
        print("No Bollinger Bands strategies met the criteria.")

    # Run MACD Strategy
    print("Running MACD Strategy...")
    macd_param_grid = [{'fast': f, 'slow': s, 'signal': sig}
                       for f in [6, 12, 18]
                       for s in [20, 26, 32]
                       for sig in [7, 9, 11]]
    macd_results = optimize_hft_parameters(
        df_low, 'MACD', param_grid=macd_param_grid)
    if not macd_results.empty:
        best_macd = macd_results.loc[macd_results['Total_Return'].idxmax()]
        print("Best MACD parameters:")
        print(best_macd)
        strategies['MACD'] = best_macd.to_dict()
        all_results_list.append(macd_results)

        print("Generating trade list for best MACD strategy...")
        best_macd_df = calculate_macd(df_low.copy(),
                                      fast=best_macd['fast'],
                                      slow=best_macd['slow'],
                                      signal=best_macd['signal'])
        best_macd_df = generate_macd_signals(best_macd_df)
        macd_trades = generate_trade_list(best_macd_df, 'MACD')
        macd_trades.to_csv('macd_trades.csv', index=False)
        print("MACD trades saved to 'macd_trades.csv'")
    else:
        print("No MACD strategies met the criteria.")

    # Run RAMM Strategy
    print("Running RAMM Strategy...")
    ramm_results = optimize_ramm_parameters(df_high, max_iterations=50)
    if not ramm_results.empty:
        best_ramm = ramm_results.loc[ramm_results['Total_Return'].idxmax()]
        print("Best RAMM parameters:")
        print(best_ramm)
        strategies['RAMM'] = best_ramm.to_dict()
        all_results_list.append(ramm_results)

        print("Generating trade list for best RAMM strategy...")
        best_ramm_df = calculate_ramm_signals(
            df_high.copy(),
            ma_short=best_ramm['MA_Short'],
            ma_long=best_ramm['MA_Long'],
            rsi_period=best_ramm['RSI_Period'],
            rsi_ob=best_ramm['RSI_Overbought'],
            rsi_os=best_ramm['RSI_Oversold'],
            regime_lookback=best_ramm['Regime_Lookback']
        )
        ramm_trades = generate_trade_list(best_ramm_df, 'RAMM')
        ramm_trades.to_csv('ramm_trades.csv', index=False)
        print("RAMM trades saved to 'ramm_trades.csv'")
    else:
        print("No RAMM strategies met the criteria.")

    # Run Adaptive VWMA Strategy
    print("Running Adaptive VWMA Strategy...")
    adaptive_vwma_results = optimize_adaptive_vwma_parameters(df_high)
    if not adaptive_vwma_results.empty:
        best_adaptive_vwma = adaptive_vwma_results.loc[adaptive_vwma_results['Total_Return'].idxmax()]
        print("Best Adaptive VWMA parameters:")
        print(best_adaptive_vwma)
        strategies['Adaptive_VWMA'] = best_adaptive_vwma.to_dict()
        all_results_list.append(adaptive_vwma_results)

        print("Generating trade list for best Adaptive VWMA strategy...")
        best_adaptive_vwma_df = calculate_adaptive_vwma(
            df_high.copy(),
            base_window=best_adaptive_vwma['Base_Window']
        )
        best_adaptive_vwma_df = generate_adaptive_vwma_signals(
            best_adaptive_vwma_df,
            vol_scale=best_adaptive_vwma['Volume_Scale']
        )
        adaptive_vwma_trades = generate_trade_list(best_adaptive_vwma_df, 'Adaptive_VWMA')
        adaptive_vwma_trades.to_csv('adaptive_vwma_trades.csv', index=False)
        print("Adaptive VWMA trades saved to 'adaptive_vwma_trades.csv'")
    else:
        print("No Adaptive VWMA strategies met the criteria.")

    if strategies:
        print("Preparing detailed strategy results...")
        print("\nDetailed Strategy Results:")
        for name, result in strategies.items():
            print(f"{name}:")
            for key, value in result.items():
                print(f"{key}: {value}")
            print()

        print("\nStrategy Comparison:")
        comparison_df = pd.DataFrame.from_dict(strategies, orient='index')
        print(comparison_df)

        # Safely extract and compare total returns
        total_returns = {name: result.get('Total_Return', 0)
                         for name, result in strategies.items()}

        if total_returns:
            best_strategy = max(total_returns.items(), key=lambda x: x[1])[0]
            print(f"\nBest overall strategy: {best_strategy}")
            print(
                f"Best strategy Total Return: {total_returns[best_strategy]:.2f}%")

            print("\nAll strategy returns:")
            for strategy, total_return in total_returns.items():
                print(f"{strategy}: {total_return:.2f}%")
        else:
            print("\nNo strategies met the criteria.")

        if all_results_list:
            all_results = pd.concat(all_results_list, ignore_index=True)
            all_results.to_csv('optimization_results.csv', index=False)
            print("\nAll optimization results saved to 'optimization_results.csv'")
        else:
            print("\nNo optimization results to save.")
            all_results = pd.DataFrame()
    else:
        print("No strategies met the criteria. No comparison or results to display.")
        comparison_df = pd.DataFrame()
        all_results = pd.DataFrame()

    return all_results, comparison_df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Bitcoin trade log data.')
    parser.add_argument('--start-window-days-back', type=int, default=30,
                        help='Number of days to subtract from the current date as the start window')
    parser.add_argument('--end-window-days-back', type=int, default=0,
                        help='Number of days to subtract from the current date as the end window')
    parser.add_argument('--trading-window-days', type=int,
                        help='Number of days to analyze from the start date (overrides end-window-days-back)')
    parser.add_argument('--max-iterations', type=int, default=50,
                        help='Maximum number of iterations for parameter optimization')
    # New arguments for sampling frequencies
    parser.add_argument('--high-frequency', type=str, default='1H',
                        help='Sampling frequency for higher timeframe (e.g., "1H" for hourly)')
    parser.add_argument('--low-frequency', type=str, default='15T',
                        help='Sampling frequency for lower timeframe (e.g., "15T" for 15 minutes)')
    args = parser.parse_args()

    file_path = 'btcusd.log'
    metadata_file_path = f"{file_path}.metadata"

    if not os.path.exists(metadata_file_path):
        print("Metadata file not found. Creating it now...")
        create_metadata_file(file_path, metadata_file_path)
        print("Metadata file created.")

    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    print(f"Log file size: {file_size:.2f} MB")

    current_date = datetime.now()

    if args.start_window_days_back > 0:
        start_date = current_date - timedelta(days=args.start_window_days_back)
        print(f"Analyzing data from {start_date} onwards")
    else:
        start_date = None
        print("No start date specified")

    # Handle trading window days if specified
    if args.trading_window_days is not None:
        if start_date is None:
            raise ValueError(
                "Cannot use trading-window-days without specifying start-window-days-back")
        end_date = start_date + timedelta(days=args.trading_window_days)
        print(f"Using trading window of {args.trading_window_days} days")
        print(f"Analysis window: {start_date} to {end_date}")
    elif args.end_window_days_back > 0:
        end_date = current_date - timedelta(days=args.end_window_days_back)
        print(f"Analyzing data up to {end_date}")
    else:
        end_date = None
        print("No end date specified")

    if start_date and end_date and start_date >= end_date:
        raise ValueError("Start date must be earlier than end date")

    print("Starting to parse log file...")
    df = parse_log_file(file_path, start_date, end_date)
    print(f"Parsed {len(df)} trade events.")
    print(f"DataFrame shape after parsing: {df.shape}")
    print(
        f"DataFrame memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")

    print("Starting data analysis...")
    analyze_data(df)
    print("Analysis complete. Check the current directory for generated PNG files.")

    print("Running trading system...")
    try:
        optimization_results, strategy_comparison = run_trading_system(
            df,
            high_frequency=args.high_frequency,
            low_frequency=args.low_frequency,
            max_iterations=args.max_iterations
        )
        print("Trading system analysis complete.")

        if not strategy_comparison.empty:
            print("\nBest strategy for each type:")
            for strategy in strategy_comparison.index:
                print(
                    f"{strategy}: Total Return = {strategy_comparison.loc[strategy, 'Total_Return']:.2f}%")

            if len(strategy_comparison) > 1:
                best_strategy = strategy_comparison['Total_Return'].idxmax()
                print(f"\nOverall best strategy: {best_strategy}")
                print(
                    f"Best strategy Total Return: {strategy_comparison.loc[best_strategy, 'Total_Return']:.2f}%")
        else:
            print("No strategies met the criteria. No comparison results to display.")
    except Exception as e:
        print(f"An error occurred during trading system analysis: {str(e)}")
        print("Partial results may have been saved.")


if __name__ == "__main__":
    main()
