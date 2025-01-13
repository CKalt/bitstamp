# src/utils/analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import traceback
import os
from indicators.technical_indicators import ensure_datetime_index
from backtesting.backtester import generate_trade_list
from optimization.optimizer import (
    optimize_ma_parameters,
    optimize_rsi_parameters,
    optimize_hft_parameters,
    optimize_ramm_parameters,
    optimize_adaptive_vwma_parameters,
    optimize_ma_frequency
)
from strategies.ramm_strategy import calculate_ramm_signals
from indicators.technical_indicators import (
    add_moving_averages,
    generate_ma_signals,
    calculate_rsi,
    generate_rsi_signals,
    calculate_bollinger_bands,
    generate_bollinger_band_signals,
    calculate_macd,
    generate_macd_signals,
    calculate_adaptive_vwma,
    generate_adaptive_vwma_signals
)
from utils.helpers import ensure_datetime_index, print_strategy_results


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
    plt.bar(hourly_volume.index, hourly_volume.values, width=0.02)
    plt.title('Hourly Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume (USD)')
    plt.savefig('btc_hourly_volume.png')
    plt.close()
    df.reset_index(inplace=True)
    print("Analysis complete. Check the current directory for generated PNG files.")


###############################################################################
# CHANGED: Now accepts `config` with constraint settings
###############################################################################
def run_trading_system(df, high_frequency='1H', low_frequency='15T', max_iterations=50, config=None):
    print("Starting run_trading_system function...")
    df = ensure_datetime_index(df)
    print(f"DataFrame shape after ensuring datetime index: {df.shape}")

    # If config is None, default it to an empty dict
    # so we don't get KeyError below
    if config is None:
        config = {}

    # Extract our constraints (with fallback defaults)
    constraints = config.get("strategy_constraints", {})
    min_trades_per_day  = constraints.get("min_trades_per_day", 1)
    max_trades_per_day  = constraints.get("max_trades_per_day", 4)
    min_total_return    = constraints.get("min_total_return", 0.0)
    min_profit_per_trade = constraints.get("min_profit_per_trade", 0.0)

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

    # NEW: Dictionary to hold final DataFrames used for each "best" strategy
    strategy_dfs = {}  # We'll store e.g. strategy_dfs['MA'] = best_ma_df, etc.

    print("Running MA Crossover Strategy...")
    ma_results = optimize_ma_parameters(
        df_high,
        range(4, 25, 2),
        range(26, 51, 2),
        min_trades_per_day=min_trades_per_day,
        max_trades_per_day=max_trades_per_day,
        min_total_return=min_total_return,
        min_profit_per_trade=min_profit_per_trade
    )

    if not ma_results.empty:
        best_ma = ma_results.loc[ma_results['Total_Return'].idxmax()]
        print("Initial MA Crossover parameters:")
        print(best_ma)
        
        # Add frequency optimization for best parameters
        print("\nOptimizing trading frequency for best MA parameters...")
        freq_results = optimize_ma_frequency(df, {
            'Short_Window': int(best_ma['Short_Window']),
            'Long_Window': int(best_ma['Long_Window'])
        })
        
        if not freq_results.empty:
            # Find best frequency result
            best_freq_result = freq_results.loc[freq_results['Total_Return'].idxmax()]
            print("\nBest frequency results:")
            print(best_freq_result)
            
            # Update the best MA parameters with frequency-optimized version
            best_ma = best_freq_result
            
        # Generate trade list using best parameters
        best_ma_df = add_moving_averages(
            df_high.copy(), 
            int(best_ma['Short_Window']), 
            int(best_ma['Long_Window'])
        )
        best_ma_df = generate_ma_signals(best_ma_df)
        
        print("Best MA Crossover parameters:")
        print(best_ma)
        
        ma_trades = generate_trade_list(best_ma_df, 'MA')
        ma_trades.to_csv('ma_trades.csv', index=False)
        print("MA Crossover trades saved to 'ma_trades.csv'")

        strategy_dfs['MA'] = best_ma_df
        
        # Save frequency optimization results if available
        if 'freq_results' in locals() and not freq_results.empty:
            freq_results.to_csv('ma_frequency_optimization.csv', index=False)
            print("Frequency optimization results saved to 'ma_frequency_optimization.csv'")
            
        strategies['MA'] = best_ma.to_dict()
        all_results_list.append(ma_results)
    else:
        print("No MA Crossover strategies met the criteria.")

    print("Running RSI Strategy...")
    rsi_results = optimize_rsi_parameters(
        df_high,
        range(10, 21, 2),
        range(65, 81, 5),
        range(20, 36, 5),
        min_trades_per_day=min_trades_per_day,
        max_trades_per_day=max_trades_per_day,
        min_total_return=min_total_return,
        min_profit_per_trade=min_profit_per_trade
    )
    if not rsi_results.empty:
        best_rsi = rsi_results.loc[rsi_results['Total_Return'].idxmax()]
        print("Best RSI parameters:")
        print(best_rsi)
        strategies['RSI'] = best_rsi.to_dict()
        all_results_list.append(rsi_results)

        print("Generating trade list for best RSI strategy...")
        best_rsi_df = calculate_rsi(df_high.copy(), int(best_rsi['RSI_Window']))
        best_rsi_df = generate_rsi_signals(best_rsi_df,
                                           int(best_rsi['Overbought']),
                                           int(best_rsi['Oversold']))
        rsi_trades = generate_trade_list(best_rsi_df, 'RSI')
        rsi_trades.to_csv('rsi_trades.csv', index=False)
        print("RSI trades saved to 'rsi_trades.csv'")

        strategy_dfs['RSI'] = best_rsi_df
    else:
        print("No RSI strategies met the criteria.")

    print("Running Bollinger Bands Strategy...")
    bb_param_grid = [{'window': w, 'num_std': s}
                     for w in range(10, 31, 5) for s in [1.5, 2, 2.5]]
    bb_results = optimize_hft_parameters(
        df_low,
        'BB',
        param_grid=bb_param_grid,
        min_trades_per_day=min_trades_per_day,
        max_trades_per_day=max_trades_per_day,
        min_total_return=min_total_return,
        min_profit_per_trade=min_profit_per_trade
    )
    if not bb_results.empty:
        best_bb = bb_results.loc[bb_results['Total_Return'].idxmax()]
        print("Best Bollinger Bands parameters:")
        print(best_bb)
        strategies['Bollinger Bands'] = best_bb.to_dict()
        all_results_list.append(bb_results)

        print("Generating trade list for best Bollinger Bands strategy...")
        best_bb_df = calculate_bollinger_bands(
            df_low.copy(),
            window=int(best_bb['window']),
            num_std=best_bb['num_std']
        )
        best_bb_df = generate_bollinger_band_signals(best_bb_df)
        bb_trades = generate_trade_list(best_bb_df, 'BB')
        bb_trades.to_csv('bb_trades.csv', index=False)
        print("Bollinger Bands trades saved to 'bb_trades.csv'")

        strategy_dfs['Bollinger Bands'] = best_bb_df
    else:
        print("No Bollinger Bands strategies met the criteria.")

    print("Running MACD Strategy...")
    macd_param_grid = [{'fast': f, 'slow': s, 'signal': sig}
                       for f in [6, 12, 18]
                       for s in [20, 26, 32]
                       for sig in [7, 9, 11]]
    macd_results = optimize_hft_parameters(
        df_low,
        'MACD',
        param_grid=macd_param_grid,
        min_trades_per_day=min_trades_per_day,
        max_trades_per_day=max_trades_per_day,
        min_total_return=min_total_return,
        min_profit_per_trade=min_profit_per_trade
    )
    if not macd_results.empty:
        best_macd = macd_results.loc[macd_results['Total_Return'].idxmax()]
        print("Best MACD parameters:")
        print(best_macd)
        strategies['MACD'] = best_macd.to_dict()
        all_results_list.append(macd_results)

        print("Generating trade list for best MACD strategy...")
        best_macd_df = calculate_macd(
            df_low.copy(),
            fast=int(best_macd['fast']),
            slow=int(best_macd['slow']),
            signal=int(best_macd['signal'])
        )
        best_macd_df = generate_macd_signals(best_macd_df)
        macd_trades = generate_trade_list(best_macd_df, 'MACD')
        macd_trades.to_csv('macd_trades.csv', index=False)
        print("MACD trades saved to 'macd_trades.csv'")

        strategy_dfs['MACD'] = best_macd_df
    else:
        print("No MACD strategies met the criteria.")

    print("Running RAMM Strategy...")
    ramm_results = optimize_ramm_parameters(
        df_high,
        max_iterations=max_iterations,
        min_trades_per_day=min_trades_per_day,
        max_trades_per_day=max_trades_per_day,
        min_total_return=min_total_return,
        min_profit_per_trade=min_profit_per_trade
    )
    if not ramm_results.empty:
        best_ramm = ramm_results.loc[ramm_results['Total_Return'].idxmax()]
        print("Best RAMM parameters:")
        print(best_ramm)
        strategies['RAMM'] = best_ramm.to_dict()
        all_results_list.append(ramm_results)

        print("Generating trade list for best RAMM strategy...")
        best_ramm_df = calculate_ramm_signals(
            df_high.copy(),
            ma_short=int(best_ramm['MA_Short']),
            ma_long=int(best_ramm['MA_Long']),
            rsi_period=int(best_ramm['RSI_Period']),
            rsi_ob=int(best_ramm['RSI_Overbought']),
            rsi_os=int(best_ramm['RSI_Oversold']),
            regime_lookback=int(best_ramm['Regime_Lookback'])
        )
        ramm_trades = generate_trade_list(best_ramm_df, 'RAMM')
        ramm_trades.to_csv('ramm_trades.csv', index=False)
        print("RAMM trades saved to 'ramm_trades.csv'")

        strategy_dfs['RAMM'] = best_ramm_df
    else:
        print("No RAMM strategies met the criteria.")

    print("Running Adaptive VWMA Strategy...")
    adaptive_vwma_results = optimize_adaptive_vwma_parameters(
        df_high,
        min_trades_per_day=min_trades_per_day,
        max_trades_per_day=max_trades_per_day,
        min_total_return=min_total_return,
        min_profit_per_trade=min_profit_per_trade
    )
    if not adaptive_vwma_results.empty:
        best_adaptive_vwma = adaptive_vwma_results.loc[adaptive_vwma_results['Total_Return'].idxmax()]
        print("Best Adaptive VWMA parameters:")
        print(best_adaptive_vwma)
        strategies['Adaptive_VWMA'] = best_adaptive_vwma.to_dict()
        all_results_list.append(adaptive_vwma_results)

        print("Generating trade list for best Adaptive VWMA strategy...")
        best_adaptive_vwma_df = calculate_adaptive_vwma(
            df_high.copy(),
            base_window=int(best_adaptive_vwma['Base_Window'])
        )
        best_adaptive_vwma_df = generate_adaptive_vwma_signals(
            best_adaptive_vwma_df,
            vol_scale=best_adaptive_vwma['Volume_Scale']
        )
        adaptive_vwma_trades = generate_trade_list(best_adaptive_vwma_df, 'Adaptive_VWMA')
        adaptive_vwma_trades.to_csv('adaptive_vwma_trades.csv', index=False)
        print("Adaptive VWMA trades saved to 'adaptive_vwma_trades.csv'")

        strategy_dfs['Adaptive_VWMA'] = best_adaptive_vwma_df
    else:
        print("No Adaptive VWMA strategies met the criteria.")

    if strategies:
        print("Preparing detailed strategy results...")
        print_strategy_results(strategies)

        print("\nStrategy Comparison:")
        comparison_df = pd.DataFrame.from_dict(strategies, orient='index')
        print(comparison_df)

        # Safely extract and compare total returns
        total_returns = {name: result.get('Total_Return', 0)
                         for name, result in strategies.items()}

        if total_returns:
            best_strategy = max(total_returns.items(), key=lambda x: x[1])[0]
            print(f"\nBest overall strategy: {best_strategy}")
            print(f"Best strategy Total Return: {total_returns[best_strategy]:.2f}%")

            # Save the best strategy parameters to a JSON file
            def convert_types(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                else:
                    return obj

            best_strategy_params_converted = {
                key: convert_types(value) for key, value in strategies[best_strategy].items()
            }

            # NEW: Determine the last non-zero signal for the best strategy
            df_best = strategy_dfs.get(best_strategy, None)
            if df_best is not None:
                # Handle naming for signal column
                if best_strategy == "Bollinger Bands":
                    signal_col = "BB_Signal"
                else:
                    signal_col = f"{best_strategy}_Signal".replace(" ", "_")

                last_signals = df_best[df_best[signal_col] != 0]
                if not last_signals.empty:
                    last_row = last_signals.iloc[-1]
                    last_signal_value = last_row[signal_col]
                    best_strategy_params_converted["Last_Signal_Timestamp"] = int(last_row["timestamp"])
                    if last_signal_value == 1:
                        best_strategy_params_converted["Last_Signal_Action"] = "GO LONG"
                    else:
                        best_strategy_params_converted["Last_Signal_Action"] = "GO SHORT"
                else:
                    best_strategy_params_converted["Last_Signal_Timestamp"] = None
                    best_strategy_params_converted["Last_Signal_Action"] = None

            # Also store the last trade's timestamp and price from the final row of df
            if not df.empty:
                best_strategy_params_converted["Last_Trade_Timestamp"] = int(df['timestamp'].iloc[-1])
                best_strategy_params_converted["Last_Trade_Price"] = float(df['price'].iloc[-1])
            else:
                best_strategy_params_converted["Last_Trade_Timestamp"] = None
                best_strategy_params_converted["Last_Trade_Price"] = None

            # Ensure do_live_trades = false
            best_strategy_params_converted["do_live_trades"] = False

            try:
                with open('best_strategy.json', 'w') as f:
                    json.dump(best_strategy_params_converted, f, indent=4)
                print("\nBest strategy parameters saved to 'best_strategy.json'")
            except Exception as e:
                print("An error occurred while writing the best strategy parameters to 'best_strategy.json':")
                traceback.print_exc()

            # ADDED: Write out a full trade list for the best strategy to best_strategy_trades.json
            #        This includes both long and short trades, as updated in generate_trade_list.
            if df_best is not None:
                try:
                    best_strategy_trades = generate_trade_list(df_best, best_strategy)
                    trades_records = best_strategy_trades.to_dict(orient='records')
                    with open('best_strategy_trades.json', 'w') as f:
                        json.dump(trades_records, f, indent=4)
                    print("\nAll trades for best strategy saved to 'best_strategy_trades.json'")
                except Exception as e:
                    print("An error occurred while writing best_strategy_trades.json:")
                    traceback.print_exc()

            print("\nAll strategy returns:")
            for strategy, t_ret in total_returns.items():
                print(f"{strategy}: {t_ret:.2f}%")
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
