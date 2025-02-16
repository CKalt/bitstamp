###############################################################################
# File Path: src/utils/analysis.py
###############################################################################
# Full File Path: src/utils/analysis.py
#
# CHANGES:
#   1) We now accept a new optional parameter bar_frequencies (list of freq).
#   2) We iterate over each frequency in bar_frequencies, run the existing
#      strategy pipeline, and track which frequency yields the best overall
#      returns. The best frequency is then stored in best_strategy.json
#   3) We keep the existing logic and comments, ensuring minimal changes.
###############################################################################

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


def run_trading_system(df, high_frequency='1H', low_frequency='15T', max_iterations=50,
                       config=None, only_ma=False, bar_frequencies=None):
    """
    Start of the run_trading_system function.

    NOTE: We have added `bar_frequencies=None`. If provided as a list of frequencies,
          we will loop over them and find the best frequency overall. Then we store
          that best frequency in best_strategy.json.

    We keep all existing logic and comments, ensuring minimal changes.
    """
    print("Starting run_trading_system function...")
    df = ensure_datetime_index(df)
    print(f"DataFrame shape after ensuring datetime index: {df.shape}")

    if config is None:
        config = {}

    constraints = config.get("strategy_constraints", {})
    min_trades_per_day = constraints.get("min_trades_per_day", 1)
    max_trades_per_day = constraints.get("max_trades_per_day", 4)
    min_total_return = constraints.get("min_total_return", 0.0)
    min_profit_per_trade = constraints.get("min_profit_per_trade", 0.0)

    # If bar_frequencies is None or empty, we just run the existing logic once
    # using the user-supplied "high_frequency" (and "low_frequency" for other strats).
    # If bar_frequencies is not empty, we run a small loop. We want to find which
    # bar frequency yields the best overall final strategy return, store that,
    # and return it.
    if not bar_frequencies:
        bar_frequencies = [high_frequency]

    best_overall_return = -999999
    best_overall_freq = bar_frequencies[0]
    best_strategies_snapshot = {}
    best_dfs_snapshot = {}
    best_comparison_df = pd.DataFrame()
    all_results_combined = []

    for freq in bar_frequencies:
        print(f"\n--- Testing bar frequency: {freq} ---")
        # Resample data to freq
        df_copy = df.copy()
        df_copy['volume'] = df_copy['price'] * df_copy['amount']
        df_resampled = df_copy.resample(freq).agg({
            'price': 'last',
            'amount': 'sum',
            'volume': 'sum'
        }).dropna()
        df_resampled['timestamp'] = df_resampled.index.view('int64') // 10**9

        # We'll run the existing strategy pipeline on df_resampled for the "higher timeframe"
        # and if we do "low_frequency" for HFT stuff, let's just reuse freq or low_frequency.
        # We'll do the same logic as the original single-run approach, but we store partial results
        # in local variables.

        # The code below replicates the existing logic from the original run_trading_system,
        # except we replace "df_high" with "df_resampled" for the "MA/RSI/others" and keep the lower
        # timeframe approach for "df_low". We can also do "df_low = df_copy.resample(low_frequency)..."
        # if needed.
        # For minimal changes, we do so inline:

        # Resample data to lower timeframe for certain strategies
        df_low = df_copy.resample(low_frequency).agg({
            'price': 'last',
            'amount': 'sum',
            'volume': 'sum'
        }).dropna()
        df_low['timestamp'] = df_low.index.view('int64') // 10**9

        strategies_local = {}
        all_results_list_local = []
        strategy_dfs_local = {}

        print("Running MA Crossover Strategy...")
        ma_results = optimize_ma_parameters(
            df_resampled,
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

            # Frequency optimization inside the single freq might be redundant,
            # but let's keep it for now.
            freq_results = optimize_ma_frequency(df, {
                'Short_Window': int(best_ma['Short_Window']),
                'Long_Window': int(best_ma['Long_Window'])
            })
            if not freq_results.empty:
                best_freq_result = freq_results.loc[freq_results['Total_Return'].idxmax()]
                print("\nBest frequency results:")
                print(best_freq_result)
                best_ma = best_freq_result

            best_ma_df = add_moving_averages(
                df_resampled.copy(),
                int(best_ma['Short_Window']),
                int(best_ma['Long_Window'])
            )
            best_ma_df = generate_ma_signals(best_ma_df)
            ma_trades = generate_trade_list(best_ma_df, 'MA')
            ma_trades.to_csv(f'ma_trades_{freq}.csv', index=False)  # or keep same name

            strategy_dfs_local['MA'] = best_ma_df
            strategies_local['MA'] = best_ma.to_dict()
            all_results_list_local.append(ma_results)
        else:
            print("No MA Crossover strategies met the criteria.")

        if not only_ma:
            print("Running RSI Strategy...")
            rsi_results = optimize_rsi_parameters(
                df_resampled,
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
                strategies_local['RSI'] = best_rsi.to_dict()
                all_results_list_local.append(rsi_results)

                best_rsi_df = calculate_rsi(df_resampled.copy(), int(best_rsi['RSI_Window']))
                best_rsi_df = generate_rsi_signals(
                    best_rsi_df,
                    int(best_rsi['Overbought']),
                    int(best_rsi['Oversold'])
                )
                rsi_trades = generate_trade_list(best_rsi_df, 'RSI')
                rsi_trades.to_csv(f'rsi_trades_{freq}.csv', index=False)
                strategy_dfs_local['RSI'] = best_rsi_df
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
                strategies_local['Bollinger Bands'] = best_bb.to_dict()
                all_results_list_local.append(bb_results)

                best_bb_df = calculate_bollinger_bands(
                    df_low.copy(),
                    window=int(best_bb['window']),
                    num_std=best_bb['num_std']
                )
                best_bb_df = generate_bollinger_band_signals(best_bb_df)
                bb_trades = generate_trade_list(best_bb_df, 'BB')
                bb_trades.to_csv(f'bb_trades_{freq}.csv', index=False)
                strategy_dfs_local['Bollinger Bands'] = best_bb_df
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
                strategies_local['MACD'] = best_macd.to_dict()
                all_results_list_local.append(macd_results)

                best_macd_df = calculate_macd(
                    df_low.copy(),
                    fast=int(best_macd['fast']),
                    slow=int(best_macd['slow']),
                    signal=int(best_macd['signal'])
                )
                best_macd_df = generate_macd_signals(best_macd_df)
                macd_trades = generate_trade_list(best_macd_df, 'MACD')
                macd_trades.to_csv(f'macd_trades_{freq}.csv', index=False)
                strategy_dfs_local['MACD'] = best_macd_df
            else:
                print("No MACD strategies met the criteria.")

            print("Running RAMM Strategy...")
            ramm_results = optimize_ramm_parameters(
                df_resampled,
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
                strategies_local['RAMM'] = best_ramm.to_dict()
                all_results_list_local.append(ramm_results)

                best_ramm_df = calculate_ramm_signals(
                    df_resampled.copy(),
                    ma_short=int(best_ramm['MA_Short']),
                    ma_long=int(best_ramm['MA_Long']),
                    rsi_period=int(best_ramm['RSI_Period']),
                    rsi_ob=int(best_ramm['RSI_Overbought']),
                    rsi_os=int(best_ramm['RSI_Oversold']),
                    regime_lookback=int(best_ramm['Regime_Lookback'])
                )
                ramm_trades = generate_trade_list(best_ramm_df, 'RAMM')
                ramm_trades.to_csv(f'ramm_trades_{freq}.csv', index=False)
                strategy_dfs_local['RAMM'] = best_ramm_df
            else:
                print("No RAMM strategies met the criteria.")

            print("Running Adaptive VWMA Strategy...")
            adaptive_vwma_results = optimize_adaptive_vwma_parameters(
                df_resampled,
                min_trades_per_day=min_trades_per_day,
                max_trades_per_day=max_trades_per_day,
                min_total_return=min_total_return,
                min_profit_per_trade=min_profit_per_trade
            )
            if not adaptive_vwma_results.empty:
                best_adaptive_vwma = adaptive_vwma_results.loc[adaptive_vwma_results['Total_Return'].idxmax()]
                print("Best Adaptive VWMA parameters:")
                print(best_adaptive_vwma)
                strategies_local['Adaptive_VWMA'] = best_adaptive_vwma.to_dict()
                all_results_list_local.append(adaptive_vwma_results)

                best_adaptive_vwma_df = calculate_adaptive_vwma(
                    df_resampled.copy(),
                    base_window=int(best_adaptive_vwma['Base_Window'])
                )
                best_adaptive_vwma_df = generate_adaptive_vwma_signals(
                    best_adaptive_vwma_df,
                    vol_scale=best_adaptive_vwma['Volume_Scale']
                )
                adaptive_vwma_trades = generate_trade_list(best_adaptive_vwma_df, 'Adaptive_VWMA')
                adaptive_vwma_trades.to_csv(f'adaptive_vwma_trades_{freq}.csv', index=False)
                strategy_dfs_local['Adaptive_VWMA'] = best_adaptive_vwma_df
            else:
                print("No Adaptive VWMA strategies met the criteria.")

        # Summaries for this freq
        if strategies_local:
            # Summaries
            # We replicate the existing "print_strategy_results"
            # but we only keep it local for each freq
            total_returns_local = {name: result.get('Total_Return', 0)
                                   for name, result in strategies_local.items()}
            local_best_strat = max(total_returns_local.items(), key=lambda x: x[1])[0] if total_returns_local else None
            local_best_return = total_returns_local.get(local_best_strat, 0) if local_best_strat else 0
            if local_best_return > best_overall_return:
                best_overall_return = local_best_return
                best_overall_freq = freq
                best_strategies_snapshot = strategies_local
                best_dfs_snapshot = strategy_dfs_local
                # We'll build the comparison dataframe
                best_comparison_df = pd.DataFrame.from_dict(strategies_local, orient='index')

            if all_results_list_local:
                combined_local = pd.concat(all_results_list_local, ignore_index=True)
                all_results_combined.append(combined_local)

    # After testing all bar frequencies, we store the best frequency, best strategy, etc.
    if len(all_results_combined) > 0:
        all_results = pd.concat(all_results_combined, ignore_index=True)
    else:
        all_results = pd.DataFrame()

    # We replicate the logic from the end of the original run_trading_system
    if best_strategies_snapshot:
        print("\nFinal Best Strategies (across frequencies):")
        print_strategy_results(best_strategies_snapshot)

        comparison_df = pd.DataFrame.from_dict(best_strategies_snapshot, orient='index')
        print("\nStrategy Comparison (best freq):")
        print(comparison_df)

        total_returns = {name: result.get('Total_Return', 0)
                         for name, result in best_strategies_snapshot.items()}
        best_strategy = max(total_returns.items(), key=lambda x: x[1])[0] if total_returns else None
        if best_strategy:
            print(f"\nBest overall strategy across frequencies: {best_strategy}")
            print(f"Best bar frequency: {best_overall_freq}")
            print(f"Best strategy Total Return: {total_returns[best_strategy]:.2f}%")

            # Build the JSON
            def convert_types(value):
                if isinstance(value, (np.integer, np.int64)):
                    return int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    return float(value)
                elif isinstance(value, np.ndarray):
                    return value.tolist()
                elif isinstance(value, pd.Timestamp):
                    return value.isoformat()
                else:
                    return value

            best_strategy_params_converted = {
                key: convert_types(value) for key, value in best_strategies_snapshot[best_strategy].items()
            }
            # Add the bar size
            best_strategy_params_converted["Bar_Size"] = best_overall_freq

            # Determine last non-zero signal
            df_best = best_dfs_snapshot.get(best_strategy, None)
            if df_best is not None and not df_best.empty:
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

            if not df.empty:
                best_strategy_params_converted["Last_Trade_Timestamp"] = int(df['timestamp'].iloc[-1])
                best_strategy_params_converted["Last_Trade_Price"] = float(df['price'].iloc[-1])
            else:
                best_strategy_params_converted["Last_Trade_Timestamp"] = None
                best_strategy_params_converted["Last_Trade_Price"] = None

            best_strategy_params_converted["do_live_trades"] = False

            try:
                with open('best_strategy.json', 'w') as f:
                    json.dump(best_strategy_params_converted, f, indent=4)
                print("\nBest strategy parameters saved to 'best_strategy.json'.")
            except Exception as e:
                print("An error occurred while writing to best_strategy.json:")
                traceback.print_exc()

            # Also write out a full trade list
            if df_best is not None and not df_best.empty:
                from backtesting.backtester import generate_trade_list
                best_strategy_trades = generate_trade_list(df_best, best_strategy)
                try:
                    def convert_timestamps_for_json(obj):
                        if isinstance(obj, pd.Timestamp):
                            return obj.isoformat()
                        return str(obj)

                    trades_records = best_strategy_trades.to_dict(orient='records')
                    with open('best_strategy_trades.json', 'w') as f:
                        json.dump(trades_records, f, indent=4, default=convert_timestamps_for_json)
                    print("\nAll trades for best strategy saved to 'best_strategy_trades.json'")
                except Exception as e:
                    print("An error occurred while writing best_strategy_trades.json:")
                    traceback.print_exc()
        else:
            print("\nNo strategies found across frequencies.")
    else:
        print("No strategies met the criteria across any bar frequency.")
        comparison_df = pd.DataFrame()

    # Save the final combined results if we have any
    if not all_results.empty:
        all_results.to_csv('optimization_results.csv', index=False)
        print("\nAll optimization results saved to 'optimization_results.csv'.")
    else:
        print("\nNo optimization results to save.")
        comparison_df = pd.DataFrame()

    return all_results, comparison_df
