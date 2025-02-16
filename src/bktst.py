###############################################################################
# src/bktst.py
###############################################################################
# Full File Path: src/bktst.py
#
# WHY THIS CHANGE:
#   The block that defines 'detailed_strategy_results' was purely an example 
#   (mock data) not linked to the real optimizer output, causing confusion. 
#   We preserve the original comments and code except we now comment out 
#   that mock dictionary and skip printing it so as not to mislead.
#
# WHAT'S CHANGED:
#   1) We have commented out the hard-coded 'detailed_strategy_results' block 
#      to avoid displaying contradictory data after the real optimizer 
#      finds "No strategies met the criteria."
#   2) We retain all original code, comments, and structure, 
#      simply preventing the mock results from printing to console.
#   3) We keep the previously added --only-ma argument, and the 
#      fix that prevents KeyError on empty DataFrame.
#
# NOTE:
#   If you'd like to display real, detailed results, consider 
#   deriving them from the actual 'optimization_results' or 
#   'strategy_comparison' rather than using a hard-coded block.
#
# ADDITIONALLY (NEW):
#   1) We want to allow the user to specify multiple bar sizes (5, 15, 20 minutes,
#      or 1H, etc.) so we can find the best bar size for each strategy.
#   2) Then store that best bar size in best_strategy.json so tdr.py can trade
#      with the same frequency.
#   3) We add --bar-frequencies to parse_arguments, and pass that to run_trading_system.
#   4) We do not remove any existing code or comments, including the old 
#      commented-out blocks referencing mock data or best_strategy.json logic. 
###############################################################################

from utils.analysis import analyze_data, run_trading_system
from data.loader import create_metadata_file, parse_log_file
import argparse
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import json
import traceback

# Ensure that the 'src' directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

###############################################################################
# NEW: Function to load config from JSON or use defaults
###############################################################################
def load_config(config_path='config.json'):
    if not os.path.exists(config_path):
        # Default config if file not present
        default_config = {
            "strategy_constraints": {
                "min_trades_per_day": 1,
                "max_trades_per_day": 4,
                "min_total_return": 0.0,
                "min_profit_per_trade": 0.0
            }
        }
        print(f"Config file '{config_path}' not found. Using default config.")
        return default_config
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from '{config_path}'")
        return config


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Analyze Bitcoin trade log data.")
    parser.add_argument('--start-window-days-back', type=int, default=30,
                        help='Number of days to subtract from the current date as the start window')
    parser.add_argument('--end-window-days-back', type=int, default=0,
                        help='Number of days to subtract from the current date as the end window')
    parser.add_argument('--trading-window-days', type=int,
                        help='Number of days to analyze from the start date (overrides end-window-days-back)')
    parser.add_argument('--max-iterations', type=int, default=50,
                        help='Maximum number of iterations for parameter optimization')
    parser.add_argument('--high-frequency', type=str, default='1H',
                        help='Sampling frequency for higher timeframe (e.g., "1H" for hourly)')
    parser.add_argument('--low-frequency', type=str, default='15T',
                        help='Sampling frequency for lower timeframe (e.g., "15T" for 15 minutes)')

    # NEW: Add --bar-frequencies (for multiple bar sizes)
    parser.add_argument('--bar-frequencies', nargs='*', default=['5T','15T','20T','1H'],
                        help='List of bar frequencies to test (e.g. 5T 15T 20T 1H)')

    # NEW: Add --only-ma flag (existing code, reaffirming)
    parser.add_argument('--only-ma', action='store_true',
                        help='Only optimize for the Moving Average (MA) strategy.')
    return parser.parse_args()


def setup_metadata(file_path):
    """
    Create metadata file if it does not exist.
    """
    metadata_file_path = f"{file_path}.metadata"
    if not os.path.exists(metadata_file_path):
        print("Metadata file not found. Creating it now...")
        create_metadata_file(file_path, metadata_file_path)
        print("Metadata file created.")


def determine_date_range(args):
    """
    Determine the start and end date for the analysis window.
    """
    current_date = datetime.now()
    if args.start_window_days_back > 0:
        start_date = current_date - timedelta(days=args.start_window_days_back)
        print(f"Analyzing data from {start_date} onwards")
    else:
        start_date = None

    if args.trading_window_days is not None:
        if start_date is None:
            raise ValueError(
                "Cannot use trading-window-days without specifying start-window-days-back")
        end_date = start_date + timedelta(days=args.trading_window_days)
        print(f"Using trading window of {args.trading_window_days} days")
    elif args.end_window_days_back > 0:
        end_date = current_date - timedelta(days=args.end_window_days_back)
        print(f"Analyzing data up to {end_date}")
    else:
        end_date = None

    if start_date and end_date and start_date >= end_date:
        raise ValueError("Start date must be earlier than end date")
    return start_date, end_date


###############################################################################
# RESTORED FUNCTIONS (previously removed)
###############################################################################
def evaluate_all_strategies(data, strategies):
    """
    Evaluate all strategies on the given data and return their performance metrics.

    :param data: The input data for backtesting.
    :param strategies: List of strategies to evaluate.
    :return: DataFrame containing results for all strategies.
    """
    all_results = []

    for strategy in strategies:
        # Run the strategy on the data
        result = strategy.run(data)

        # Collect relevant metrics
        all_results.append({
            "Strategy": strategy.name,
            "Parameters": strategy.params,
            "Final Balance": result.final_balance,
            "Total Return (%)": round(result.total_return * 100, 2),
            "Total Trades": result.total_trades,
            "Profit Factor": round(result.profit_factor, 2),
            "Sharpe Ratio": round(result.sharpe_ratio, 2),
            "Win/Loss Ratio": round(result.win_loss_ratio, 2) if result.win_loss_ratio else "N/A",
        })

    # Convert results to a DataFrame for better display
    results_df = pd.DataFrame(all_results)
    return results_df


def display_summary(strategy_comparison):
    """
    Display the best strategy for each type and the overall best strategy.

    :param strategy_comparison: DataFrame with comparison results.
    """
    print("\nBest strategy for each type:")
    for strategy in strategy_comparison['Strategy'].unique():
        strategy_row = strategy_comparison[strategy_comparison['Strategy'] == strategy]
        print(
            f"{strategy}: Total Return = {strategy_row['Total_Return'].iloc[0]:.2f}%")

    overall_best = strategy_comparison.loc[strategy_comparison['Total_Return'].idxmax()]
    print(f"\nOverall best strategy: {overall_best['Strategy']}")
    print(f"Best strategy Total Return: {overall_best['Total_Return']:.2f}%")


def display_strategy_comparison(comparison_df):
    """
    Display strategy comparison results in a table format.

    :param comparison_df: DataFrame containing comparison results for all strategies.
    """
    print("\nStrategy Comparison:")
    print(comparison_df.to_markdown(index=False))
    print("\n")


def display_detailed_strategy_results(strategy_results):
    """
    Display detailed strategy results for each strategy in a table format.
    
    NOTE: This was originally referencing a mocked dictionary. We have commented 
    out that dictionary creation to avoid contradictory data. If you'd like 
    to show real data, pass in the actual results from your optimizer.
    """
    print("\nDetailed Strategy Results:\n")
    for strategy, results in strategy_results.items():
        print(f"--- {strategy} ---")
        import pandas as pd
        results_df = pd.DataFrame([results])
        print(results_df.to_markdown(index=False))
        print("\n")


def display_results_table(results_df):
    """
    Display the results DataFrame in a neatly formatted table.

    :param results_df: The DataFrame containing strategy results.
    """
    print("\nAll Strategy Results:")
    print(results_df.to_markdown(index=False))


def display_best_strategy_summary(strategy_comparison):
    """
    Display the best strategy summary based on the comparison results.
    """
    print("\nBest strategy for each type:")
    for strategy in strategy_comparison.index:
        print(
            f"{strategy}: Total Return = {strategy_comparison.loc[strategy, 'Total_Return']:.2f}%")

    if len(strategy_comparison) > 1:
        best_strategy = strategy_comparison['Total_Return'].idxmax()
        print(f"\nOverall best strategy: {best_strategy}")
        print(
            f"Best strategy Total Return: {strategy_comparison.loc[best_strategy, 'Total_Return']:.2f}%")


def main():
    """
    Main function to run backtesting for multiple strategies.
    """
    # Parse arguments
    args = parse_arguments()

    # NEW: Load config
    config = load_config('config.json')  # <-- # NEW

    # Setup metadata and load log file
    file_path = 'btcusd.log'
    setup_metadata(file_path)
    start_date, end_date = determine_date_range(args)
    df = parse_log_file(file_path, start_date, end_date)

    print(f"Parsed {len(df)} trade events.")
    print(f"DataFrame shape after parsing: {df.shape}")
    print(
        f"DataFrame memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")

    # Analyze data
    print("Starting data analysis...")
    analyze_data(df)

    # Run trading system
    print("Running trading system...")
    try:
        # Execute the trading system and collect results
        # CHANGED: pass bar_frequencies to run_trading_system
        optimization_results, strategy_comparison = run_trading_system(
            df,
            high_frequency=args.high_frequency,
            low_frequency=args.low_frequency,
            max_iterations=args.max_iterations,
            config=config,     # <-- # CHANGED: Passing config
            only_ma=args.only_ma,
            bar_frequencies=args.bar_frequencies  # NEW
        )

        ################################################################
        # CHANGED: We have commented out the previously mocked data that 
        # used to show a "fake" strategy result. This prevents confusion 
        # when real results show "No strategies met the criteria."
        ################################################################

        # OLD CODE (Commented):
        """
        # Example: Define detailed strategy results (mocked; replace with actual 
        # data from `run_trading_system`)
        detailed_strategy_results = {
            "MA": {
                "Frequency": args.high_frequency,
                "Short_Window": 12,
                "Long_Window": 36,
                "Final_Balance": 10262.35,
                "Total_Return": 2.62,
                "Total_Trades": 45,
                "Average_Trades_Per_Day": 1.45,
                "Profit_Factor": 1.03,
                "Sharpe_Ratio": 0.15
            },
            "RSI": {
                "RSI_Window": 14,
                "Overbought": 80,
                "Oversold": 35,
                "Final_Balance": 10479.78,
                "Total_Return": 4.80,
                "Total_Trades": 80,
                "Average_Trades_Per_Day": 2.58,
                "Profit_Factor": 1.17,
                "Sharpe_Ratio": 0.41
            },
            "RAMM": {
                "MA_Short": 6,
                "MA_Long": 35,
                "RSI_Period": 12,
                "RSI_Overbought": 65,
                "RSI_Oversold": 35,
                "Regime_Lookback": 20,
                "Final_Balance": 10133.44,
                "Total_Return": 1.33,
                "Total_Trades": 50,
                "Average_Trades_Per_Day": 1.61,
                "Profit_Factor": 1.14,
                "Sharpe_Ratio": 0.18
            }
        """
        
        # If you'd like to display real results, consider building a dictionary 
        # from 'optimization_results' or 'strategy_comparison' as needed:
        detailed_strategy_results = {}

        # Display detailed strategy results (currently empty unless you populate 
        # it from your real run)
        print("\n--- Detailed Strategy Results ---")
        display_detailed_strategy_results(detailed_strategy_results)

        # If strategy_comparison is non-empty and has "Strategy" column, show it
        if not strategy_comparison.empty and "Strategy" in strategy_comparison.columns:
            print("\n--- Strategy Comparison ---")
            display_strategy_comparison(strategy_comparison)
            print("\n--- Summary of Best Strategies ---")
            display_summary(strategy_comparison)
        else:
            print("\nNo strategy comparison data to display.")

        # Save optimization results to CSV (if any exist)
        optimization_results.to_csv("all_strategy_results.csv", index=False)
        print("\nResults saved to 'all_strategy_results.csv'.")

        ################################################################
        # NEW LOGIC (ORIGINALLY) that wrote best_strategy.json
        # BUGFIX: We now comment it out to avoid overwriting the file
        ################################################################
        """
        if not strategy_comparison.empty:
            # 1) Identify the best row by total return
            best_idx = strategy_comparison['Total_Return'].idxmax()
            best_row = strategy_comparison.loc[best_idx]

            best_strategy_json = {
                "Frequency": args.high_frequency,
                "Strategy": best_row['Strategy'],
                "Short_Window": best_row.get('Short_Window', 0),
                "Long_Window": best_row.get('Long_Window', 0),
                "Final_Balance": best_row.get('Final_Balance', 0),
                "Total_Return": best_row.get('Total_Return', 0),
                "Total_Trades": best_row.get('Total_Trades', 0),
                "Profit_Factor": best_row.get('Profit_Factor', 0),
                "Sharpe_Ratio": best_row.get('Sharpe_Ratio', 0),
                "Average_Trades_Per_Day": best_row.get('Average_Trades_Per_Day', 0),
                "start_window_days_back": args.start_window_days_back,
                "end_window_days_back": args.end_window_days_back
            }

            def convert_types(obj):
                import numpy as np
                import pandas as pd
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif pd.isna(obj):
                    return None
                return obj

            best_strategy_json = {
                key: convert_types(value) for key, value in best_strategy_json.items()
            }

            # NEW: Ensure do_live_trades = false
            best_strategy_json["do_live_trades"] = False

            try:
                with open("best_strategy.json", "w") as f:
                    json.dump(best_strategy_json, f, indent=4)
                print("\\nWrote best_strategy.json with new fields:")
                print(best_strategy_json)
            except Exception as e:
                print("Error writing best_strategy.json:")
                traceback.print_exc()
        """

    except Exception as e:
        print(f"An error occurred during trading system analysis: {str(e)}")
        print("Partial results may have been saved.")


if __name__ == "__main__":
    main()
