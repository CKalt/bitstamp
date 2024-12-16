# src/bktst.py
import argparse
import os
import sys
from datetime import datetime, timedelta
import pandas as pd

# Ensure that the 'src' directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from data.loader import create_metadata_file, parse_log_file
from utils.analysis import analyze_data, run_trading_system


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze Bitcoin trade log data.")
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
            raise ValueError("Cannot use trading-window-days without specifying start-window-days-back")
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
        print(f"{strategy}: Total Return = {strategy_comparison.loc[strategy, 'Total_Return']:.2f}%")

    if len(strategy_comparison) > 1:
        best_strategy = strategy_comparison['Total_Return'].idxmax()
        print(f"\nOverall best strategy: {best_strategy}")
        print(f"Best strategy Total Return: {strategy_comparison.loc[best_strategy, 'Total_Return']:.2f}%")


def main():
    """
    Main function to run backtesting for multiple strategies.
    """
    # Parse arguments
    args = parse_arguments()

    # Setup metadata and load log file
    file_path = 'btcusd.log'
    setup_metadata(file_path)
    start_date, end_date = determine_date_range(args)
    df = parse_log_file(file_path, start_date, end_date)

    print(f"Parsed {len(df)} trade events.")
    print(f"DataFrame shape after parsing: {df.shape}")
    print(f"DataFrame memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")

    # Analyze data
    print("Starting data analysis...")
    analyze_data(df)

    # Run trading system
    print("Running trading system...")
    try:
        optimization_results, strategy_comparison = run_trading_system(
            df,
            high_frequency=args.high_frequency,
            low_frequency=args.low_frequency,
            max_iterations=args.max_iterations
        )

        # Display all strategies in a table
        display_results_table(optimization_results)

        # Display best strategy summary
        display_best_strategy_summary(strategy_comparison)

        # Save results to CSV
        optimization_results.to_csv("all_strategy_results.csv", index=False)
        print("\nResults saved to 'all_strategy_results.csv'.")

    except Exception as e:
        print(f"An error occurred during trading system analysis: {str(e)}")
        print("Partial results may have been saved.")


if __name__ == "__main__":
    main()
