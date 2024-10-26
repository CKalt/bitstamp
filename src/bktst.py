# src/bktst.py

import argparse
import os
import sys
from datetime import datetime, timedelta

# Ensure that the 'src' directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data.loader import create_metadata_file, parse_log_file
from utils.analysis import analyze_data, run_trading_system

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
