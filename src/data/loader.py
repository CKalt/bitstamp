# src/data/loader.py

import json
import os
import pandas as pd
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                    if str(date) not in metadata:
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

# Global variable to track progress
parsing_progress = {
    'total_lines': 0,
    'processed_lines': 0,
    'status': 'Not started',
    'percent': 0
}

def parse_log_file(file_path, start_date=None, end_date=None, progress_callback=None):
    global parsing_progress
    metadata_file_path = f"{file_path}.metadata"
    
    # Only create metadata if it doesn't exist
    # Don't refresh just because file is newer - it's ALWAYS being updated
    if not os.path.exists(metadata_file_path):
        print("Creating metadata file for first time...")
        create_metadata_file(file_path, metadata_file_path)

    data = []
    with open(metadata_file_path, 'r') as file:
        metadata = json.load(file)

    total_lines = metadata['total_lines']
    parsing_progress['status'] = 'Starting'
    print(f"Total lines in log file: {total_lines}")

    start_line = 1
    if start_date:
        start_line = get_start_line_from_metadata(
            metadata_file_path, start_date)
        print(
            f"Starting from line {start_line} based on start date {start_date}")

    last_date = None
    skipped_count = start_line - 1
    processed_count = 0
    end_reached = False
    
    # Calculate interval for status updates
    lines_to_process = total_lines - start_line + 1
    
    # ALWAYS use 1M line intervals for console output
    # Don't trust the metadata estimate - it's often wrong for live-updating files
    status_interval = 1000000
    next_status_line = start_line + status_interval
    
    print(f"Total lines in metadata (may be outdated): {total_lines:,}")
    print(f"Starting from line: {start_line:,}")
    print(f"Minimum lines to process: {lines_to_process:,}")
    print(f"Progress interval: Every {status_interval:,} lines")
    print(f"Note: File is continuously updated, actual lines will be more")
    
    # If we're processing very few lines from a large file, warn the user
    if lines_to_process < 10000 and total_lines > 1000000:
        print(f"⚠️  WARNING: Only processing {lines_to_process:,} lines from a {total_lines:,} line file!")
        print(f"   This suggests the start date {start_date} is very recent.")
        print(f"   Consider using an earlier start date to process more historical data.")
    
    # Initialize for actual line counting
    actual_lines = 0
    
    # Update initial progress
    parsing_progress['total_lines'] = lines_to_process  # This will be updated if file is larger
    parsing_progress['processed_lines'] = 0
    parsing_progress['percent'] = 0
    parsing_progress['status'] = f'Starting to process lines'

    with open(file_path, 'r') as file:
        for i, line in enumerate(file, 1):
            if i < start_line:
                continue

            # Update progress counter
            current_line = i - start_line + 1
            actual_lines = i  # Track actual line number
            
            # If we've exceeded expected lines, just keep processing
            # This is normal for a continuously updated log file
            
            # Update parsing progress continuously for API access
            if lines_to_process > 0:
                parsing_progress['processed_lines'] = current_line
                parsing_progress['percent'] = min(100, int((current_line / lines_to_process) * 100))
            
            # Show status updates periodically to console
            if i >= next_status_line:
                parsing_progress['status'] = f'Processing line {current_line:,} of {lines_to_process:,}'
                # Show actual line number from file and percentage
                print(f"Status: Reading historical data - Line {i:,} ({current_line:,}/{lines_to_process:,} processed, {parsing_progress['percent']}%) - Last date: {last_date}")
                next_status_line += status_interval

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

    # Final status update
    parsing_progress['status'] = 'Creating DataFrame'
    parsing_progress['percent'] = 100
    print(f"Status: Finished reading {processed_count:,} trades - Last date: {last_date}")
    
    # Check if file was larger than metadata indicated
    if actual_lines > total_lines:
        print(f"⚠️  Note: File has {actual_lines:,} lines (metadata showed {total_lines:,})")
        print(f"   Consider refreshing metadata by deleting {metadata_file_path}")
    
    # Log completion details
    logger.info(f"Finished reading log file. Last date: {last_date}")
    logger.info(f"Total entries skipped: {skipped_count}")
    logger.info(f"Total entries processed: {processed_count}")
    logger.info(f"Actual lines in file: {actual_lines}")
    if end_reached:
        logger.info(f"Reached end date: {end_date}")
    
    # Creating DataFrame is a significant operation - signal this
    parsing_progress['status'] = f'Creating DataFrame from {processed_count:,} trades'
    print(f"Status: Creating DataFrame from {processed_count:,} trades...")
    df = pd.DataFrame(data)
    
    # Signal completion after DataFrame is created
    parsing_progress['status'] = 'Complete'
    print("Status: DataFrame created successfully")

    # Optimize data types
    df['price'] = pd.to_numeric(df['price'], downcast='float')
    df['amount'] = pd.to_numeric(df['amount'], downcast='float')
    df['type'] = df['type'].astype('int8')

    return df
