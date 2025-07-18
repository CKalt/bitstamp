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
    if not os.path.exists(metadata_file_path):
        create_metadata_file(file_path, metadata_file_path)

    data = []
    with open(metadata_file_path, 'r') as file:
        metadata = json.load(file)

    total_lines = metadata['total_lines']
    parsing_progress['total_lines'] = total_lines
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
    # Update every 100k lines for consistent progress
    status_interval = 100000
    next_status_line = start_line + status_interval

    with open(file_path, 'r') as file:
        for i, line in enumerate(file, 1):
            if i < start_line:
                continue

            # Show status updates periodically
            if i >= next_status_line:
                lines_processed = i - start_line + 1
                parsing_progress['processed_lines'] = lines_processed
                parsing_progress['percent'] = int((lines_processed / lines_to_process) * 100)
                parsing_progress['status'] = f'Processing line {lines_processed:,} of {total_lines:,}'
                print(f"Status: Reading historical data - {lines_processed:,} lines processed ({parsing_progress['percent']}%) - Last date: {last_date}")
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
    
    # Log completion details
    logger.info(f"Finished reading log file. Last date: {last_date}")
    logger.info(f"Total entries skipped: {skipped_count}")
    logger.info(f"Total entries processed: {processed_count}")
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
