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
        start_line = get_start_line_from_metadata(
            metadata_file_path, start_date)
        print(
            f"Starting from line {start_line} based on start date {start_date}")

    last_date = None
    skipped_count = start_line - 1
    processed_count = 0
    end_reached = False
    
    # Calculate progress interval based on lines we'll actually process
    lines_to_process = total_lines - start_line + 1
    # Show 10 progress updates based on lines to process, not total lines
    progress_interval = max(lines_to_process // 10, 1)
    next_progress_line = start_line + progress_interval
    last_progress_printed = -1

    with open(file_path, 'r') as file:
        for i, line in enumerate(file, 1):
            if i < start_line:
                continue

            # Check if we've reached a progress milestone
            if i >= next_progress_line:
                lines_processed = i - start_line + 1
                progress = min(lines_processed / lines_to_process * 100, 100.0)
                # Only print if progress has actually changed
                if progress != last_progress_printed:
                    print(f"Progress: {progress:.1f}% - Last date: {last_date}")
                    last_progress_printed = progress
                next_progress_line += progress_interval

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

    # Final progress update only if we haven't already printed 100%
    if last_progress_printed < 100.0:
        print(f"Progress: 100.0% - Last date: {last_date}")
    
    # Log completion details without overwriting progress status
    logger.info(f"Finished processing log file. Last date: {last_date}")
    logger.info(f"Total entries skipped: {skipped_count}")
    logger.info(f"Total entries processed: {processed_count}")
    if end_reached:
        logger.info(f"Reached end date: {end_date}")
    df = pd.DataFrame(data)

    # Optimize data types
    df['price'] = pd.to_numeric(df['price'], downcast='float')
    df['amount'] = pd.to_numeric(df['amount'], downcast='float')
    df['type'] = df['type'].astype('int8')

    return df
