#!/usr/bin/env python3
"""
Build index file for btcusd.log to enable fast random access by date
Creates btcusd.log.idx with daily summaries
"""

import json
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

class LogIndexBuilder:
    """Build and maintain index for large log files"""
    
    def __init__(self, log_file):
        """Initialize with log file path"""
        self.log_file = log_file
        self.index_file = f"{log_file}.idx"
        self.index_data = {
            'version': 1,
            'created': None,
            'file_size': 0,
            'total_lines': 0,
            'first_timestamp': None,
            'last_timestamp': None,
            'days': {}
        }
        
    def build_index(self, force_rebuild=False):
        """Build or update the index file"""
        # Check if index exists and is up to date
        if not force_rebuild and self.is_index_current():
            print(f"Index file {self.index_file} is up to date")
            return True
        
        print(f"Building index for {self.log_file}...")
        print("This may take a few minutes for large files...")
        
        # Get file size
        file_size = os.path.getsize(self.log_file)
        self.index_data['file_size'] = file_size
        
        # Process the log file
        daily_data = defaultdict(lambda: {
            'first_line': None,
            'last_line': None,
            'first_offset': None,
            'last_offset': None,
            'first_timestamp': None,
            'last_timestamp': None,
            'trade_count': 0
        })
        
        line_num = 0
        byte_offset = 0
        last_progress = 0
        
        with open(self.log_file, 'r') as f:
            while True:
                # Record byte offset before reading line
                current_offset = f.tell()
                line = f.readline()
                
                if not line:
                    break
                
                line_num += 1
                
                # Show progress every 100k lines
                if line_num % 100000 == 0:
                    progress = (current_offset / file_size) * 100
                    print(f"  Progress: {progress:.1f}% - Processed {line_num:,} lines...")
                
                # Parse JSON line
                try:
                    data = json.loads(line.strip())
                    
                    # Skip non-trade events
                    if data.get('event') != 'trade':
                        continue
                    
                    # Extract timestamp
                    trade_data = data.get('data', {})
                    timestamp = trade_data.get('timestamp')
                    
                    if not timestamp:
                        continue
                    
                    # Convert timestamp to datetime
                    ts = int(timestamp)
                    dt = datetime.fromtimestamp(ts)
                    date_key = dt.strftime('%Y-%m-%d')
                    
                    # Update daily data
                    day_data = daily_data[date_key]
                    
                    if day_data['first_line'] is None:
                        day_data['first_line'] = line_num
                        day_data['first_offset'] = current_offset
                        day_data['first_timestamp'] = ts
                    
                    day_data['last_line'] = line_num
                    day_data['last_offset'] = current_offset
                    day_data['last_timestamp'] = ts
                    day_data['trade_count'] += 1
                    
                    # Update global first/last
                    if self.index_data['first_timestamp'] is None:
                        self.index_data['first_timestamp'] = ts
                    self.index_data['last_timestamp'] = ts
                    
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Skip malformed lines
                    continue
        
        # Store results
        self.index_data['total_lines'] = line_num
        self.index_data['created'] = datetime.now().isoformat()
        self.index_data['days'] = dict(daily_data)
        
        # Save index file
        print(f"\nWriting index to {self.index_file}...")
        with open(self.index_file, 'w') as f:
            json.dump(self.index_data, f, indent=2, sort_keys=True)
        
        # Print summary
        print(f"\nIndex created successfully!")
        print(f"  Total lines: {line_num:,}")
        print(f"  Total days: {len(daily_data)}")
        print(f"  Date range: {min(daily_data.keys())} to {max(daily_data.keys())}")
        
        return True
    
    def is_index_current(self):
        """Check if index exists and is newer than log file"""
        if not os.path.exists(self.index_file):
            return False
        
        # Check if log file has been modified since index was created
        log_mtime = os.path.getmtime(self.log_file)
        idx_mtime = os.path.getmtime(self.index_file)
        
        if log_mtime > idx_mtime:
            print(f"Log file has been modified since index was created")
            return False
        
        # Load index and check file size
        try:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            current_size = os.path.getsize(self.log_file)
            indexed_size = index.get('file_size', 0)
            
            if current_size != indexed_size:
                print(f"Log file size changed: {indexed_size} -> {current_size}")
                return False
            
            return True
            
        except (json.JSONDecodeError, KeyError):
            return False
    
    def load_index(self):
        """Load existing index file"""
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"Index file {self.index_file} not found")
        
        with open(self.index_file, 'r') as f:
            return json.load(f)
    
    def get_lines_for_date_range(self, start_date, end_date):
        """Get line numbers for a date range using the index"""
        # Load or build index
        if not os.path.exists(self.index_file):
            self.build_index()
        
        index = self.load_index()
        days = index['days']
        
        # Find relevant days
        relevant_days = []
        for date_str in sorted(days.keys()):
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            if start_date <= date <= end_date:
                relevant_days.append(date_str)
        
        if not relevant_days:
            return None, None
        
        # Get first line of first day and last line of last day
        first_day = days[relevant_days[0]]
        last_day = days[relevant_days[-1]]
        
        return first_day['first_line'], last_day['last_line']
    
    def get_offsets_for_date_range(self, start_date, end_date):
        """Get byte offsets for a date range using the index"""
        # Load or build index
        if not os.path.exists(self.index_file):
            self.build_index()
        
        index = self.load_index()
        days = index['days']
        
        # Convert dates to strings
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Find relevant days
        relevant_days = []
        for date_str in sorted(days.keys()):
            if start_str <= date_str <= end_str:
                relevant_days.append(date_str)
        
        if not relevant_days:
            return None, None
        
        # Get first offset of first day and last offset of last day
        first_day = days[relevant_days[0]]
        last_day = days[relevant_days[-1]]
        
        return first_day['first_offset'], last_day['last_offset']


def main():
    """Command line interface for index builder"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build index for btcusd.log')
    parser.add_argument('--file', default='btcusd.log', help='Log file to index')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild even if index exists')
    parser.add_argument('--info', action='store_true', help='Show index info')
    
    args = parser.parse_args()
    
    builder = LogIndexBuilder(args.file)
    
    if args.info:
        # Show index info
        try:
            index = builder.load_index()
            print(f"Index file: {builder.index_file}")
            print(f"Created: {index['created']}")
            print(f"File size: {index['file_size']:,} bytes")
            print(f"Total lines: {index['total_lines']:,}")
            print(f"Days indexed: {len(index['days'])}")
            
            if index['days']:
                days = sorted(index['days'].keys())
                print(f"Date range: {days[0]} to {days[-1]}")
                
                # Show some daily stats
                print("\nSample daily statistics:")
                for day in days[:3]:  # First 3 days
                    day_data = index['days'][day]
                    print(f"  {day}: {day_data['trade_count']:,} trades")
                print("  ...")
                for day in days[-3:]:  # Last 3 days
                    day_data = index['days'][day]
                    print(f"  {day}: {day_data['trade_count']:,} trades")
        except FileNotFoundError:
            print(f"No index file found at {builder.index_file}")
    else:
        # Build index
        builder.build_index(force_rebuild=args.rebuild)


if __name__ == '__main__':
    main()