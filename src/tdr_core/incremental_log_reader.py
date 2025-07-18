# src/tdr_core/incremental_log_reader.py
"""
Incremental log reader that efficiently reads new lines from btcusd.log
without re-reading the entire file.
"""

import os
import time
import threading
import logging
from datetime import datetime

class IncrementalLogReader:
    """
    Reads new lines from a log file as they are appended.
    """
    def __init__(self, log_file, data_manager, check_interval=5):
        """
        Initialize the incremental log reader.
        
        Args:
            log_file: Path to the log file (e.g., btcusd.log)
            data_manager: DataManager instance to update with new trades
            check_interval: How often to check for new lines (seconds)
        """
        self.log_file = log_file
        self.data_manager = data_manager
        self.check_interval = check_interval
        self.logger = logging.getLogger(self.__class__.__name__)
        self.running = False
        self.reader_thread = None
        self.last_position = 0
        self.lines_processed = 0
        
        # Get initial file position (end of file)
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                f.seek(0, 2)  # Seek to end
                self.last_position = f.tell()
                self.logger.info(f"Starting from position {self.last_position} in {self.log_file}")
        
    def start(self):
        """Start the incremental reader thread."""
        if self.running:
            return
            
        self.running = True
        self.reader_thread = threading.Thread(target=self._reader_loop)
        self.reader_thread.daemon = True
        self.reader_thread.start()
        self.logger.info("Incremental log reader started")
        
    def stop(self):
        """Stop the reader thread."""
        self.running = False
        if self.reader_thread:
            self.reader_thread.join()
        self.logger.info(f"Incremental log reader stopped. Processed {self.lines_processed} new lines")
        
    def _reader_loop(self):
        """Main loop that checks for new lines."""
        while self.running:
            try:
                self._check_for_new_lines()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in reader loop: {e}")
                time.sleep(10)  # Back off on errors
                
    def _check_for_new_lines(self):
        """Check for and process new lines in the log file."""
        if not os.path.exists(self.log_file):
            return
            
        with open(self.log_file, 'r') as f:
            # Seek to last position
            f.seek(self.last_position)
            
            new_lines = 0
            for line in f:
                line = line.strip()
                if line:
                    self._process_line(line)
                    new_lines += 1
                    
            # Update position
            self.last_position = f.tell()
            
            if new_lines > 0:
                self.lines_processed += new_lines
                self.logger.debug(f"Processed {new_lines} new lines from log")
                
    def _process_line(self, line):
        """Process a single log line and add to data manager."""
        try:
            # Expected format: timestamp,price[,trade_type]
            parts = line.split(',')
            if len(parts) >= 2:
                timestamp = int(parts[0])
                price = float(parts[1])
                trade_type = parts[2] if len(parts) > 2 else "Live Trade"
                
                # Add to data manager (which will update DataFrame and notify observers)
                self.data_manager.add_trade('btcusd', price, timestamp, trade_type)
                
                # Also update the main DataFrame if it exists
                if hasattr(self.data_manager, 'data') and 'btcusd' in self.data_manager.data:
                    df = self.data_manager.data['btcusd']
                    dt = datetime.fromtimestamp(timestamp)
                    
                    # Add new row to DataFrame
                    new_row = {
                        'timestamp': timestamp,
                        'price': price,
                        'close': price,
                        'open': price,
                        'high': price,
                        'low': price,
                        'volume': 0.0
                    }
                    
                    # Append to DataFrame (this is what was missing!)
                    df.loc[dt] = new_row
                    
        except Exception as e:
            self.logger.error(f"Error processing line '{line}': {e}")
            
    def force_check(self):
        """Force an immediate check for new lines."""
        self._check_for_new_lines()