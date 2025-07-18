# src/tdr_core/auto_trade_monitor.py
"""
Real-time trading monitor that connects WebSocket updates to the auto trader.
This is the CRITICAL missing piece that makes auto trading actually automatic.
"""

import threading
import time
import logging
from datetime import datetime, timedelta
import pandas as pd

class AutoTradeMonitor:
    """
    Monitors live price updates and triggers auto trader signal checks.
    """
    def __init__(self, auto_trader, data_manager, check_interval=60):
        """
        Initialize the auto trade monitor.
        
        Args:
            auto_trader: The AdaptiveMultiStrategy instance
            data_manager: The DataManager instance
            check_interval: How often to check for signals (seconds)
        """
        self.auto_trader = auto_trader
        self.data_manager = data_manager
        self.check_interval = check_interval
        self.logger = logging.getLogger(self.__class__.__name__)
        self.running = False
        self.monitor_thread = None
        self.last_signal_check = datetime.utcnow()
        self.last_price = None
        
        # Register for trade updates
        self.data_manager.add_trade_observer(self.on_new_trade)
        
    def on_new_trade(self, symbol, price, timestamp):
        """Called whenever a new trade comes in via WebSocket."""
        self.last_price = price
        
        # Update the auto trader's view of current price
        if hasattr(self.auto_trader, 'last_price'):
            self.auto_trader.last_price = price
            
    def start(self):
        """Start the monitoring thread."""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Auto trade monitor started")
        
    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Auto trade monitor stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop that periodically checks for trading signals."""
        while self.running:
            try:
                now = datetime.utcnow()
                
                # Check if it's time to evaluate signals
                if (now - self.last_signal_check).total_seconds() >= self.check_interval:
                    self._check_signals()
                    self.last_signal_check = now
                    
                # Sleep briefly to avoid busy waiting
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                time.sleep(5)  # Back off on errors
                
    def _check_signals(self):
        """Check for trading signals and execute trades if needed."""
        if not self.last_price:
            return
            
        try:
            # Get fresh data
            symbol = 'btcusd'
            df = self.data_manager.get_price_dataframe(symbol)
            
            if df is None or df.empty:
                self.logger.warning("No price data available for signal check")
                return
                
            # Get the latest timestamp from the data
            latest_time = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index[-1])
            
            # Check if auto trader should make a trade
            # This replicates the logic from check_for_signals
            self.logger.info(f"Checking signals at price ${self.last_price:.2f}")
            
            # The auto trader needs to evaluate its strategy
            if hasattr(self.auto_trader, 'check_for_signals'):
                # Build a signal dict similar to what the strategy expects
                signal = {
                    'timestamp': latest_time,
                    'price': self.last_price,
                    'action': None  # Will be determined by strategy
                }
                
                # Let the auto trader check its signals
                self.auto_trader.check_for_signals(signal, self.last_price, latest_time)
            else:
                self.logger.error("Auto trader missing check_for_signals method")
                
        except Exception as e:
            self.logger.error(f"Error checking signals: {e}")

    def force_signal_check(self):
        """Force an immediate signal check (useful for testing)."""
        self.logger.info("Forcing immediate signal check")
        self._check_signals()
        self.last_signal_check = datetime.utcnow()