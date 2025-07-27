#!/usr/bin/env python3
"""
Early Warning Monitor - Runs parallel to main system, READ-ONLY.
Provides advance notice of likely MA crossovers without any trading.

SAFETY FEATURES:
1. Completely separate from trading system
2. Read-only - cannot place trades
3. Only logs warnings
4. Can be disabled instantly
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from typing import Dict, Tuple, Optional

class EarlyWarningMonitor:
    """
    Monitor for potential MA crossovers using higher frequency data.
    This is READ-ONLY and cannot affect trading.
    """
    
    def __init__(self, data_manager, logger=None):
        self.data_manager = data_manager
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = True
        
        # Configuration - using same MA periods but on 5-min bars
        self.timeframe = "5m"
        self.ma_short_period = 6    # 6 * 5min = 30 minutes
        self.ma_long_period = 34    # 34 * 5min = 170 minutes
        
        # Tracking
        self.last_warning_time = None
        self.warning_threshold = 0.5  # Warn when MAs within 0.5%
        self.last_5min_signal = 0
        self.consecutive_signals = 0
        
        # Safety limits
        self.max_warnings_per_hour = 4  # Prevent spam
        self.warnings_this_hour = []
        
        self.logger.info("Early Warning Monitor initialized (READ-ONLY)")
        
    def check_5min_mas(self) -> Optional[Dict]:
        """
        Calculate MAs on 5-minute data and check for potential crossover.
        Returns None if disabled or on error.
        """
        if not self.enabled:
            return None
            
        try:
            # Get 5-minute candles (need at least 34 for long MA)
            candles = self.data_manager.get_recent_candles(
                symbol="btcusd",
                timeframe="5m",
                limit=50
            )
            
            if len(candles) < self.ma_long_period:
                return None
                
            # Calculate MAs
            closes = pd.Series([c['close'] for c in candles])
            ma_short = closes.rolling(window=self.ma_short_period).mean().iloc[-1]
            ma_long = closes.rolling(window=self.ma_long_period).mean().iloc[-1]
            
            # Calculate proximity
            ma_diff = ma_short - ma_long
            proximity = abs(ma_diff) / ma_long * 100
            
            # Determine signal
            signal = 1 if ma_short > ma_long else -1
            
            return {
                'timestamp': datetime.now(),
                'ma_short': ma_short,
                'ma_long': ma_long,
                'proximity': proximity,
                'signal': signal,
                'ma_diff': ma_diff
            }
            
        except Exception as e:
            self.logger.error(f"Error in check_5min_mas: {e}")
            return None
    
    def check_warning_conditions(self, data_5min: Dict, hourly_position: int) -> Optional[str]:
        """
        Determine if we should issue a warning.
        Returns warning message or None.
        """
        proximity = data_5min['proximity']
        signal = data_5min['signal']
        
        # Track consecutive signals
        if signal == self.last_5min_signal:
            self.consecutive_signals += 1
        else:
            self.consecutive_signals = 1
            self.last_5min_signal = signal
        
        # Check rate limiting
        now = datetime.now()
        self.warnings_this_hour = [w for w in self.warnings_this_hour 
                                   if now - w < timedelta(hours=1)]
        
        if len(self.warnings_this_hour) >= self.max_warnings_per_hour:
            return None
            
        # Warning conditions
        warning = None
        
        # 1. Approaching crossover
        if proximity <= self.warning_threshold and signal != hourly_position:
            if self.consecutive_signals >= 2:  # Need 2 consecutive 5-min candles
                warning = (f"⚠️ EARLY WARNING: 5-min MAs approaching crossover! "
                          f"Proximity: {proximity:.3f}% | "
                          f"5min Signal: {'LONG' if signal == 1 else 'SHORT'} | "
                          f"Current Position: {'LONG' if hourly_position == 1 else 'SHORT'}")
        
        # 2. Crossover detected on 5-min
        elif proximity <= 0.3 and signal != hourly_position:
            if self.consecutive_signals >= 3:  # Need 3 consecutive for crossover
                warning = (f"🚨 EARLY WARNING: 5-min MA CROSSOVER DETECTED! "
                          f"Signal: {'LONG' if signal == 1 else 'SHORT'} | "
                          f"Proximity: {proximity:.3f}% | "
                          f"Confirmed for {self.consecutive_signals * 5} minutes")
        
        # Record warning if issued
        if warning:
            self.warnings_this_hour.append(now)
            self.last_warning_time = now
            
        return warning
    
    def monitor_step(self, hourly_position: int) -> None:
        """
        Single monitoring step. Called by main loop.
        
        Args:
            hourly_position: Current position from hourly system (1 or -1)
        """
        if not self.enabled:
            return
            
        # Get 5-minute MA data
        data_5min = self.check_5min_mas()
        if not data_5min:
            return
            
        # Check for warning conditions
        warning = self.check_warning_conditions(data_5min, hourly_position)
        
        # Log warning if any
        if warning:
            self.logger.warning(warning)
            
        # Always log debug info for analysis
        self.logger.debug(
            f"Early Warning Check: 5min MA{self.ma_short_period}={data_5min['ma_short']:.0f} "
            f"MA{self.ma_long_period}={data_5min['ma_long']:.0f} "
            f"Prox={data_5min['proximity']:.3f}% "
            f"Signal={data_5min['signal']} "
            f"Consecutive={self.consecutive_signals}"
        )
    
    def get_status(self) -> Dict:
        """Get current monitor status."""
        return {
            'enabled': self.enabled,
            'last_warning': self.last_warning_time.isoformat() if self.last_warning_time else None,
            'warnings_this_hour': len(self.warnings_this_hour),
            'consecutive_signals': self.consecutive_signals,
            'current_5min_signal': self.last_5min_signal
        }
    
    def disable(self) -> None:
        """Disable the monitor."""
        self.enabled = False
        self.logger.info("Early Warning Monitor DISABLED")
        
    def enable(self) -> None:
        """Enable the monitor."""
        self.enabled = True
        self.logger.info("Early Warning Monitor ENABLED")


class EarlyWarningLogger:
    """
    Separate logger for early warning analysis.
    Logs to separate file for post-analysis.
    """
    
    def __init__(self, log_file="logs/early_warning_analysis.log"):
        self.logger = logging.getLogger("EarlyWarning")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def log_comparison(self, hourly_data: Dict, five_min_data: Dict):
        """Log comparison between hourly and 5-min signals."""
        self.logger.info(
            f"COMPARISON | "
            f"Hourly: Signal={hourly_data.get('signal', 'N/A')} "
            f"Prox={hourly_data.get('proximity', 'N/A'):.3f}% | "
            f"5-Min: Signal={five_min_data.get('signal', 'N/A')} "
            f"Prox={five_min_data.get('proximity', 'N/A'):.3f}% | "
            f"Difference: {abs(hourly_data.get('proximity', 0) - five_min_data.get('proximity', 0)):.3f}%"
        )


# Example integration (DO NOT ADD TO PRODUCTION YET)
"""
# In strategies.py, add to run_strategy method:

# Initialize early warning (once)
if not hasattr(self, 'early_warning'):
    self.early_warning = EarlyWarningMonitor(self.data_manager, self.logger)

# In the main loop (every 30 seconds)
if hasattr(self, 'early_warning'):
    self.early_warning.monitor_step(self.position)
"""