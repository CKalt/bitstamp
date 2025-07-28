#!/usr/bin/env python3
"""
Enhanced MA Strategy with detailed comparison logging
This wraps the existing MACrossoverStrategy to add comprehensive logging
for live vs backtest comparison
"""
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tdr_core.strategies import MACrossoverStrategy
from src.tdr_core.backtest_comparison_logger import BacktestComparisonLogger


class EnhancedMAStrategy(MACrossoverStrategy):
    """
    Enhanced MA strategy that logs everything needed for backtest comparison
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize parent class
        super().__init__(*args, **kwargs)
        
        # Initialize comparison logger
        self.comparison_logger = BacktestComparisonLogger()
        
        # Track last hourly bar to detect new hours
        self.last_hour_processed = None
        
        # Override to ensure we log initial state
        self._log_initial_state()
        
    def _log_initial_state(self):
        """Log initial configuration and state"""
        self.logger.info(f"🔍 Enhanced MA Strategy initialized with comparison logging")
        self.logger.info(f"   MA periods: {self.short_window}/{self.long_window}")
        self.logger.info(f"   Live trading: {self.live_trading}")
        
    def evaluate_signal_and_place_order(self, signal_type, signal_value):
        """Override to add detailed logging before evaluation"""
        
        # Get current time and check if we're in a new hour
        current_time = datetime.now()
        current_hour = current_time.replace(minute=0, second=0, microsecond=0)
        
        # Check if we have hourly data
        if hasattr(self.data_manager, 'df') and len(self.data_manager.df) > 0:
            df = self.data_manager.df
            
            # Get latest hourly bar
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
            # Resample to hourly if needed
            hourly_df = df.resample('1H').agg({
                'price': ['first', 'max', 'min', 'last'],
                'amount': 'sum'
            }).dropna()
            
            if len(hourly_df) > 0:
                hourly_df.columns = ['open', 'high', 'low', 'close', 'volume']
                
                # Calculate MAs on hourly data
                hourly_df['MA_short'] = hourly_df['close'].rolling(window=self.short_window).mean()
                hourly_df['MA_long'] = hourly_df['close'].rolling(window=self.long_window).mean()
                
                # Get latest complete hourly bar
                if len(hourly_df) > 0:
                    latest_bar = hourly_df.iloc[-1]
                    bar_time = hourly_df.index[-1]
                    
                    # Log hourly bar if it's new
                    if self.last_hour_processed is None or bar_time > self.last_hour_processed:
                        self.comparison_logger.log_hourly_bar(
                            bar_time=bar_time,
                            open_price=latest_bar['open'],
                            high=latest_bar['high'],
                            low=latest_bar['low'],
                            close=latest_bar['close'],
                            volume=latest_bar['volume'],
                            ma_short=self.short_window,
                            ma_long=self.long_window,
                            ma_short_value=latest_bar['MA_short'],
                            ma_long_value=latest_bar['MA_long']
                        )
                        self.last_hour_processed = bar_time
                    
                    # Log signal evaluation
                    current_price = self.data_manager.get_latest_price()
                    previous_signal = getattr(self, 'last_signal', 0)
                    
                    # Determine current signal
                    if latest_bar['MA_short'] > latest_bar['MA_long']:
                        current_signal = 1  # Long
                    elif latest_bar['MA_short'] < latest_bar['MA_long']:
                        current_signal = -1  # Short
                    else:
                        current_signal = 0  # Neutral (shouldn't happen)
                    
                    # Check if we will trade
                    will_trade = False
                    reason = ""
                    
                    if previous_signal != current_signal:
                        if not self.live_trading:
                            will_trade = False
                            reason = "Live trading disabled"
                        elif self.trade_count_today >= self.max_trades_per_day:
                            will_trade = False
                            reason = f"Max trades per day ({self.max_trades_per_day}) reached"
                        elif self.recent_trade_count >= self.max_trades_per_hour:
                            will_trade = False
                            reason = f"Max trades per hour ({self.max_trades_per_hour}) reached"
                        else:
                            # Check minimum time between trades
                            if hasattr(self, 'last_trade_time') and self.last_trade_time:
                                time_since_last = (current_time - self.last_trade_time).total_seconds() / 60
                                if time_since_last < self.min_time_between_trades:
                                    will_trade = False
                                    reason = f"Too soon since last trade ({time_since_last:.1f} < {self.min_time_between_trades} min)"
                                else:
                                    will_trade = True
                                    reason = "All conditions met"
                            else:
                                will_trade = True
                                reason = "First trade"
                    else:
                        reason = "No signal change"
                    
                    self.comparison_logger.log_signal_evaluation(
                        timestamp=current_time,
                        current_price=current_price,
                        ma_short_value=latest_bar['MA_short'],
                        ma_long_value=latest_bar['MA_long'],
                        previous_signal=previous_signal,
                        current_signal=current_signal,
                        will_trade=will_trade,
                        reason=reason
                    )
                    
                    # Store current signal
                    self.last_signal = current_signal
        
        # Call parent implementation
        return super().evaluate_signal_and_place_order(signal_type, signal_value)
        
    def _execute_trade(self, side, amount, price=None):
        """Override to log trade execution details"""
        
        # Get pre-trade state
        position_before = self.position
        
        # Get current MA values
        ma_short_value = 0
        ma_long_value = 0
        hourly_bar_time = None
        
        if hasattr(self.data_manager, 'df') and len(self.data_manager.df) > 0:
            df = self.data_manager.df
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
            hourly_df = df.resample('1H').agg({
                'price': 'last'
            }).dropna()
            
            if len(hourly_df) > 0:
                hourly_df['MA_short'] = hourly_df['price'].rolling(window=self.short_window).mean()
                hourly_df['MA_long'] = hourly_df['price'].rolling(window=self.long_window).mean()
                
                latest = hourly_df.iloc[-1]
                ma_short_value = latest['MA_short']
                ma_long_value = latest['MA_long']
                hourly_bar_time = hourly_df.index[-1]
        
        # Execute trade
        result = super()._execute_trade(side, amount, price)
        
        # Log trade if successful
        if result and 'error' not in result:
            self.comparison_logger.log_trade_decision(
                timestamp=datetime.now(),
                trade_type=side.upper(),
                price=result.get('price', price or self.data_manager.get_latest_price()),
                amount=amount,
                position_before=position_before,
                position_after=self.position,
                ma_short=ma_short_value,
                ma_long=ma_long_value,
                hourly_bar_time=hourly_bar_time or datetime.now(),
                exact_trigger_time=datetime.now()
            )
            
            self.logger.info(f"🔍 Trade logged for comparison: {side.upper()} @ ${result.get('price', 0):.2f}")
        
        return result
        
    def check_and_log_no_trade_hours(self):
        """Called periodically to log hours where no trade occurred"""
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        if hasattr(self, 'last_logged_hour') and self.last_logged_hour == current_hour:
            return  # Already logged this hour
            
        if not hasattr(self, 'trades_executed') or len(self.trades_executed) == 0:
            # No trades this hour
            if hasattr(self.data_manager, 'df') and len(self.data_manager.df) > 0:
                df = self.data_manager.df
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                
                hourly_df = df.resample('1H').last()
                if len(hourly_df) > 0:
                    hourly_df['MA_short'] = hourly_df['price'].rolling(window=self.short_window).mean()
                    hourly_df['MA_long'] = hourly_df['price'].rolling(window=self.long_window).mean()
                    
                    latest = hourly_df.iloc[-1]
                    
                    reason = "No signal change"
                    if latest['MA_short'] == latest['MA_long']:
                        reason = "MAs equal"
                    elif not self.live_trading:
                        reason = "Live trading disabled"
                        
                    self.comparison_logger.log_no_trade_hour(
                        bar_time=current_hour,
                        reason=reason,
                        ma_short=latest['MA_short'],
                        ma_long=latest['MA_long']
                    )
        
        self.last_logged_hour = current_hour