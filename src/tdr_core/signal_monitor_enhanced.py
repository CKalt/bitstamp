"""
Enhanced Signal Monitoring System

This module provides comprehensive signal monitoring with detailed logging
to ensure no signals are missed and all evaluations are tracked.
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SignalMonitor:
    """Monitor and log all signal evaluations with detailed tracking."""
    
    def __init__(self, log_dir: str = "logs", alert_threshold_seconds: int = 120):
        """
        Initialize the signal monitor.
        
        Args:
            log_dir: Directory for signal monitoring logs
            alert_threshold_seconds: Alert if no signal check for this many seconds
        """
        self.log_dir = log_dir
        self.alert_threshold = alert_threshold_seconds
        self.last_evaluation_time = None
        self.evaluation_count = 0
        self.missed_signals = []
        self.evaluation_history = []
        
        # Create signal monitoring log file
        os.makedirs(log_dir, exist_ok=True)
        self.signal_log_file = os.path.join(log_dir, f"signal_monitor_{datetime.now().strftime('%Y%m%d')}.json")
        self.summary_log_file = os.path.join(log_dir, "signal_monitor_summary.json")
        
        # Load existing evaluation history
        self._load_history()
        
    def _load_history(self):
        """Load existing evaluation history from file."""
        if os.path.exists(self.signal_log_file):
            try:
                with open(self.signal_log_file, 'r') as f:
                    data = json.load(f)
                    self.evaluation_history = data.get('evaluations', [])
                    self.evaluation_count = len(self.evaluation_history)
                    if self.evaluation_history:
                        last_eval = self.evaluation_history[-1]
                        self.last_evaluation_time = datetime.fromisoformat(last_eval['timestamp'])
            except Exception as e:
                logger.error(f"Failed to load signal history: {e}")
    
    def log_evaluation(self, 
                      signal: int, 
                      position: int,
                      price: float,
                      ma_short: float,
                      ma_long: float,
                      will_trade: bool,
                      reason: str,
                      additional_data: Optional[Dict] = None) -> Dict:
        """
        Log a signal evaluation with comprehensive details.
        
        Returns:
            Dict containing the evaluation record
        """
        self.evaluation_count += 1
        self.last_evaluation_time = datetime.now()
        
        # Create evaluation record (convert numpy types to Python types for JSON)
        evaluation = {
            'timestamp': self.last_evaluation_time.isoformat(),
            'evaluation_number': self.evaluation_count,
            'signal': int(signal),  # Convert numpy int64 to Python int
            'position': int(position),  # Convert numpy int64 to Python int
            'price': round(float(price), 2),  # Ensure float
            'ma_short': round(float(ma_short), 2),  # Ensure float
            'ma_long': round(float(ma_long), 2),  # Ensure float
            'ma_diff': round(float(ma_short - ma_long), 2),
            'ma_diff_pct': round(float((ma_short - ma_long) / ma_long * 100), 4),
            'will_trade': bool(will_trade),  # Ensure bool
            'reason': str(reason),  # Ensure string
            'time_since_last': self._time_since_last_evaluation()
        }
        
        # Add any additional data
        if additional_data:
            evaluation['additional'] = additional_data
        
        # Check for missed evaluations
        if self._check_missed_evaluation():
            evaluation['alert'] = 'MISSED_EVALUATION_DETECTED'
            self.missed_signals.append(evaluation)
        
        # Add to history
        self.evaluation_history.append(evaluation)
        
        # Keep only last 1000 evaluations in memory
        if len(self.evaluation_history) > 1000:
            self.evaluation_history = self.evaluation_history[-1000:]
        
        # Save to file
        self._save_evaluation(evaluation)
        
        # Log to standard logger with clear formatting
        status_icon = "✅" if will_trade else "⏸️"
        logger.info(f"{status_icon} SIGNAL_CHECK #{self.evaluation_count}: "
                   f"Signal={signal} Pos={position} Price=${price:.0f} "
                   f"MA{int(additional_data.get('short_window', 0))}={ma_short:.0f} "
                   f"MA{int(additional_data.get('long_window', 0))}={ma_long:.0f} "
                   f"Diff={ma_short - ma_long:.0f} ({(ma_short - ma_long) / ma_long * 100:.2f}%) "
                   f"Action={'TRADE' if will_trade else 'HOLD'} Reason={reason}")
        
        return evaluation
    
    def _time_since_last_evaluation(self) -> Optional[float]:
        """Calculate seconds since last evaluation."""
        if not self.evaluation_history or len(self.evaluation_history) < 2:
            return None
        
        try:
            current = datetime.fromisoformat(self.evaluation_history[-1]['timestamp'])
            previous = datetime.fromisoformat(self.evaluation_history[-2]['timestamp'])
            return (current - previous).total_seconds()
        except:
            return None
    
    def _check_missed_evaluation(self) -> bool:
        """Check if we might have missed an evaluation."""
        time_since_last = self._time_since_last_evaluation()
        if time_since_last and time_since_last > self.alert_threshold:
            logger.warning(f"⚠️  ALERT: {time_since_last:.0f} seconds since last evaluation "
                         f"(threshold: {self.alert_threshold}s)")
            return True
        return False
    
    def _save_evaluation(self, evaluation: Dict):
        """Save evaluation to file."""
        try:
            # Append to daily log
            with open(self.signal_log_file, 'w') as f:
                json.dump({
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'total_evaluations': self.evaluation_count,
                    'missed_count': len(self.missed_signals),
                    'evaluations': self.evaluation_history
                }, f, indent=2)
            
            # Update summary
            self._update_summary()
            
        except Exception as e:
            logger.error(f"Failed to save evaluation: {e}")
    
    def _update_summary(self):
        """Update the summary statistics file."""
        try:
            # Calculate statistics
            total_evaluations = self.evaluation_count
            trade_signals = sum(1 for e in self.evaluation_history if e.get('will_trade', False))
            missed_count = len(self.missed_signals)
            
            # Calculate average time between evaluations
            times_between = [e.get('time_since_last', 0) for e in self.evaluation_history 
                           if e.get('time_since_last') is not None]
            avg_time_between = sum(times_between) / len(times_between) if times_between else 0
            
            # Find longest gap
            max_gap = max(times_between) if times_between else 0
            
            summary = {
                'last_updated': datetime.now().isoformat(),
                'total_evaluations': total_evaluations,
                'trade_signals': trade_signals,
                'hold_signals': total_evaluations - trade_signals,
                'missed_evaluations': missed_count,
                'average_seconds_between': round(avg_time_between, 1),
                'max_gap_seconds': round(max_gap, 1),
                'monitoring_health': 'HEALTHY' if max_gap < self.alert_threshold * 2 else 'DEGRADED'
            }
            
            with open(self.summary_log_file, 'w') as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update summary: {e}")
    
    def get_recent_evaluations(self, count: int = 10) -> List[Dict]:
        """Get the most recent evaluations."""
        return self.evaluation_history[-count:] if self.evaluation_history else []
    
    def get_monitoring_status(self) -> Dict:
        """Get current monitoring status."""
        now = datetime.now()
        time_since_last = (now - self.last_evaluation_time).total_seconds() if self.last_evaluation_time else None
        
        status = {
            'status': 'HEALTHY' if time_since_last and time_since_last < self.alert_threshold else 'WARNING',
            'last_evaluation': self.last_evaluation_time.isoformat() if self.last_evaluation_time else None,
            'seconds_since_last': round(time_since_last, 1) if time_since_last else None,
            'total_evaluations': self.evaluation_count,
            'missed_signals': len(self.missed_signals),
            'alert_threshold': self.alert_threshold
        }
        
        if time_since_last and time_since_last > self.alert_threshold:
            status['alert'] = f'No evaluation for {time_since_last:.0f} seconds!'
            
        return status
    
    def check_health(self) -> Tuple[bool, str]:
        """
        Check if signal monitoring is healthy.
        
        Returns:
            (is_healthy, message)
        """
        status = self.get_monitoring_status()
        
        if status['status'] == 'HEALTHY':
            return True, f"Signal monitoring healthy. Last check {status['seconds_since_last']:.0f}s ago."
        else:
            return False, status.get('alert', 'Signal monitoring unhealthy')


def integrate_signal_monitor(strategy_instance):
    """
    Integrate signal monitoring into an existing strategy.
    
    This monkey-patches the strategy to add comprehensive signal logging.
    """
    # Create monitor instance
    monitor = SignalMonitor()
    strategy_instance._signal_monitor = monitor
    
    # Store original check_for_signals
    original_check = strategy_instance.check_for_signals
    
    def monitored_check_for_signals(latest_signal, current_price, signal_time):
        """Wrapped version with monitoring."""
        
        # Get MA values if available
        ma_short = ma_long = 0
        if hasattr(strategy_instance, 'df_ma') and not strategy_instance.df_ma.empty:
            ma_short = strategy_instance.df_ma.iloc[-1].get('Short_MA', 0)
            ma_long = strategy_instance.df_ma.iloc[-1].get('Long_MA', 0)
        
        # Determine if trade will happen
        will_trade = False
        reason = ""
        
        if latest_signal == 1 and strategy_instance.position <= 0:
            if strategy_instance.trade_count_today >= strategy_instance.max_trades_per_day:
                reason = f"Daily limit reached ({strategy_instance.trade_count_today}/{strategy_instance.max_trades_per_day})"
            else:
                will_trade = True
                reason = "BUY signal - flipping to LONG"
        elif latest_signal == -1 and strategy_instance.position >= 0:
            if strategy_instance.trade_count_today >= strategy_instance.max_trades_per_day:
                reason = f"Daily limit reached ({strategy_instance.trade_count_today}/{strategy_instance.max_trades_per_day})"
            else:
                will_trade = True
                reason = "SELL signal - flipping to SHORT"
        else:
            reason = f"Signal ({latest_signal}) matches position ({strategy_instance.position})"
        
        # Log the evaluation
        monitor.log_evaluation(
            signal=latest_signal,
            position=strategy_instance.position,
            price=current_price,
            ma_short=ma_short,
            ma_long=ma_long,
            will_trade=will_trade,
            reason=reason,
            additional_data={
                'short_window': strategy_instance.short_window,
                'long_window': strategy_instance.long_window,
                'trade_count_today': strategy_instance.trade_count_today,
                'signal_time': signal_time.isoformat() if hasattr(signal_time, 'isoformat') else str(signal_time)
            }
        )
        
        # Call original method
        return original_check(latest_signal, current_price, signal_time)
    
    # Replace method
    strategy_instance.check_for_signals = monitored_check_for_signals
    
    logger.info("✅ Signal monitoring integrated into strategy")
    return monitor