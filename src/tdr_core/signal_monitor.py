#!/usr/bin/env python3
"""
Signal Monitoring and Detection System
Provides real-time monitoring of trading signals and detection of missed opportunities
"""
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class SignalMonitor:
    """Monitor trading signals and detect missed opportunities"""
    
    def __init__(self, data_manager, strategy):
        self.data_manager = data_manager
        self.strategy = strategy
        self.signal_history = []
        self.last_check_time = datetime.now()
        self.missed_signals = []
        
    def get_current_signal_status(self) -> Dict:
        """Get comprehensive current signal status"""
        current_time = datetime.now()
        
        # Get current market data
        current_price = self.data_manager.get_current_price('btcusd')
        if not current_price:
            return {"error": "No current price available"}
            
        # Get current MA values
        df = self.data_manager.get_dataframe('btcusd')
        if df is None or df.empty:
            return {"error": "No data available"}
            
        # Get the latest MA values
        latest = df.iloc[-1]
        short_ma = latest.get(f'SMA_{self.strategy.short_window}', None)
        long_ma = latest.get(f'SMA_{self.strategy.long_window}', None)
        
        if short_ma is None or long_ma is None:
            return {"error": "MA values not calculated"}
            
        # Calculate signal strength and distance
        ma_diff = short_ma - long_ma
        ma_diff_pct = (ma_diff / long_ma) * 100
        
        # Determine current signal
        current_signal = 1 if short_ma > long_ma else -1
        signal_desc = "BUY (Long)" if current_signal == 1 else "SELL (Short)"
        
        # Check if signal matches position
        position_matches = (current_signal == self.strategy.position)
        
        # Calculate distance to signal flip
        distance_to_flip = abs(ma_diff)
        distance_to_flip_pct = abs(ma_diff_pct)
        
        # Estimate bars until signal (simplified)
        recent_ma_changes = []
        if len(df) > 10:
            for i in range(1, 6):
                if i < len(df):
                    prev_short = df.iloc[-i-1].get(f'SMA_{self.strategy.short_window}', 0)
                    curr_short = df.iloc[-i].get(f'SMA_{self.strategy.short_window}', 0)
                    recent_ma_changes.append(curr_short - prev_short)
                    
        avg_ma_change = sum(recent_ma_changes) / len(recent_ma_changes) if recent_ma_changes else 0
        bars_to_signal = int(distance_to_flip / abs(avg_ma_change)) if avg_ma_change != 0 else 999
        
        # Check signal confirmation status
        if hasattr(self.strategy, 'signal_confirmations'):
            confirmations = self.strategy.signal_confirmations
            bars_confirmed = self.strategy.bars_since_signal
            confirmation_required = self.strategy.signal_confirmation_bars
        else:
            confirmations = 0
            bars_confirmed = 0
            confirmation_required = 2
            
        # Time calculations
        time_since_last_check = (current_time - self.last_check_time).total_seconds()
        
        # Build comprehensive status
        status = {
            "timestamp": current_time.isoformat(),
            "current_price": current_price,
            "position": {
                "side": "LONG" if self.strategy.position == 1 else "SHORT" if self.strategy.position == -1 else "FLAT",
                "size": getattr(self.strategy, 'position_size', 0),
                "entry_price": getattr(self.strategy, 'last_trade_price', 0),
                "unrealized_pnl": self._calculate_pnl()
            },
            "moving_averages": {
                "short_ma": {
                    "period": self.strategy.short_window,
                    "value": short_ma,
                    "label": f"SMA_{self.strategy.short_window}"
                },
                "long_ma": {
                    "period": self.strategy.long_window,
                    "value": long_ma,
                    "label": f"SMA_{self.strategy.long_window}"
                },
                "difference": ma_diff,
                "difference_pct": ma_diff_pct
            },
            "signal": {
                "current": signal_desc,
                "raw_value": current_signal,
                "matches_position": position_matches,
                "distance_to_flip": distance_to_flip,
                "distance_to_flip_pct": distance_to_flip_pct,
                "estimated_bars_to_signal": bars_to_signal,
                "avg_ma_change_per_bar": avg_ma_change
            },
            "confirmation": {
                "bars_confirmed": bars_confirmed,
                "bars_required": confirmation_required,
                "is_confirmed": bars_confirmed >= confirmation_required,
                "confirmations": confirmations
            },
            "timing": {
                "last_check": self.last_check_time.isoformat(),
                "seconds_since_check": time_since_last_check,
                "bar_interval": getattr(self.strategy, 'bar_size', '5T')
            },
            "alerts": self._check_alerts(current_signal, position_matches, bars_to_signal)
        }
        
        # Update last check time
        self.last_check_time = current_time
        
        # Store in history
        self.signal_history.append({
            "timestamp": current_time,
            "signal": current_signal,
            "ma_diff_pct": ma_diff_pct,
            "position_matches": position_matches
        })
        
        # Keep only last 100 entries
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]
            
        return status
        
    def _calculate_pnl(self) -> float:
        """Calculate current unrealized P&L"""
        if self.strategy.position == 0:
            return 0.0
            
        current_price = self.data_manager.get_current_price('btcusd')
        entry_price = getattr(self.strategy, 'last_trade_price', 0)
        position_size = getattr(self.strategy, 'position_size', 0)
        
        if self.strategy.position == 1:  # LONG
            return (current_price - entry_price) * position_size
        else:  # SHORT
            return (entry_price - current_price) * abs(position_size)
            
    def _check_alerts(self, current_signal: int, position_matches: bool, bars_to_signal: int) -> List[str]:
        """Check for important alerts"""
        alerts = []
        
        # Signal approaching
        if bars_to_signal <= 3 and position_matches:
            alerts.append(f"⚠️ SIGNAL APPROACHING: Potential flip in ~{bars_to_signal} bars")
            
        # Signal conflict
        if not position_matches:
            alerts.append("🚨 SIGNAL CONFLICT: Current signal does not match position!")
            
        # Check for missed signals
        missed = self._check_missed_signals()
        if missed:
            alerts.extend(missed)
            
        return alerts
        
    def _check_missed_signals(self) -> List[str]:
        """Check signal history for missed opportunities"""
        alerts = []
        
        # Look for sustained signals that didn't result in trades
        if len(self.signal_history) >= 5:
            # Check last 5 entries
            recent = self.signal_history[-5:]
            
            # All same signal and not matching position?
            signals = [h['signal'] for h in recent]
            if len(set(signals)) == 1 and not recent[-1]['position_matches']:
                alerts.append("❌ MISSED SIGNAL: Signal has been active for 5+ bars without execution")
                
        return alerts
        
    def get_signal_history(self, hours: int = 1) -> List[Dict]:
        """Get signal history for the specified number of hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [h for h in self.signal_history if h['timestamp'] > cutoff_time]
        
    def detect_missed_trades(self) -> List[Dict]:
        """Analyze history to detect potentially missed trading opportunities"""
        missed = []
        
        # This would need access to trade history to properly detect missed trades
        # For now, return signals that persisted without position changes
        
        return missed