#!/usr/bin/env python3
"""
Backtest Comparison Logger
Creates detailed logs for comparing live trading with backtesting results
Ensures we can verify that trades happen at the exact same times
"""
import json
import os
from datetime import datetime
import hashlib


class BacktestComparisonLogger:
    """
    Logger that creates structured logs for comparing live trading with backtests
    """
    
    def __init__(self, log_dir="logs/backtest_comparison"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create daily log file
        today = datetime.now().strftime("%Y-%m-%d")
        self.log_file = os.path.join(log_dir, f"live_trading_{today}.jsonl")
        self.signal_file = os.path.join(log_dir, f"signals_{today}.jsonl")
        self.hourly_file = os.path.join(log_dir, f"hourly_bars_{today}.jsonl")
        
        # Log session start
        self.log_session_start()
        
    def log_session_start(self):
        """Log the start of a trading session"""
        event = {
            "event_type": "SESSION_START",
            "timestamp": datetime.now().isoformat(),
            "timestamp_unix": datetime.now().timestamp(),
            "strategy_config": self._get_strategy_config()
        }
        self._append_to_log(self.log_file, event)
        
    def log_hourly_bar(self, bar_time, open_price, high, low, close, volume, 
                      ma_short, ma_long, ma_short_value, ma_long_value):
        """Log hourly bar data with calculated indicators"""
        event = {
            "event_type": "HOURLY_BAR",
            "bar_time": bar_time.isoformat(),
            "bar_time_unix": bar_time.timestamp(),
            "logged_at": datetime.now().isoformat(),
            "ohlcv": {
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume)
            },
            "indicators": {
                "ma_short_period": ma_short,
                "ma_long_period": ma_long,
                "ma_short_value": float(ma_short_value),
                "ma_long_value": float(ma_long_value),
                "ma_spread": float(ma_short_value - ma_long_value),
                "ma_spread_pct": float((ma_short_value - ma_long_value) / ma_long_value * 100)
            },
            "bar_hash": self._calculate_bar_hash(bar_time, open_price, high, low, close)
        }
        self._append_to_log(self.hourly_file, event)
        
    def log_signal_evaluation(self, timestamp, current_price, ma_short_value, ma_long_value,
                            previous_signal, current_signal, will_trade, reason):
        """Log every signal evaluation for comparison"""
        event = {
            "event_type": "SIGNAL_EVALUATION",
            "timestamp": timestamp.isoformat(),
            "timestamp_unix": timestamp.timestamp(),
            "evaluation_time": datetime.now().isoformat(),
            "price": float(current_price),
            "indicators": {
                "ma_short": float(ma_short_value),
                "ma_long": float(ma_long_value),
                "ma_spread": float(ma_short_value - ma_long_value),
                "ma_spread_pct": float((ma_short_value - ma_long_value) / ma_long_value * 100)
            },
            "signals": {
                "previous": previous_signal,
                "current": current_signal,
                "changed": previous_signal != current_signal
            },
            "decision": {
                "will_trade": will_trade,
                "reason": reason
            }
        }
        self._append_to_log(self.signal_file, event)
        
    def log_trade_decision(self, timestamp, trade_type, price, amount, 
                         position_before, position_after, ma_short, ma_long,
                         hourly_bar_time, exact_trigger_time):
        """Log trade execution with all relevant data"""
        event = {
            "event_type": "TRADE_EXECUTION",
            "timestamp": timestamp.isoformat(),
            "timestamp_unix": timestamp.timestamp(),
            "execution_time": datetime.now().isoformat(),
            "hourly_bar_time": hourly_bar_time.isoformat(),
            "exact_trigger_time": exact_trigger_time.isoformat(),
            "trade": {
                "type": trade_type,
                "price": float(price),
                "amount": float(amount),
                "fee_rate": 0.0012
            },
            "position": {
                "before": position_before,
                "after": position_after
            },
            "indicators_at_trade": {
                "ma_short": float(ma_short),
                "ma_long": float(ma_long),
                "ma_spread": float(ma_short - ma_long),
                "ma_spread_pct": float((ma_short - ma_long) / ma_long * 100)
            },
            "verification_hash": self._calculate_trade_hash(
                hourly_bar_time, trade_type, ma_short, ma_long
            )
        }
        self._append_to_log(self.log_file, event)
        
    def log_no_trade_hour(self, bar_time, reason, ma_short, ma_long):
        """Log hours where no trade occurred for completeness"""
        event = {
            "event_type": "NO_TRADE_HOUR",
            "bar_time": bar_time.isoformat(),
            "bar_time_unix": bar_time.timestamp(),
            "logged_at": datetime.now().isoformat(),
            "reason": reason,
            "indicators": {
                "ma_short": float(ma_short),
                "ma_long": float(ma_long),
                "ma_spread": float(ma_short - ma_long)
            }
        }
        self._append_to_log(self.log_file, event)
        
    def _get_strategy_config(self):
        """Get current strategy configuration"""
        # Read from best_strategy.json if it exists
        try:
            with open("best_strategy.json", "r") as f:
                config = json.load(f)
                return {
                    "short_window": config.get("Short_Window"),
                    "long_window": config.get("Long_Window"),
                    "strategy_type": config.get("strategy_type", "MA")
                }
        except:
            return {"error": "Could not load config"}
            
    def _calculate_bar_hash(self, bar_time, open_price, high, low, close):
        """Create hash of bar data for verification"""
        data = f"{bar_time.isoformat()}|{open_price}|{high}|{low}|{close}"
        return hashlib.md5(data.encode()).hexdigest()[:8]
        
    def _calculate_trade_hash(self, bar_time, trade_type, ma_short, ma_long):
        """Create hash for trade verification"""
        data = f"{bar_time.isoformat()}|{trade_type}|{ma_short:.2f}|{ma_long:.2f}"
        return hashlib.md5(data.encode()).hexdigest()[:8]
        
    def _append_to_log(self, filename, data):
        """Append JSON line to log file"""
        with open(filename, "a") as f:
            f.write(json.dumps(data) + "\n")