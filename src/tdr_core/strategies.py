# ----------------------------------------------------------------------------
# FULL FILE PATH: src/tdr_core/strategies.py
# ----------------------------------------------------------------------------
# CHANGES MADE:
#   1) Introduced position-based cost-basis tracking for accurate entry price & PnL.
#   2) Distinguish 'trades.json' (live trades) vs 'non-live-trades.json' (dry-run).
#   3) Return 'theoretical_trade' in get_status() if no real trade executed.
#   4) Clear self.theoretical_trade after a real trade.
#   5) (NEW) If live_trading=True, we now append new trades to 'trades.json' in real time
#      so you don't have to wait for strategy.stop().
#   6) (NEW) We skip trades if fill_btc < 1e-8, avoiding "zero position" confusion.
#   7) (Previously) Removed partial-fill clamp in the 'sell' side to allow short entries.
#   8) (NEW) For the 'buy' side, we now properly handle leftover BTC if you move from short to a net long.
#
# NOTE: We have taken care to preserve all existing comments and code, only adding
#       the minimal lines required for short->long leftover logic.
# ----------------------------------------------------------------------------
# BUG FIXES IN THIS VERSION:
#   1) Fixed position tracking: After selling all BTC, position is now correctly set to -1 (SHORT)
#      instead of 0 (neutral). This system is never neutral - always 100% BTC or 100% USD.
#   2) Fixed short position cost basis tracking for accurate P&L calculations
#   3) Fixed parameter mismatch between config and hardcoded thresholds
# ----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import json
import time
import logging
import threading
import os
from datetime import datetime, timedelta

from indicators.technical_indicators import (
    ensure_datetime_index,
    add_moving_averages,
    generate_ma_signals,
    calculate_rsi,
    generate_rsi_signals,
    calculate_bollinger_bands,
    generate_bollinger_band_signals,
    calculate_macd,
    generate_macd_signals
)


###############################################################################
class DiagnosticLogger:
    """
    Creates a diagnostic log file for each trading session that captures:
    - Periodic snapshots every 10 minutes
    - Signal evaluations and why they were/weren't taken
    - Position changes and P&L
    - Market regime changes
    - Errors and warnings
    """
    
    def __init__(self, strategy_name="strategy"):
        self.start_time = datetime.now()
        self.filename = f"diagnostics_{strategy_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        self.events = []
        self.last_snapshot_time = datetime.now()
        self.snapshot_interval = 600  # 10 minutes

        self.last_signal_eval = None  # Track last signal to avoid duplicates
        self.signal_eval_count = 0
        self.max_events = 2000  # Limit total events to keep file size down
        
        # Create initial file immediately
        self._save()
        
    def log_event(self, event_type, data):
        """Log an event with timestamp."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        self.events.append(event)
        self._save()
        
    def log_snapshot(self, position_info, market_data, strategy_state):
        """Log a periodic snapshot of system state."""
        snapshot = {
            "position": position_info,
            "market": market_data,
            "strategy": strategy_state
        }
        self.log_event("SNAPSHOT", snapshot)
        self.last_snapshot_time = datetime.now()
        
    def log_signal_evaluation(self, signal_type, signal_value, reason, will_trade, why_not=None):
        """Log when signals are evaluated - but avoid duplicates."""
        # Create a signature of this signal evaluation
        signal_signature = f"{signal_type}:{signal_value}:{will_trade}"
        
        # Skip if this is the same as the last signal (avoid logging every minute)
        if signal_signature == self.last_signal_eval and not will_trade:
            self.signal_eval_count += 1
            # Log every 5th duplicate or if it's been 10 minutes (more frequent logging)
            if self.signal_eval_count < 5 and (datetime.now() - self.last_snapshot_time).total_seconds() < 600:
                return
                
        self.last_signal_eval = signal_signature
        self.signal_eval_count = 0

        data = {
            "signal_type": signal_type,
            "signal_value": signal_value,
            "reason": reason,
            "will_trade": will_trade,
            "why_not": why_not
        }
        self.log_event("SIGNAL_EVAL", data)
        
    def log_trade_execution(self, trade_type, price, amount, position_before, position_after, pnl):
        """Log trade executions."""
        data = {
            "trade_type": trade_type,
            "price": price,
            "amount": amount,
            "position_before": position_before,
            "position_after": position_after,
            "pnl": pnl
        }
        self.log_event("TRADE", data)
        
    def log_regime_change(self, old_regime, new_regime, confidence, metrics):
        """Log market regime changes."""
        data = {
            "old_regime": old_regime,
            "new_regime": new_regime,
            "confidence": confidence,
            "metrics": metrics
        }
        self.log_event("REGIME_CHANGE", data)
        
    def log_error(self, error_msg, stack_trace=None):
        """Log errors and warnings."""
        import traceback
        data = {
            "error": error_msg,
            "stack_trace": stack_trace or traceback.format_exc()
        }
        self.log_event("ERROR", data)
        
    def log_position_anomaly(self, description, details):
        """Log position tracking anomalies."""
        data = {
            "description": description,
            "details": details
        }
        self.log_event("POSITION_ANOMALY", data)
        
    def log_position_validation(self, validation_results):
        """Log position validation results for debugging."""
        self.log_event("POSITION_VALIDATION", {
            "timestamp": datetime.now().isoformat(),
            "validation_results": validation_results,
            "corrections_made": validation_results.get("corrections", []),
            "warnings": validation_results.get("warnings", [])
        })

    def should_snapshot(self):
        """Check if it's time for a periodic snapshot."""
        return (datetime.now() - self.last_snapshot_time).total_seconds() >= self.snapshot_interval
        
    def _save(self):
        """Save events to file."""
        try:
            # Trim events if we exceed max
            if len(self.events) > self.max_events:
                # Keep the first 100 events (for context) and the most recent events
                self.events = self.events[:100] + self.events[-(self.max_events-100):]
                # Add a marker event
                self.events.insert(100, {
                    "timestamp": datetime.now().isoformat(),
                    "type": "TRIMMED",
                    "data": {"message": f"Trimmed {len(self.events) - self.max_events} old events to save space"}
                })
            
            summary = {
                "session_start": self.start_time.isoformat(),
                "last_update": datetime.now().isoformat(),
                "total_events": len(self.events),
                "event_types": {
                    event_type: len([e for e in self.events if e["type"] == event_type])
                    for event_type in set(e["type"] for e in self.events)
                },
                "file_size_kb": os.path.getsize(self.filename) / 1024 if os.path.exists(self.filename) else 0,
                "events": self.events
            }
            with open(self.filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save diagnostic log: {e}")


    def export_summary(self, max_events_per_type=5):
        """Export a condensed summary of recent diagnostic events."""
        try:
            summary = {
                "session_info": {
                    "start_time": self.start_time.isoformat(),
                    "duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
                    "total_events": len(self.events),
                    "file_size_kb": os.path.getsize(self.filename) / 1024 if os.path.exists(self.filename) else 0
                },
                "event_summary": {},
                "recent_events": {}
            }
            
            # Count events by type
            event_types = {}
            for event in self.events:
                event_type = event["type"]
                event_types[event_type] = event_types.get(event_type, 0) + 1
            summary["event_summary"] = event_types
            
            # Get recent events of each important type
            important_types = ["TRADE", "REGIME_CHANGE", "SIGNAL_EVAL", "ERROR", "POSITION_ANOMALY", "MULTI_PART_TRADE"]
            
            for event_type in important_types:
                matching_events = [e for e in self.events if e["type"] == event_type]
                recent_events = matching_events[-max_events_per_type:] if matching_events else []
                
                # Condense the event data
                condensed_events = []
                for event in recent_events:
                    condensed = {
                        "timestamp": event["timestamp"],
                        "type": event["type"]
                    }
                    
                    # Add key data based on event type
                    if event_type == "SIGNAL_EVAL":
                        data = event.get("data", {})
                        # Convert numpy types to native Python types
                        signal_value = data.get("signal_value")
                        if hasattr(signal_value, 'item'):
                            signal_value = signal_value.item()
                        elif isinstance(signal_value, (np.integer, np.floating)):
                            signal_value = float(signal_value)
                        condensed.update({
                            "signal_type": data.get("signal_type"),
                            "signal_value": signal_value,  # Use the converted value
                            "will_trade": data.get("will_trade"),
                            "why_not": data.get("why_not")
                        })
                    elif event_type == "TRADE":
                        data = event.get("data", {})
                        condensed.update({
                            "trade_type": data.get("trade_type"),
                            "price": float(data.get("price", 0)),
                            "amount": float(data.get("amount", 0)),
                            "pnl": float(data.get("pnl", 0))

                        })

                    elif event_type == "REGIME_CHANGE":
                        data = event.get("data", {})
                        confidence = data.get("confidence")
                        if hasattr(confidence, 'item'):
                            confidence = confidence.item()
                        elif isinstance(confidence, (np.integer, np.floating)):
                            confidence = float(confidence)
                        condensed.update({
                            "old_regime": data.get("old_regime"),
                            "new_regime": data.get("new_regime"),
                            "confidence": confidence  # Use converted value
                        })
                    else:
                        # For other types, include limited data
                        condensed["summary"] = str(event.get("data", {}))[:200]
                    
                    condensed_events.append(condensed)
                
                if condensed_events:
                    summary["recent_events"][event_type] = condensed_events
            
            return summary
            
        except Exception as e:
            return {"error": f"Failed to create summary: {e}"}
 
            
    def close(self):
        """Final save on shutdown - add a closing event."""
        self.log_event("SESSION_END", {
            "reason": "Normal shutdown",
            "duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "final_event_count": len(self.events)
        })
        
    def crash_recovery(self):
        """Called on unexpected exit - tries to save current state."""
        self.log_event("CRASH", {
            "reason": "Unexpected termination",
            "last_event": self.events[-1] if self.events else None
        })
        self._save()


    def log_multi_part_trade(self, parts, total_btc, avg_price, reason):
        """Log a multi-part trade execution."""
        data = {
            "parts": parts,
            "total_btc": total_btc,
            "average_price": avg_price,
            "reason": reason,
            "note": "Multiple trades executed as one logical trade to avoid 90% rule"
        }
        self.log_event("MULTI_PART_TRADE", data)

###############################################################################
class MACrossoverStrategy:
    """
    Implements a basic Moving Average Crossover strategy with position tracking
    and optional daily trade limits.
    """

    def __init__(
        self,
        data_manager,
        short_window,
        long_window,
        amount,
        symbol,
        logger,
        live_trading=False,
        max_trades_per_day=5,
        initial_position=0,
        initial_balance_btc=0.0,
        initial_balance_usd=0.0,
        **kwargs  # Accept additional keyword arguments
    ):
        self.data_manager = data_manager
        self.order_placer = data_manager.order_placer
        self.short_window = short_window
        self.long_window = long_window
        self.initial_amount = amount
        self.current_amount = amount
        self.symbol = symbol
        self.logger = logger
        self.position = initial_position
        self.running = False
        self.live_trading = live_trading
        self.trade_log = []

        # Decide which trades file to use (live vs. non-live).
        # Use absolute path in the project root directory for consistency
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if self.live_trading:
            self.trade_log_file = os.path.join(project_root, 'trades.json')
        else:
            self.trade_log_file = os.path.join(project_root, 'non-live-trades.json')

        self.last_signal_time = None
        self.last_trade_reason = None
        self.last_trade_data_source = None
        self.last_trade_signal_timestamp = None
        self.next_trigger = None
        self.current_trends = {}
        self.df_ma = pd.DataFrame()
        self.strategy_start_time = datetime.now()

        # Initial balances for BTC & USD (and legacy "amount" for P&L).
        self.initial_balance_btc = initial_balance_btc
        self.initial_balance_usd = initial_balance_usd
        self.initial_balance = amount
        self.current_balance = amount

        self.balance_btc = initial_balance_btc
        self.balance_usd = initial_balance_usd

        self.fee_percentage = 0.0012
        self.last_trade_price = None
        self.total_fees_paid = 0
        self.trades_executed = 0
        self.profitable_trades = 0
        self.total_profit_loss = 0

        # Daily trade limits
        self.max_trades_per_day = max_trades_per_day
        self.default_max_trades_per_day = max_trades_per_day  # Store default for daily reset
        self.trade_count_today = 0
        self.current_day = datetime.utcnow().date()
        self.logger.debug(
            f"Trade limit set to {self.max_trades_per_day} trades/day.")

        self.trades_this_hour = []

        # Cost basis logic
        self._position_cost_basis = 0.0
        self._position_size = 0.0
        self._debug_position_tracking = True  # Enable position tracking debug
        
        # Track position changes
        self._last_position_log = {'size': 0.0, 'cost_basis': 0.0}

        # For storing an initial theoretical trade if hist_position matches user request
        self.theoretical_trade = None
        
        # MA separation threshold (default 0.3%)
        self.ma_separation_threshold = 0.3
        
        # Initialize last_trade_time to prevent save_resume_state errors
        self.last_trade_time = None

        # Initialize diagnostic logger
        self.diagnostic_logger = DiagnosticLogger(f"MA_{short_window}_{long_window}")
        self.diagnostic_logger.log_event("SESSION_START", {
            "strategy": "MA_CROSSOVER",
            "parameters": {"short": short_window, "long": long_window, "live": live_trading}
        })
        
        # Initialize comparison logger if enabled
        self.comparison_logger = None
        enable_comparison = kwargs.get('enable_comparison_logging', False)
        self.logger.info(f"[COMPARISON_DEBUG] enable_comparison_logging = {enable_comparison}")
        
        if enable_comparison:
            try:
                self.logger.info("[COMPARISON_DEBUG] Attempting to import BacktestComparisonLogger...")
                from tdr_core.backtest_comparison_logger import BacktestComparisonLogger
                self.logger.info("[COMPARISON_DEBUG] Import successful, creating instance...")
                self.comparison_logger = BacktestComparisonLogger()
                self.logger.info("✅ Initialized BacktestComparisonLogger for live/backtest comparison")
            except Exception as e:
                self.logger.error(f"[COMPARISON_DEBUG] Could not initialize comparison logger: {e}")
                import traceback
                self.logger.error(f"[COMPARISON_DEBUG] Traceback: {traceback.format_exc()}")
        
        # Initialize whipsaw tracking
        self.whipsaw_tracker = {
            'trades': [],  # List of all trades with timestamps
            'whipsaws': [],  # Detected whipsaw patterns
            'stats': {
                'total_whipsaws': 0,
                'whipsaw_losses': 0.0,
                'whipsaw_timeframes': [],  # Time between flip-flops
                'false_breakouts': 0,
                'avg_whipsaw_cost': 0.0
            },
            'detection_window': 3600 * 4,  # 4 hours to detect whipsaw
            'last_analysis': None
        }

        # Register real-time callback
        data_manager.add_trade_observer(self.check_instant_signal)

        # Track staleness detection
        mtm_usd, _ = self.get_mark_to_market_values()
        self.max_mtm_usd = mtm_usd
        self.min_mtm_usd = mtm_usd
        self.max_balance_usd = self.balance_usd
        self.min_balance_usd = self.balance_usd
        self.max_balance_btc = self.balance_btc
        self.min_balance_btc = self.balance_btc
        
        # Load recent trades for whipsaw tracking
        self._load_recent_trades_for_whipsaw()
        
        # Initialize System Verifier for continuous regression detection
        self.system_verifier = None
        self.last_verification_time = None
        self.verification_interval = 30  # seconds
        try:
            from tdr_core.system_verifier import SystemVerifier
            self.system_verifier = SystemVerifier(self, data_manager, logger)
            self.logger.info("✅ System Verifier initialized for continuous regression detection")
        except Exception as e:
            self.logger.warning(f"Could not initialize System Verifier: {e}")
    
    @property
    def position_cost_basis(self):
        return self._position_cost_basis
    
    @position_cost_basis.setter
    def position_cost_basis(self, value):
        if value != self._position_cost_basis:
            old_value = self._position_cost_basis
            self._position_cost_basis = value
            if self._debug_position_tracking:
                import traceback
                stack = traceback.extract_stack()
                caller = stack[-2] if len(stack) >= 2 else None
                self.logger.warning(f"[POSITION_DEBUG] position_cost_basis changed from ${old_value:.2f} to ${value:.2f}")
                if caller:
                    self.logger.warning(f"[POSITION_DEBUG]   Changed by: {caller.filename}:{caller.lineno} in {caller.name}()")
                    
    @property
    def position_size(self):
        return self._position_size
    
    @position_size.setter
    def position_size(self, value):
        if value != self._position_size:
            old_value = self._position_size
            self._position_size = value
            if self._debug_position_tracking:
                import traceback
                stack = traceback.extract_stack()
                caller = stack[-2] if len(stack) >= 2 else None
                self.logger.warning(f"[POSITION_DEBUG] position_size changed from {old_value:.8f} to {value:.8f}")
                if caller:
                    self.logger.warning(f"[POSITION_DEBUG]   Changed by: {caller.filename}:{caller.lineno} in {caller.name}()")

    def _clean_up_hourly_trades(self):
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        self.trades_this_hour = [
            t for t in self.trades_this_hour if t > one_hour_ago]

    def start(self):
        """
        Start the strategy loop in a background thread.
        """
        self.running = True
        self.strategy_thread = threading.Thread(
            target=self.run_strategy_loop, daemon=True)
        self.strategy_thread.start()
        self.logger.info("Strategy loop started.")

    def stop(self):
        """
        Stop the strategy loop and, if in dry-run, save trades to file.
        If in live mode, we assume real trades were appended in real time,
        but we might still finalize them here if needed.
        """
        self.running = False
        self.logger.info("Strategy loop stopped.")
        
        # Close diagnostic logger
        if hasattr(self, 'diagnostic_logger'):
            self.diagnostic_logger.close()
            self.logger.info(f"Diagnostic log saved to: {self.diagnostic_logger.filename}")
        
        # Save trades (existing code)
        if not self.live_trading and self.trade_log:
            try:
                file_path = os.path.abspath(self.trade_log_file)
                with open(file_path, 'w') as f:
                    json.dump([t.to_dict()
                              for t in self.trade_log], f, indent=2)
                self.logger.info(
                    f"Trades logged to '{file_path}' (dry-run mode).")
            except Exception as e:
                self.logger.error(f"Failed to write trades: {e}")

    def calculate_fee(self, trade_amount, price):
        trade_value = trade_amount * price
        return trade_value * self.fee_percentage

    def run_strategy_loop(self):
        """
        Strategy loop that checks for signals every minute.
        """
        evaluation_count = 0
        last_evaluation_log = datetime.now()
        
        # Log initial position state
        self.logger.info(f"[POSITION_DEBUG] Strategy loop starting with position: size={self.position_size}, cost_basis={self.position_cost_basis}")
        
        while self.running:
            evaluation_count += 1
            current_time = datetime.now()
            
            # Log evaluation frequency every 5 evaluations or every 5 minutes
            if evaluation_count % 5 == 0 or (current_time - last_evaluation_log).total_seconds() > 300:
                self.logger.info(f"📊 Strategy evaluation #{evaluation_count} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                last_evaluation_log = current_time
            
            # Run System Verifier checks every 30 seconds
            if self.system_verifier and (self.last_verification_time is None or 
                                        (current_time - self.last_verification_time).total_seconds() >= self.verification_interval):
                try:
                    verification_results = self.system_verifier.run_all_checks()
                    self.last_verification_time = current_time
                    
                    # Log errors if any found
                    if verification_results.get('errors'):
                        self.logger.error(f"🚨 SYSTEM VERIFIER DETECTED {len(verification_results['errors'])} ISSUES!")
                        for error in verification_results['errors']:
                            self.logger.error(f"  ❌ {error}")
                except Exception as e:
                    self.logger.error(f"System Verifier failed: {e}")
            
            df = self.data_manager.get_price_dataframe(self.symbol)
            if not df.empty:
                try:
                    df = ensure_datetime_index(df)
                    df_resampled = df.resample('1H').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum',
                        'trades': 'sum',
                        'timestamp': 'last',
                        'source': 'last'
                    }).dropna()

                    if len(df_resampled) >= self.long_window:
                        df_ma = add_moving_averages(
                            df_resampled.copy(), self.short_window, self.long_window, price_col='close')
                        df_ma = generate_ma_signals(df_ma)

                        latest_signal = df_ma.iloc[-1]['MA_Signal']
                        signal_time = df_ma.index[-1]
                        current_price = df_ma.iloc[-1]['close']
                        signal_source = df_ma.iloc[-1]['source']

                        self.next_trigger = self.determine_next_trigger(df_ma)
                        self.current_trends = self.get_current_trends(df_ma)
                        self.df_ma = df_ma

                        # Validate position tracking if method exists
                        if hasattr(self, 'validate_position_tracking'):
                            self.validate_position_tracking()

                        # Check signals (MA crossover)
                        
                        # ENHANCED LOGGING: Log EVERY signal evaluation for debugging
                        short_ma = df_ma.iloc[-1]['Short_MA']
                        long_ma = df_ma.iloc[-1]['Long_MA']
                        ma_diff = short_ma - long_ma
                        ma_proximity = abs(ma_diff) / long_ma * 100
                        
                        # Store MA values for use in trade execution
                        self._last_ma_short = short_ma
                        self._last_ma_long = long_ma
                        
                        # Log comprehensive signal evaluation data
                        eval_data = {
                            "timestamp": current_time.strftime('%Y-%m-%d %H:%M:%S'),
                            "current_price": current_price,
                            "short_ma": round(short_ma, 2),
                            "long_ma": round(long_ma, 2),
                            "ma_diff": round(ma_diff, 2),
                            "ma_proximity": round(ma_proximity, 4),
                            "signal": latest_signal,
                            "position": self.position,
                            "balance_btc": self.balance_btc,
                            "balance_usd": self.balance_usd,
                            "trades_today": self.trade_count_today,
                            "live_trading": self.live_trading
                        }
                        
                        # Determine if trade would happen
                        will_trade = False
                        why_not = []
                        
                        if latest_signal == 1 and self.position <= 0:
                            # Signal says go LONG but we're SHORT or NEUTRAL
                            if self.trade_count_today >= self.max_trades_per_day:
                                why_not.append(f"Daily limit: {self.trade_count_today}/{self.max_trades_per_day}")
                            else:
                                will_trade = True
                                eval_data["action"] = "WILL_BUY"
                        elif latest_signal == -1 and self.position >= 0:
                            # Signal says go SHORT but we're LONG or NEUTRAL
                            if self.trade_count_today >= self.max_trades_per_day:
                                why_not.append(f"Daily limit: {self.trade_count_today}/{self.max_trades_per_day}")
                            else:
                                will_trade = True
                                eval_data["action"] = "WILL_SELL"
                        else:
                            why_not.append(f"Signal({latest_signal}) matches position({self.position})")
                            eval_data["action"] = "NO_TRADE"
                        
                        if why_not:
                            eval_data["blocked_reason"] = "; ".join(why_not)
                        
                        # Log EVERY evaluation to both logger and diagnostic file
                        self.logger.info(f"📊 SIGNAL_EVAL v2: MA{self.short_window}={short_ma:.0f} MA{self.long_window}={long_ma:.0f} "
                                       f"Diff={ma_diff:.0f} Prox={ma_proximity:.2f}% Sig={latest_signal} Pos={self.position} "
                                       f"Action={eval_data.get('action', 'NO_TRADE')}")
                        
                        # Log to comparison logger if available
                        if self.comparison_logger:
                            try:
                                self.comparison_logger.log_signal_evaluation(
                                    timestamp=current_time,
                                    current_price=float(current_price),
                                    ma_short_value=float(short_ma),
                                    ma_long_value=float(long_ma),
                                    previous_signal=int(getattr(self, 'last_logged_signal', 0)),
                                    current_signal=int(latest_signal),
                                    will_trade=bool(will_trade),
                                    reason=eval_data.get('action', 'NO_TRADE')
                                )
                                self.last_logged_signal = latest_signal
                            except Exception as e:
                                self.logger.error(f"[COMPARISON_LOG] Error logging signal: {e}")
                                import traceback
                                self.logger.error(f"[COMPARISON_LOG] Traceback: {traceback.format_exc()}")
                        
                        # CRITICAL: Log when we're in trigger zone
                        if ma_proximity <= self.ma_separation_threshold:
                            self.logger.warning(f"🚨 IN TRIGGER ZONE! Proximity {ma_proximity:.3f}% <= {self.ma_separation_threshold}% threshold")
                        
                        # Also log to diagnostic file
                        self.diagnostic_logger.log_event("SIGNAL_EVALUATION", eval_data)
                        
                        # Execute check_for_signals
                        self.check_for_signals(
                            latest_signal, current_price, signal_time)
                    else:
                        self.logger.debug("Not enough data to compute MAs.")
                except Exception as e:
                    self.logger.error(
                        f"Error in strategy loop for {self.symbol}: {e}")
                    self.diagnostic_logger.log_error(f"Strategy loop error: {e}")
            else:
                self.logger.debug(f"No data loaded for {self.symbol} yet.")

            # Hourly status report
            if not hasattr(self, '_last_hourly_status'):
                self._last_hourly_status = datetime.now()
            
            if (datetime.now() - self._last_hourly_status).total_seconds() >= 3600:
                self._log_hourly_status()
                self._last_hourly_status = datetime.now()

            # Reduce sleep time to 30 seconds for more responsive evaluations
            time.sleep(30)

    def _log_diagnostic_snapshot(self):
        """Create a diagnostic snapshot."""
        try:
            current_price = self.data_manager.get_current_price(self.symbol) or 0
            mtm_usd, mtm_btc = self.get_mark_to_market_values()
            
            position_info = {
                "direction": self.position,
                "btc_balance": self.balance_btc,
                "usd_balance": self.balance_usd,
                "position_size": self.position_size,
                "cost_basis": self.position_cost_basis,
                "mtm_usd": mtm_usd,
                "unrealized_pnl": self._calculate_unrealized_pnl()
            }
            
            market_data = {
                "current_price": current_price,
                "short_ma": self.df_ma.iloc[-1]['Short_MA'] if not self.df_ma.empty else None,
                "long_ma": self.df_ma.iloc[-1]['Long_MA'] if not self.df_ma.empty else None,
                "signal": self.df_ma.iloc[-1]['MA_Signal'] if not self.df_ma.empty else None
            }
            
            strategy_state = {
                "trades_today": self.trade_count_today,
                "total_trades": self.trades_executed,
                "total_pnl": self.total_profit_loss
            }
            
            self.diagnostic_logger.log_snapshot(position_info, market_data, strategy_state)
        except Exception as e:
            self.diagnostic_logger.log_error(f"Failed to create snapshot: {e}")

    def _log_trade_status(self):
        """Log full status after a trade execution."""
        try:
            status = self.get_status()
            self.diagnostic_logger.log_event("POST_TRADE_STATUS", {
                "position": status['position'],
                "balance_btc": status['balance_btc'],
                "balance_usd": status['balance_usd'],
                "position_info": status.get('position_info', {}),
                "total_pnl": status['total_profit_loss'],
                "trades_today": status['trade_count_today'],
                "last_trade": status.get('last_trade', 'Unknown')
            })
        except Exception as e:
            self.diagnostic_logger.log_error(f"Failed to log post-trade status: {e}")
            
    def _log_hourly_status(self):
        """Log comprehensive hourly status report."""
        try:
            status = self.get_status()
            current_price = self.data_manager.get_current_price(self.symbol) or 0
            
            hourly_report = {
                "current_price": current_price,
                "position": {
                    "direction": status['position'],
                    "btc": status['balance_btc'],
                    "usd": status['balance_usd'],
                    "mtm_usd": status['mark_to_market_usd'],
                    "mtm_btc": status['mark_to_market_btc']
                },
                "position_details": status.get('position_info', {}),
                "performance": {
                    "total_return_pct": status['total_return_pct'],
                    "total_pnl": status['total_profit_loss'],
                    "trades_executed": status['trades_executed'],
                    "win_rate": status['win_rate'],
                    "fees_paid": status['total_fees_paid']
                },
                "trading_activity": {
                    "trades_today": status['trade_count_today'],
                    "remaining_trades": status['remaining_trades_today'],
                    "last_trade": status.get('last_trade', 'None')
                },
                "technical": {
                    "ma_difference": status.get('ma_difference', 0),
                    "signal_proximity": status.get('ma_signal_proximity', 0)
                }
            }
            
            self.diagnostic_logger.log_event("HOURLY_STATUS", hourly_report)
            self.logger.info("Logged hourly status report to diagnostics")
            
        except Exception as e:
            self.diagnostic_logger.log_error(f"Failed to log hourly status: {e}")

    def determine_next_trigger(self, df_ma):
        """
        Return text describing the potential next trigger, if signals changed.
        """
        if len(df_ma) < 2:
            return None
        last_signal = df_ma.iloc[-1]['MA_Signal']
        prev_signal = df_ma.iloc[-2]['MA_Signal']
        if last_signal != prev_signal:
            if last_signal == 1:
                return "Next trigger: Potential SELL if short crosses below long."
            elif last_signal == -1:
                return "Next trigger: Potential BUY if short crosses above long."
        return "Next trigger: Awaiting next crossover signal."

    def get_current_trends(self, df_ma):
        """
        Analyze short/long MA slopes and price trend.
        """
        if len(df_ma) < 2:
            return {}
        short_ma_curr = df_ma.iloc[-1]['Short_MA']
        short_ma_prev = df_ma.iloc[-2]['Short_MA']
        long_ma_curr = df_ma.iloc[-1]['Long_MA']
        long_ma_prev = df_ma.iloc[-2]['Long_MA']

        short_ma_slope = short_ma_curr - short_ma_prev
        long_ma_slope = long_ma_curr - long_ma_prev

        return {
            'Short_MA_Slope': 'Upwards' if short_ma_slope > 0 else 'Downwards',
            'Long_MA_Slope': 'Upwards' if long_ma_slope > 0 else 'Downwards',
            'Price_Trend': 'Bullish' if short_ma_curr > long_ma_curr else 'Bearish',
            'Trend_Strength': abs(short_ma_curr - long_ma_curr) / long_ma_curr * 100 if long_ma_curr else 0
        }

    def check_instant_signal(self, symbol, price, timestamp, trade_reason):
        """
        Real-time callback for each new trade. If there's a new crossover, act now.
        """
        if not self.running:
            return
        if symbol != self.symbol:
            return

        df_live = self.data_manager.get_price_dataframe(symbol)
        if df_live.empty:
            return

        df_live = ensure_datetime_index(df_live)
        if len(df_live) < self.long_window:
            return

        df_ma = df_live.copy()
        df_ma['Short_MA'] = df_ma['close'].rolling(self.short_window).mean()
        df_ma['Long_MA'] = df_ma['close'].rolling(self.long_window).mean()
        df_ma.dropna(inplace=True)
        if df_ma.empty:
            return

        latest = df_ma.iloc[-1]
        short_ma_now = latest['Short_MA']
        long_ma_now = latest['Long_MA']
        signal_now = 1 if short_ma_now > long_ma_now else -1

        if len(df_ma) < 2:
            return
        prev = df_ma.iloc[-2]
        prev_signal = 1 if prev['Short_MA'] > prev['Long_MA'] else -1
        if signal_now == prev_signal:
            return

        signal_time = df_ma.index[-1]
        self.check_for_signals(signal_now, price, signal_time)

    def check_for_signals(self, latest_signal, current_price, signal_time):
        """
        If the new MA signal differs from our current position, place trades.
        Also checks daily trade-limit; if at max, it skips.
        """
        # ENHANCED LOGGING: Log entry to check_for_signals
        self.logger.info(f"🔍 CHECK_FOR_SIGNALS: signal={latest_signal}, price=${current_price:.0f}, "
                        f"position={self.position}, time={signal_time}, live={self.live_trading}")
        
        today = datetime.utcnow().date()
        if today != self.current_day:
            self.current_day = today
            self.trade_count_today = 0
            # Reset to default trade limit on new day
            if hasattr(self, 'default_max_trades_per_day'):
                self.max_trades_per_day = self.default_max_trades_per_day
                self.logger.debug(f"New day, resetting daily trade count and limit to {self.max_trades_per_day}.")
            else:
                self.logger.debug("New day, resetting daily trade count.")

        if self.last_signal_time == signal_time:
            self.logger.debug(f"⏭️ Skipping - same signal time as last: {signal_time}")
            return

        # CRITICAL TRADE DECISION LOG
        self.logger.warning(f"🎯 TRADE DECISION: Signal={latest_signal} vs Position={self.position} | "
                          f"Will trade? {(latest_signal == 1 and self.position <= 0) or (latest_signal == -1 and self.position >= 0)} | "
                          f"Live={self.live_trading} | Today's trades={self.trade_count_today}/{self.max_trades_per_day}")

        # If we see a BUY signal
        if latest_signal == 1 and self.position <= 0:
            if self.trade_count_today >= self.max_trades_per_day:
                self.logger.info(
                    f"Reached daily trade limit {self.max_trades_per_day}, skipping trade.")
                return

            self.logger.info(f"Buy signal triggered at {current_price}")
            
            # Log position before trade
            position_before = {
                "btc": self.balance_btc,
                "usd": self.balance_usd,
                "position": self.position
            }

            # Store reason before trade
            self.last_trade_reason = "MA Crossover: short above long."
            
            # Execute trade FIRST
            self.buy_in_three_parts(
                current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), signal_time
            )
            
            # Update position AFTER successful trade execution
            self.position = 1
            self.trade_count_today += 1
            self.last_signal_time = signal_time
            
            # Sync position to data_manager
            if hasattr(self, 'data_manager') and self.data_manager:
                self.data_manager.position = self.position
                self.logger.info(f"[POSITION_SYNC] After BUY: synced position={self.position} to data_manager")
            
            # Log position after trade
            self.diagnostic_logger.log_trade_execution(
                trade_type="BUY",
                price=current_price,
                amount=self.position_size,
                position_before=position_before,
                position_after={"btc": self.balance_btc, "usd": self.balance_usd, "position": self.position},
                pnl=self.total_profit_loss
            )
            
            # Log to comparison logger if available
            if self.comparison_logger:
                try:
                    self.comparison_logger.log_trade_decision(
                        timestamp=signal_time,
                        trade_type="BUY",
                        price=float(current_price),
                        amount=float(self.position_size),
                        position_before=int(position_before["position"]),
                        position_after=int(self.position),
                        ma_short=int(self.short_window),
                        ma_long=int(self.long_window),
                        hourly_bar_time=signal_time,
                        exact_trigger_time=datetime.now()
                    )
                except Exception as e:
                    self.logger.error(f"[COMPARISON_LOG] Error logging trade: {e}")
            
            # Log full status after trade
            self._log_trade_status()

        # If we see a SELL signal
        elif latest_signal == -1 and self.position >= 0:
            if self.trade_count_today >= self.max_trades_per_day:
                self.logger.info(
                    f"Reached daily trade limit {self.max_trades_per_day}, skipping trade.")
                return

            self.logger.info(f"Sell signal triggered at {current_price}")

            # Log position before trade
            position_before = {
                "btc": self.balance_btc,
                "usd": self.balance_usd,
                "position": self.position
            }
 
            # Store reason before trade
            self.last_trade_reason = "MA Crossover: short below long."
            trade_btc = round(self.balance_btc, 8)
            
            # Execute trade FIRST
            self.execute_trade(
                "sell",
                current_price,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                signal_time,
                trade_btc
            )
            
            # Update position AFTER successful trade execution
            self.position = -1
            self.trade_count_today += 1
            self.last_signal_time = signal_time
            
            # Sync position to data_manager
            if hasattr(self, 'data_manager') and self.data_manager:
                self.data_manager.position = self.position
                self.logger.info(f"[POSITION_SYNC] After SELL: synced position={self.position} to data_manager")
            
            # Log position after trade
            self.diagnostic_logger.log_trade_execution(
                trade_type="SELL",
                price=current_price,
                amount=trade_btc,
                position_before=position_before,
                position_after={"btc": self.balance_btc, "usd": self.balance_usd, "position": self.position},
                pnl=self.total_profit_loss
            )
            
            # Log to comparison logger if available
            if self.comparison_logger:
                try:
                    self.comparison_logger.log_trade_decision(
                        timestamp=signal_time,
                        trade_type="SELL",
                        price=float(current_price),
                        amount=float(trade_btc),
                        position_before=int(position_before["position"]),
                        position_after=int(self.position),
                        ma_short=int(self.short_window),
                        ma_long=int(self.long_window),
                        hourly_bar_time=signal_time,
                        exact_trigger_time=datetime.now()
                    )
                except Exception as e:
                    self.logger.error(f"[COMPARISON_LOG] Error logging trade: {e}")
            
            # Log full status after trade
            self._log_trade_status()

            # Additional diagnostic logging for shorts
            if hasattr(self, 'diagnostic_logger'):
                self.diagnostic_logger.log_event("SHORT_POSITION_DETAILS", {
                    "action": "opening_short",
                    "btc_to_sell": trade_btc,
                    "price": current_price,
                    "position_size_before": self.position_size,
                    "position_size_after": self.position_size - trade_btc,  # Will be negative
                    "cost_basis_after": trade_btc * current_price,
                    "expected_tracking": "position_size should be negative for shorts"
                })

    def buy_in_three_parts(self, price, timestamp, signal_time):
        """
        Simulate a multi-part buy so we can keep within a 90% rule but only 1 daily trade.
        """
        # Store initial state
        initial_position_size = self.position_size
        initial_cost_basis = self.position_cost_basis
        initial_usd = self.balance_usd
        
        # Disable resume state saving during multi-part trade
        self._in_multi_part_trade = True
        
        # Generate unique trade group ID for this multi-part trade
        trade_group_id = f"BUY_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self._current_trade_group_id = trade_group_id
        self._multi_part_total = 3
        
        # Log the start of multi-part trade
        self.diagnostic_logger.log_event("MULTI_PART_TRADE_START", {
            "reason": "3-part buy to avoid 90% rule",
            "initial_usd": initial_usd,
            "target_price": price,
            "counts_as_trades": 1,
            "trade_group_id": trade_group_id,
            "note": "Will execute 3 buys but count as single daily trade"
        })
        
        parts = []

        self._current_multi_part_sequence = 1
        partial_btc_1 = self.get_89pct_btc_of_usd(price)
        self.execute_trade("buy", price, timestamp, signal_time, partial_btc_1)
        parts.append({"part": 1, "btc": partial_btc_1, "price": price})

        self._current_multi_part_sequence = 2
        partial_btc_2 = self.get_89pct_btc_of_usd(price)
        self.execute_trade("buy", price, timestamp, signal_time, partial_btc_2)
        parts.append({"part": 2, "btc": partial_btc_2, "price": price})

        self._current_multi_part_sequence = 3
        partial_btc_3 = self.get_89pct_btc_of_usd(price)
        self.execute_trade("buy", price, timestamp, signal_time, partial_btc_3)
        parts.append({"part": 3, "btc": partial_btc_3, "price": price})
        
        # Validate final position
        total_btc_bought = self.position_size - initial_position_size
        if total_btc_bought > 0:
            expected_cost = total_btc_bought * price * (1 + self.fee_percentage)
            actual_cost_added = self.position_cost_basis - initial_cost_basis
            
            if abs(actual_cost_added - expected_cost) > 1.0:
                self.logger.warning(f"Position tracking error detected! Expected cost: ${expected_cost:.2f}, Actual: ${actual_cost_added:.2f}")
                # Correct the cost basis
                self.diagnostic_logger.log_position_anomaly(
                    "Cost basis mismatch in update_balance",
                    {
                        "expected_cost": expected_cost,
                        "actual_cost": actual_cost_added,
                        "fill_btc": total_btc_bought,
                        "fill_price": price
                    }
                )
                # Use the actual cost added, not position_size * last_price
                self.position_cost_basis = abs(self.position_size) * price * (1 + self.fee_percentage)
                avg_price = actual_cost_added / total_btc_bought if total_btc_bought > 0 else price
                self.logger.info(f"Corrected position cost basis to ${self.position_cost_basis:.2f} (avg price: ${avg_price:.2f})")
        
        # Log final position state and multi-part summary
        self.logger.info(f"Three-part buy complete: {self.position_size:.8f} BTC, cost basis: ${self.position_cost_basis:.2f}")
        
        # Log the completed multi-part trade
        self.diagnostic_logger.log_multi_part_trade(
            parts=parts,
            total_btc=total_btc_bought,
            avg_price=price,
            reason="MA Crossover buy signal"
        )
        
        # Log full status after multi-part trade
        self._log_trade_status()
        
        # Re-enable saving and save once for the complete trade
        self._in_multi_part_trade = False
        # Clear multi-part trade tracking
        self._current_trade_group_id = None
        self._current_multi_part_sequence = None
        self._multi_part_total = None
        self.save_resume_state()

    def get_89pct_btc_of_usd(self, price):
        available_usd = self.balance_usd * 0.89
        btc_approx = available_usd / (price * (1 + self.fee_percentage))
        return round(btc_approx, 8)

    def execute_trade(self, trade_type, price, timestamp, signal_time, trade_btc):
        """
        Execute a single trade. 
        (NEW) If trade_btc < 1e-8, skip to avoid confusion with 0.0 updates.
        (NEW) If live_trading=True, append to trades.json immediately in JSONL format.
        """
        if trade_btc < 1e-8:
            self.logger.debug(
                f"Skipping trade because fill_btc is too small: {trade_btc}")
            return

        self._clean_up_hourly_trades()
        max_trades_per_hour = 3
        if len(self.trades_this_hour) >= max_trades_per_hour:
            self.logger.info(
                f"Reached hourly trade limit {max_trades_per_hour}, skipping trade.")

            # Log to diagnostics when trade is blocked
            self.diagnostic_logger.log_event("TRADE_BLOCKED", {
                "reason": "Hourly trade limit reached",
                "trades_this_hour": len(self.trades_this_hour),
                "max_per_hour": max_trades_per_hour,
                "trade_type": trade_type,
                "price": price,
                "amount": trade_btc
            })
            return

        from tdr_core.trade import Trade
        
        # Log trade reason prominently
        if "Pivot break:" in self.last_trade_reason:
            self.logger.warning(f"🎯 EXECUTING PIVOT-TRIGGERED TRADE: {self.last_trade_reason}")
        else:
            self.logger.info(f"📊 Executing trade: {self.last_trade_reason}")
            
        trade_info = Trade(
            trade_type,
            self.symbol,
            trade_btc,
            price,
            datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'),
            self.last_trade_reason,
            'live' if self.live_trading else 'historical',
            signal_time,
            live_trading=self.live_trading,
            trade_group_id=getattr(self, '_current_trade_group_id', None),
            multi_part_sequence=getattr(self, '_current_multi_part_sequence', None),
            multi_part_total=getattr(self, '_multi_part_total', None)
        )

        self.last_trade_data_source = trade_info.data_source
        self.last_trade_signal_timestamp = signal_time

        # Place order with the exchange if live.
        if self.live_trading:
            # JSONL Format: Write pre-trade entry BEFORE placing order
            try:
                file_path = os.path.abspath(self.trade_log_file)
                pre_trade_entry = {
                    "event_type": "PRE_TRADE",
                    "timestamp": datetime.now().isoformat(),
                    "signal_timestamp": signal_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "trade_type": trade_type,
                    "symbol": self.symbol,
                    "amount_btc": trade_btc,
                    "signal_price": price,
                    "reason": self.last_trade_reason,
                    "trade_group_id": getattr(self, '_current_trade_group_id', None),
                    "multi_part_sequence": getattr(self, '_current_multi_part_sequence', None),
                    "multi_part_total": getattr(self, '_multi_part_total', None),
                    "position_before": self.position,
                    "ma_values": {
                        "short_window": self.ma_short_window,
                        "long_window": self.ma_long_window,
                        "short_value": getattr(self, '_last_ma_short', None),
                        "long_value": getattr(self, '_last_ma_long', None)
                    }
                }
                
                # Append to JSONL file
                with open(file_path, 'a') as f:
                    f.write(json.dumps(pre_trade_entry) + '\n')
                    
                self.logger.debug(f"Logged PRE_TRADE to {self.trade_log_file}")
            except Exception as e:
                self.logger.error(f"Failed to log PRE_TRADE: {e}")
            
            # Place the actual order
            result = self.order_placer.place_order(
                f"market-{trade_type}", self.symbol, trade_btc)
            self.logger.info(f"Executed LIVE {trade_type} order: {result}")
            trade_info.order_result = result
            
            # JSONL Format: Write post-trade entry with Bitstamp results
            try:
                # Extract actual values from Bitstamp response
                fill_price = price  # Default to signal price
                trade_id = None
                if result.get("price"):
                    try:
                        fill_price = float(result["price"])
                        self.logger.info(f"Using actual fill price: ${fill_price:.2f} (vs signal price ${price:.2f})")
                    except:
                        pass
                
                if result.get("id"):
                    trade_id = result.get("id")
                
                post_trade_entry = {
                    "event_type": "POST_TRADE",
                    "timestamp": datetime.now().isoformat(),
                    "signal_timestamp": signal_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "trade_type": trade_type,
                    "symbol": self.symbol,
                    "amount_btc": trade_btc,
                    "signal_price": price,
                    "fill_price": fill_price,
                    "bitstamp_trade_id": trade_id,
                    "bitstamp_response": result,
                    "status": "success" if result.get("status") != "error" else "failed",
                    "trade_group_id": getattr(self, '_current_trade_group_id', None),
                    "multi_part_sequence": getattr(self, '_current_multi_part_sequence', None),
                    "multi_part_total": getattr(self, '_multi_part_total', None)
                }
                
                # Append to JSONL file
                with open(file_path, 'a') as f:
                    f.write(json.dumps(post_trade_entry) + '\n')
                    
                self.logger.debug(f"Logged POST_TRADE to {self.trade_log_file}")
            except Exception as e:
                self.logger.error(f"Failed to log POST_TRADE: {e}")
            
            if result.get("status") == "error":
                self.logger.error(f"Trade failed: {result}")
                self._log_failed_trade(trade_info)
                return
            
            # Update balances & cost basis with actual fill price
            self.update_balance(trade_type, fill_price, trade_btc)

        else:
            # Dry-run => no actual exchange order, just local simulation
            self.logger.info(
                f"Executed DRY RUN {trade_type} order: {trade_info.to_dict()}")
            self.trade_log.append(trade_info)
            self.update_balance(trade_type, price, trade_btc)

        self.trades_this_hour.append(datetime.utcnow())
        self._log_successful_trade(trade_info)
        
        # Track trade for whipsaw detection
        self.track_trade_for_whipsaw(trade_type, price, timestamp)

        # If a theoretical trade existed, clear it
        if self.theoretical_trade is not None:
            self.logger.debug(
                "Clearing theoretical trade because an actual trade occurred.")
            self.theoretical_trade = None

    def update_balance(self, trade_type, fill_price, fill_btc):
        """
        Update balance after a trade, tracking cost basis & partial fills.
        (NEW) If final fill_btc ends up 0, skip it to avoid no-op.

        REMINDER: 
         - We have removed partial-fill clamp on the 'sell' side to allow short entries.
         - We also fix leftover logic on the 'buy' side so going from short->long updates position_size properly.
        
        BUG FIX: After selling all BTC, we now correctly set position to -1 (SHORT)
                 instead of 0 (neutral), since this system is never neutral.
        """
        fee = self.calculate_fee(fill_btc, fill_price)
        self.total_fees_paid += fee

        if trade_type == "buy":
            cost_usd = fill_btc * fill_price
            total_cost_usd = cost_usd + fee
            if total_cost_usd > self.balance_usd:
                # partial fill correction
                possible_btc = self.balance_usd / \
                    (fill_price * (1 + self.fee_percentage))
                possible_btc = round(possible_btc, 8)
                if possible_btc < 1e-8:
                    self.logger.debug(
                        f"Cannot buy anything with leftover USD. Skipping.")
                    return
                fill_btc = possible_btc
                cost_usd = fill_btc * fill_price
                fee = self.calculate_fee(fill_btc, fill_price)
                total_cost_usd = cost_usd + fee

            self.balance_usd -= total_cost_usd
            self.balance_btc += fill_btc

            # Critical fix: Handle position tracking correctly
            if self.position == -1 and self.position_size <= 0:
                # Transitioning from short to long - reset tracking
                self.logger.info(f"Transitioning from SHORT to LONG")
                self.position_size = fill_btc
                self.position_cost_basis = fill_btc * fill_price
            elif self.position_size >= 0:
                # Adding to existing long position
                self.position_cost_basis += (fill_btc * fill_price)
                self.position_size += fill_btc
            else:
                # Covering short position
                if abs(self.position_size) >= fill_btc:
                    # Just reducing short
                    self.position_size += fill_btc
                    # Don't change cost basis when covering
                else:
                    # Going from short through neutral to long
                    leftover_btc = fill_btc - abs(self.position_size)
                    self.position_size = leftover_btc
                    self.position_cost_basis = leftover_btc * fill_price

            # Validate position after update
            # DISABLED: This check was incorrectly resetting valid entry prices
            # if self.position_size > 0 and self.position_cost_basis > 0:
            #     avg_entry = self.position_cost_basis / self.position_size
            #     if avg_entry > fill_price * 1.5:
            #         self.logger.error(f"Position tracking error: avg entry ${avg_entry:.2f} > 1.5x fill price ${fill_price:.2f}")
            #         # Reset to reasonable values
            #         self.position_cost_basis = self.position_size * fill_price
            #         self.logger.info(f"Reset cost basis to ${self.position_cost_basis:.2f}")

            if self.last_trade_price is not None and self.position == -1:
                # old code for short -> buy
                profit = fill_btc * (self.last_trade_price - fill_price) - fee
                self.current_balance += profit
                self.total_profit_loss += profit
                if profit > 0:
                    self.profitable_trades += 1

        elif trade_type == "sell":
            # If fill_btc > current balance, we open or add to a short (balance_btc goes negative).
            proceeds_usd = fill_btc * fill_price
            fee_sell = proceeds_usd * self.fee_percentage
            fee = fee_sell
            net_usd = proceeds_usd - fee

            self.balance_btc -= fill_btc
            self.balance_usd += net_usd

            if self.position_size > 0:
                # partial or full close of a long
                if fill_btc > self.position_size:
                    # going from long to short
                    fill_btc_for_long = self.position_size
                    ratio = 1.0  # fully closing that long portion
                    
                    # Log the transition for debugging
                    self.logger.info(f"LONG->SHORT transition: closing {fill_btc_for_long:.8f} BTC long, "
                                   f"opening {fill_btc - fill_btc_for_long:.8f} BTC short @ ${fill_price:.2f}")
                    
                    # Reset position tracking for the new SHORT position
                    self.position_size = 0.0
                    self.position_cost_basis = 0.0
                    
                    # leftover portion is new short
                    leftover_btc_for_short = fill_btc - fill_btc_for_long
                    if leftover_btc_for_short > 1e-8:
                        # For a SHORT position:
                        # position_size = negative BTC amount (what we sold)
                        # position_cost_basis = total USD received from sales
                        self.position_size = -leftover_btc_for_short
                        self.position_cost_basis = leftover_btc_for_short * fill_price
                        self.logger.info(f"New SHORT position established: {leftover_btc_for_short:.8f} BTC @ ${fill_price:.2f}")
                else:
                    # partial or full flatten only
                    ratio = fill_btc / self.position_size
                    cost_removed = ratio * self.position_cost_basis
                    self.position_cost_basis -= cost_removed
                    self.position_size -= fill_btc

            else:
                # Going short or adding to short: properly track the position
                if self.position_size <= 0:
                    # Adding to existing short or new short
                    self.position_size -= fill_btc  # Negative value indicates short
                    # FIX: Accumulate cost basis for average entry price calculation
                    self.position_cost_basis += fill_btc * fill_price  # Total USD value of BTC sold
                    avg_entry = self.position_cost_basis / abs(self.position_size) if self.position_size < 0 else 0
                    self.logger.info(f"Short position: sold {fill_btc:.8f} BTC @ ${fill_price:.2f}, "
                                   f"total short {abs(self.position_size):.8f} BTC, "
                                   f"avg entry ${avg_entry:.2f}, holding ${self.balance_usd:.2f} USD")
                else:
                    self.logger.error(f"Invalid state: trying to sell with positive position_size={self.position_size}")

            if self.last_trade_price is not None and self.position == 1:
                profit = fill_btc * (fill_price - self.last_trade_price) - fee
                self.current_balance += profit
                self.total_profit_loss += profit
                if profit > 0:
                    self.profitable_trades += 1

        self.last_trade_price = fill_price
        self.trades_executed += 1

        # Recompute 'current_amount' for old P&L logic
        ratio = self.current_balance / self.initial_balance if self.initial_balance else 1
        self.current_amount = self.initial_amount * ratio

        # BUG FIX: Correct position flag handling - system is never neutral!
        if abs(self.position_size) < 1e-8 and abs(self.balance_btc) < 1e-8:
            # We have ~0 BTC, so we must be SHORT (holding USD)
            if self.balance_usd > 1000:  # Have significant USD = SHORT
                if self.position != -1:
                    self.logger.warning(f"Position size near zero but position flag is {self.position}. Resetting to SHORT.")
                    self.position = -1  # FIX: Set to SHORT, not neutral!
                    # For SHORT positions, we need to track the entry price properly
                    # The position_cost_basis should be the USD received from the sale
                    # and position_size should be negative BTC amount sold
                    if self.last_trade_price > 0:
                        # For a SHORT position, we don't actually set tracking here
                        # because we're currently LONG (have BTC). This code path
                        # is for when position_size is near zero but we have USD,
                        # which shouldn't happen in normal operation.
                        # Just log the anomaly without setting incorrect values
                        self.logger.warning(f"Position tracking anomaly: position_size near zero with USD balance")
                        self.position_size = 0.0
                        self.position_cost_basis = 0.0
                    else:
                        # If we don't have last_trade_price, reset to zero but log the issue
                        self.position_size = 0.0
                        self.position_cost_basis = 0.0
                        self.logger.error("Cannot restore SHORT position tracking - no last_trade_price available")
                    self.diagnostic_logger.log_position_anomaly(
                        "Position flag corrected to SHORT",
                        {
                            "old_position": self.position,
                            "new_position": -1,
                            "balance_btc": self.balance_btc,
                            "balance_usd": self.balance_usd,
                            "reason": "Holding USD with ~0 BTC = SHORT position"
                        }
                    )
        
        # Log position state for debugging
        self.logger.debug(f"Position update: size={self.position_size:.8f}, cost_basis={self.position_cost_basis:.2f}, direction={self.position}")

        # Update max/min USD & BTC
        if self.balance_usd > self.max_balance_usd:
            self.max_balance_usd = self.balance_usd
        if self.balance_usd < self.min_balance_usd:
            self.min_balance_usd = self.balance_usd
        if self.balance_btc > self.max_balance_btc:
            self.max_balance_btc = self.balance_btc
        if self.balance_btc < self.min_balance_btc:
            self.min_balance_btc = self.balance_btc

        # Update max/min MTM
        mtm_usd, _ = self.get_mark_to_market_values()
        if mtm_usd > self.max_mtm_usd:
            self.max_mtm_usd = mtm_usd
        if mtm_usd < self.min_mtm_usd:
            self.min_mtm_usd = mtm_usd

        self.logger.info(
            f"Trade completed - Balance: ${self.current_balance:.2f}, "
            f"Fees: ${fee:.2f}, Next trade amount: {self.current_amount:.8f}, "
            f"Total P&L: ${self.total_profit_loss:.2f} || "
            f"[BTC Balance: {self.balance_btc:.8f}, USD Balance: {self.balance_usd:.2f}]"
        )
        
        # Sync position tracking to data_manager for consistent display
        if hasattr(self, 'data_manager') and self.data_manager:
            # Always sync position
            self.data_manager.position = self.position
            self.data_manager.balance_btc = self.balance_btc
            self.data_manager.balance_usd = self.balance_usd
            
            # Sync additional tracking if available
            if hasattr(self.data_manager, 'position_size'):
                self.data_manager.position_size = self.position_size
                self.data_manager.position_cost_basis = self.position_cost_basis
            
            self.logger.info(f"[POSITION_SYNC] Synced to data_manager: position={self.position}, btc={self.balance_btc:.8f}, usd={self.balance_usd:.2f}")
        else:
            self.logger.warning("[POSITION_SYNC] No data_manager available for position sync")
        
        # Automatically save resume state after each trade (unless in multi-part trade)
        if not getattr(self, '_in_multi_part_trade', False):
            self.save_resume_state()

    def _read_trades_jsonl(self):
        """Read trades from JSONL format file, extracting only successful POST_TRADE entries"""
        trades = []
        try:
            trades_file = os.path.abspath(self.trade_log_file)
            if not os.path.exists(trades_file):
                return trades
                
            with open(trades_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            # Only include successful POST_TRADE entries for backward compatibility
                            if entry.get('event_type') == 'POST_TRADE' and entry.get('status') == 'success':
                                # Convert to old format for compatibility
                                trade = {
                                    'type': entry['trade_type'],
                                    'symbol': entry['symbol'],
                                    'amount': entry['amount_btc'],
                                    'price': entry.get('fill_price', entry['signal_price']),
                                    'timestamp': entry['signal_timestamp'],
                                    'reason': entry.get('reason', ''),
                                    'trade_group_id': entry.get('trade_group_id'),
                                    'multi_part_sequence': entry.get('multi_part_sequence'),
                                    'multi_part_total': entry.get('multi_part_total')
                                }
                                trades.append(trade)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            self.logger.error(f"Error reading trades JSONL: {e}")
        return trades
    
    def validate_position_from_trades(self):
        """Validate and fix position tracking based on recent trades from trades.json"""
        self.logger.info(f"[POSITION_DEBUG] Starting validate_position_from_trades")
        self.logger.info(f"[POSITION_DEBUG] Current position before: size={self.position_size}, cost_basis={self.position_cost_basis}")
        try:
            trades = self._read_trades_jsonl()
                
            if not trades:
                self.logger.warning("No trades found in trades.json")
                return False
                
            # Find all trades for the current position (since last position reversal)
            # Work backwards to find the position entry
            position_trades = []
            current_position = None
            
            for trade in reversed(trades):
                if not current_position:
                    current_position = 'LONG' if trade['type'] == 'buy' else 'SHORT'
                    position_trades.append(trade)
                elif (current_position == 'LONG' and trade['type'] == 'buy') or \
                     (current_position == 'SHORT' and trade['type'] == 'sell'):
                    # Same direction, part of current position
                    position_trades.append(trade)
                else:
                    # Position reversal found, stop here
                    break
            
            # Reverse to get chronological order
            position_trades.reverse()
            
            if not position_trades:
                self.logger.warning("No position trades found")
                return False
            
            # Calculate average entry price for multi-part trades
            total_btc = 0.0
            total_cost = 0.0
            
            for trade in position_trades:
                btc_amount = float(trade['amount'])
                price = float(trade['price'])
                total_btc += btc_amount
                total_cost += btc_amount * price
            
            avg_entry_price = total_cost / total_btc if total_btc > 0 else 0
            
            # Check if position tracking matches the trades
            last_trade = trades[-1]
            if last_trade['type'] == 'sell':
                # Should be SHORT
                if self.position != -1:
                    self.logger.warning(f"Position mismatch: system thinks {self.position} but last trade was SELL")
                    
                # Set position tracking from aggregated trades
                self.position = -1
                self.position_size = -total_btc
                self.position_cost_basis = total_cost
                self.last_trade_price = float(last_trade['price'])
                
                self.logger.info(f"[POSITION_DEBUG] Setting SHORT position: size={self.position_size}, cost_basis={self.position_cost_basis}")
                self.logger.info(f"Validated SHORT position from {len(position_trades)} trades:")
                self.logger.info(f"  Total BTC sold: {total_btc:.8f}")
                self.logger.info(f"  Average entry price: ${avg_entry_price:.2f}")
                self.logger.info(f"  Total proceeds: ${total_cost:.2f}")
                if len(position_trades) > 1:
                    self.logger.info(f"  (Multi-part trade with {len(position_trades)} parts)")
                
            elif last_trade['type'] == 'buy':
                # Should be LONG
                if self.position != 1:
                    self.logger.warning(f"Position mismatch: system thinks {self.position} but last trade was BUY")
                    
                # Set position tracking from aggregated trades
                self.position = 1
                self.position_size = total_btc
                self.position_cost_basis = total_cost
                self.last_trade_price = float(last_trade['price'])
                
                self.logger.info(f"[POSITION_DEBUG] Setting LONG position: size={self.position_size}, cost_basis={self.position_cost_basis}")
                self.logger.info(f"Validated LONG position from {len(position_trades)} trades:")
                self.logger.info(f"  Total BTC bought: {total_btc:.8f}")
                self.logger.info(f"  Average entry price: ${avg_entry_price:.2f}")
                self.logger.info(f"  Total cost: ${total_cost:.2f}")
                if len(position_trades) > 1:
                    self.logger.info(f"  (Multi-part trade with {len(position_trades)} parts)")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating position from trades: {e}")
            return False
    
    def calculate_entry_price_from_trades(self):
        """Calculate the correct entry price from trades.json based on position type.
        Returns: (entry_price, position_trades)
        """
        try:
            trades = self._read_trades_jsonl()
                
            if not trades:
                return None, []
                
            # Find all trades for current position
            position_trades = []
            current_position = None
            
            for trade in reversed(trades):
                if not current_position:
                    current_position = 'LONG' if trade['type'] == 'buy' else 'SHORT'
                    position_trades.append(trade)
                elif (current_position == 'LONG' and trade['type'] == 'buy') or \
                     (current_position == 'SHORT' and trade['type'] == 'sell'):
                    position_trades.append(trade)
                else:
                    break
            
            # Reverse to get chronological order
            position_trades.reverse()
            
            if not position_trades:
                return None, []
            
            # Calculate entry price based on position type
            if current_position == 'LONG':
                # For LONG positions, average all BUY prices (handling multi-part trades)
                total_btc = sum(float(t['amount']) for t in position_trades)
                total_cost = sum(float(t['amount']) * float(t['price']) for t in position_trades)
                calculated_entry_price = total_cost / total_btc if total_btc > 0 else 0
                self.logger.info(f"Calculated LONG entry price from {len(position_trades)} trades: ${calculated_entry_price:.2f}")
            else:  # SHORT
                # For SHORT positions, use the last SELL price
                calculated_entry_price = float(position_trades[-1]['price'])
                self.logger.info(f"Using last SELL price for SHORT entry: ${calculated_entry_price:.2f}")
                
            return calculated_entry_price, position_trades
            
        except Exception as e:
            self.logger.error(f"Error calculating entry price from trades: {e}")
            return None, []
    
    def _restore_pivot_tracker_from_resume(self):
        """Restore pivot tracker from resume-auto-trade.json to preserve original levels across restarts."""
        try:
            import os
            import json
            from datetime import datetime
            
            # Find resume file (same logic as save_resume_state)
            resume_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'resume-auto-trade.json')
            
            if os.path.exists(resume_file):
                with open(resume_file, 'r') as f:
                    resume_data = json.load(f)
                
                # Check if pivot protection data exists
                pivot_data = resume_data.get('pivot_protection', {})
                if pivot_data.get('enabled') and pivot_data.get('tracker'):
                    tracker = pivot_data['tracker']
                    
                    # Restore pivot tracker with original levels
                    if tracker.get('levels_locked') and tracker.get('support_level') is not None:
                        # Restore the tracker but check if levels need updating
                        old_support = tracker.get('support_level', 0)
                        old_resistance = tracker.get('resistance_level', 0)
                        
                        # Check if restored levels protect enough profit
                        current_price = self.data_manager.get_current_price(self.symbol) or 0
                        entry_price = self.position_cost_basis / abs(self.position_size) if self.position_size != 0 else 0
                        position_value = abs(self.position_size) * current_price
                        min_profit_buffer = max(200, position_value * 0.005)
                        
                        # For LONG positions, check if support protects enough
                        if self.position == 1 and entry_price > 0:
                            min_support = entry_price + min_profit_buffer
                            if old_support < min_support:
                                self.logger.warning(f"🚨 Restored pivot level ${old_support:.0f} doesn't protect enough profit!")
                                self.logger.info(f"💰 Updating to ${min_support:.0f} to protect ${min_profit_buffer:.0f}")
                                old_support = min_support
                        
                        self.pivot_tracker = {
                            'recent_high': tracker.get('recent_high', 0),
                            'recent_low': tracker.get('recent_low', 0),
                            'last_update': datetime.now(),
                            'support_level': old_support,
                            'resistance_level': old_resistance,
                            'buffer_zone': tracker.get('buffer_zone', self.pivot_buffer),
                            'levels_locked': True,
                            'last_position_flip': tracker.get('last_position_flip')
                        }
                        
                        self.logger.info(f"🔒 PIVOT LEVELS SET: Support=${self.pivot_tracker['support_level']:.0f}, Resistance=${self.pivot_tracker['resistance_level']:.0f}")
                        return True
                        
        except Exception as e:
            self.logger.warning(f"Could not restore pivot tracker from resume file: {e}")
        
        return False
    
    def _serialize_pivot_tracker(self):
        """Serialize pivot tracker data for JSON storage, handling datetime objects."""
        if not hasattr(self, 'pivot_tracker') or not self.pivot_tracker:
            return {}
            
        tracker = self.pivot_tracker.copy()
        
        # Convert datetime objects to ISO format strings
        if 'last_update' in tracker and tracker['last_update']:
            if hasattr(tracker['last_update'], 'isoformat'):
                tracker['last_update'] = tracker['last_update'].isoformat()
        
        return tracker
    
    def update_trailing_pivot_protection(self, current_price):
        """
        Update pivot levels based on profit to lock in gains while allowing upside.
        Only moves levels favorably (up for LONG support, down for SHORT resistance).
        """
        if not hasattr(self, 'pivot_tracker') or not self.pivot_tracker:
            return False
            
        if not self.pivot_tracker.get('levels_locked', False):
            return False
            
        # Get entry price
        entry_price = self._get_entry_price()
        if not entry_price or entry_price <= 0:
            return False
            
        # Get profit tiers configuration
        profit_tiers = getattr(self, 'pivot_profit_tiers', [
            {'threshold': 0.05, 'protection_ratio': 0.70},
            {'threshold': 0.10, 'protection_ratio': 0.80},
            {'threshold': 0.15, 'protection_ratio': 0.85},
            {'threshold': 0.20, 'protection_ratio': 0.90}
        ])
        
        if self.position == 1:  # LONG position
            # Calculate profit percentage
            profit_pct = (current_price - entry_price) / entry_price
            
            if profit_pct <= 0:
                return False  # No profit to protect
                
            # Find applicable protection tier
            protection_ratio = 0
            for tier in sorted(profit_tiers, key=lambda x: x['threshold'], reverse=True):
                if profit_pct >= tier['threshold']:
                    protection_ratio = tier['protection_ratio']
                    break
                    
            if protection_ratio == 0:
                return False  # Below minimum threshold
                
            # Calculate new support level
            profit_per_unit = current_price - entry_price
            min_profit_to_keep = profit_per_unit * protection_ratio
            new_support = entry_price + min_profit_to_keep
            
            # Only raise support, never lower it
            current_support = self.pivot_tracker.get('support_level', 0)
            if new_support > current_support:
                # Find significant support level if configured
                if getattr(self, 'pivot_respect_technical_levels', True):
                    # Look for recent support in 24h data
                    lookback_hours = 24
                    df = self.data_manager.get_resampled_data()
                    if df is not None and len(df) > lookback_hours:
                        recent_data = df.iloc[-lookback_hours:]
                        recent_lows = recent_data['low'].values
                        
                        # Find technical support near our target
                        technical_support = new_support
                        for low in sorted(recent_lows, reverse=True):
                            if low > current_support and low <= new_support:
                                technical_support = low
                                break
                                
                        new_support = max(technical_support - (self.pivot_buffer / 2), current_support)
                
                # Update support level
                old_support = self.pivot_tracker['support_level']
                self.pivot_tracker['support_level'] = new_support
                self.pivot_tracker['profit_locked'] = new_support - entry_price
                self.pivot_tracker['protection_tier'] = f"{int(protection_ratio * 100)}%"
                
                self.logger.warning(f"📈 TRAILING PIVOT UPDATE: Support raised from ${old_support:.0f} to ${new_support:.0f}")
                self.logger.info(f"   Profit locked: ${self.pivot_tracker['profit_locked']:.0f} ({self.pivot_tracker['protection_tier']} of ${profit_per_unit:.0f} gain)")
                return True
                
        elif self.position == -1:  # SHORT position
            # Calculate profit percentage
            profit_pct = (entry_price - current_price) / entry_price
            
            if profit_pct <= 0:
                return False  # No profit to protect
                
            # Find applicable protection tier
            protection_ratio = 0
            for tier in sorted(profit_tiers, key=lambda x: x['threshold'], reverse=True):
                if profit_pct >= tier['threshold']:
                    protection_ratio = tier['protection_ratio']
                    break
                    
            if protection_ratio == 0:
                return False  # Below minimum threshold
                
            # Calculate new resistance level
            profit_per_unit = entry_price - current_price
            min_profit_to_keep = profit_per_unit * protection_ratio
            new_resistance = entry_price - min_profit_to_keep
            
            # Only lower resistance, never raise it
            current_resistance = self.pivot_tracker.get('resistance_level', float('inf'))
            if new_resistance < current_resistance:
                # Find significant resistance level if configured
                if getattr(self, 'pivot_respect_technical_levels', True):
                    # Look for recent resistance in 24h data
                    lookback_hours = 24
                    df = self.data_manager.get_resampled_data()
                    if df is not None and len(df) > lookback_hours:
                        recent_data = df.iloc[-lookback_hours:]
                        recent_highs = recent_data['high'].values
                        
                        # Find technical resistance near our target
                        technical_resistance = new_resistance
                        for high in sorted(recent_highs):
                            if high < current_resistance and high >= new_resistance:
                                technical_resistance = high
                                break
                                
                        new_resistance = min(technical_resistance + (self.pivot_buffer / 2), current_resistance)
                
                # Update resistance level
                old_resistance = self.pivot_tracker['resistance_level']
                self.pivot_tracker['resistance_level'] = new_resistance
                self.pivot_tracker['profit_locked'] = entry_price - new_resistance
                self.pivot_tracker['protection_tier'] = f"{int(protection_ratio * 100)}%"
                
                self.logger.warning(f"📉 TRAILING PIVOT UPDATE: Resistance lowered from ${old_resistance:.0f} to ${new_resistance:.0f}")
                self.logger.info(f"   Profit locked: ${self.pivot_tracker['profit_locked']:.0f} ({self.pivot_tracker['protection_tier']} of ${profit_per_unit:.0f} gain)")
                return True
                
        return False
    
    def _get_entry_price(self):
        """Get the current position's entry price."""
        if self.position == 1 and self.position_size > 0:
            return self.position_cost_basis / self.position_size
        elif self.position == -1 and self.position_size < 0:
            return self.position_cost_basis / abs(self.position_size)
        else:
            # Try to get from calculate_entry_price_from_trades
            calculated_price, _ = self.calculate_entry_price_from_trades()
            return calculated_price if calculated_price else self.last_trade_price
    
    def save_resume_state(self):
        """Save current position state to resume-auto-trade.json for easy restart."""
        import json
        import os
        from datetime import datetime
        
        try:
            # Get current position info
            status = self.get_status()
            position_info = status.get('position_info', {})
            current_price = self.data_manager.get_current_price(self.symbol) or 0.0
            
            # Calculate correct entry price from trades
            calculated_entry_price, position_trades = self.calculate_entry_price_from_trades()
            
            # Extract trade references
            trade_references = []
            if position_trades:
                for trade in position_trades:
                    trade_ref = {
                        'timestamp': trade['timestamp'],
                        'type': trade['type'],
                        'amount': trade['amount'],
                        'price': trade['price']
                    }
                    if 'trade_group_id' in trade:
                        trade_ref['trade_group_id'] = trade['trade_group_id']
                    trade_references.append(trade_ref)
            
            # Determine position type and amount
            if self.position == 1:  # LONG
                amount = self.balance_btc
                unit = 'btc'
                position_type = 'long'
                # Use calculated entry price from trades.json if available, otherwise fall back to position tracking
                if calculated_entry_price is not None:
                    entry_price = calculated_entry_price
                else:
                    entry_price = position_info.get('entry_price', self.last_trade_price or 0)
            elif self.position == -1:  # SHORT
                amount = self.balance_usd
                unit = 'usd'
                position_type = 'short'
                # Use calculated entry price from trades.json if available
                if calculated_entry_price is not None:
                    entry_price = calculated_entry_price
                elif self.position_size < 0:
                    entry_price = self.position_cost_basis / abs(self.position_size)
                else:
                    entry_price = position_info.get('entry_price', self.last_trade_price or 0)
            else:
                # Should not happen in this system
                return
                
            # Create resume data
            resume_data = {
                'timestamp': datetime.now().isoformat(),
                'position': position_type.upper(),
                'amount': round(amount, 8),
                'unit': unit,
                'entry_price': round(entry_price, 2),
                'current_price': round(current_price, 2),
                'unrealized_pnl': round(position_info.get('unrealized_pnl', 0), 2),
                'command': f"resume_auto_trade {amount:.8f}{unit} {position_type} {entry_price:.0f}",
                'strategy': {
                    'type': 'MACrossoverStrategy',
                    'short_window': self.short_window,
                    'long_window': self.long_window
                },
                'balances': {
                    'btc': round(self.balance_btc, 8),
                    'usd': round(self.balance_usd, 2)
                },
                'trades_executed': self.trades_executed,
                'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
                'trade_references': trade_references,
                'pivot_protection': {
                    'enabled': getattr(self, 'enable_pivot_protection', False),
                    'tracker': self._serialize_pivot_tracker()
                }
            }
            
            # Save to file
            resume_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'resume-auto-trade.json')
            with open(resume_file, 'w') as f:
                json.dump(resume_data, f, indent=2)
            
            self.logger.info(f"✅ Resume state saved: {position_type.upper()} {amount:.8f}{unit} @ ${entry_price:.2f}")
                
            # Also append to position history
            history_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'position-history.json')
            
            # Load existing history
            history = []
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                except:
                    history = []
            
            # Add current position to history
            history.append(resume_data)
            
            # Keep only last 100 entries to prevent file from growing too large
            if len(history) > 100:
                history = history[-100:]
            
            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
            self.logger.info(f"Saved resume state to {resume_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save resume state: {e}")

    def get_mark_to_market_values(self):
        """
        Returns total notional in USD and BTC, based on the current market price.
        """
        current_price = self.data_manager.get_current_price(self.symbol) or 0.0
        total_usd_value = self.balance_usd + (self.balance_btc * current_price)
        total_btc_value = self.balance_btc + \
            (self.balance_usd / current_price if current_price else 0.0)
        return total_usd_value, total_btc_value

    def _calculate_unrealized_pnl(self):
        """Calculate unrealized P&L for current position."""
        current_price = self.data_manager.get_current_price(self.symbol) or 0

        if self.position == 1 and self.position_size > 0:
            # Long position P&L
            return (self.position_size * current_price) - self.position_cost_basis
        elif self.position == -1:
            # Short position P&L - improved calculation
            if self.position_size < 0:
                # New method: position_size is negative for shorts
                btc_sold = abs(self.position_size)
                entry_price = self.position_cost_basis / btc_sold if btc_sold > 0 else 0
                return (entry_price - current_price) * btc_sold
            elif self.position_cost_basis > 0 and self.last_trade_price:
                # Fallback method using cost basis
                btc_equivalent = self.position_cost_basis / self.last_trade_price
                return (self.last_trade_price - current_price) * btc_equivalent
            else:
                # Final fallback - use USD balance difference
                initial_usd = getattr(self, 'initial_balance_usd', 0)
                if initial_usd > 0:
                    return self.balance_usd - initial_usd
        return 0

    def get_status(self):
        """
        Return a dictionary summarizing the current status, including 'position_info'
        that shows cost-basis-based entry price, position size, and unrealized PnL.
        """
        # Validate position tracking before building status
        if self.position == 1 and self.position_size > 0:
            avg_entry = self.position_cost_basis / self.position_size if self.position_size > 0 else 0
            current_price = self.data_manager.get_current_price(self.symbol) or 0
            
            # DISABLED: This sanity check was incorrectly resetting valid entry prices
            # # Sanity check: entry price shouldn't be more than 1.5x current price
            # if avg_entry > current_price * 1.5 and current_price > 0:
            #     self.logger.error(f"[SANITY_CHECK] Invalid entry price detected: ${avg_entry:.2f} vs current ${current_price:.2f}")
            #     self.logger.error(f"[SANITY_CHECK] Original values: position_size={self.position_size:.8f}, cost_basis=${self.position_cost_basis:.2f}")
            #     # Attempt to fix by recalculating based on current balance
            #     # Assume entry was 5% below current price as a reasonable estimate
            #     self.position_cost_basis = self.position_size * current_price * 0.95
            #     self.logger.error(f"[SANITY_CHECK] OVERRIDING! Reset position cost basis to ${self.position_cost_basis:.2f}")

        status = {
            'running': self.running,
            'position': self.position,
            'last_trade': None,
            'last_trade_data_source': None,
            'last_trade_signal_timestamp': None,
            'next_trigger': self.next_trigger,
            'current_trends': self.current_trends,
            'ma_difference': None,
            'ma_slope_difference': None,
            'initial_balance_btc': self.initial_balance_btc,
            'initial_balance_usd': self.initial_balance_usd,
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'balance_btc': self.balance_btc,
            'balance_usd': self.balance_usd,
            'total_return_pct': ((self.current_balance / self.initial_balance) - 1) * 100 if self.initial_balance != 0 else 0,
            'total_fees_paid': self.total_fees_paid,
            'trades_executed': self.trades_executed,
            'profitable_trades': self.profitable_trades,
            'win_rate': (self.profitable_trades / self.trades_executed * 100) if self.trades_executed else 0,
            'current_amount': self.current_amount,
            'total_profit_loss': self.total_profit_loss,
            'average_profit_per_trade': (self.total_profit_loss / self.trades_executed) if self.trades_executed else 0,
            'trade_count_today': self.trade_count_today,
            'remaining_trades_today': max(0, self.max_trades_per_day - self.trade_count_today),
            'theoretical_trade': self.theoretical_trade
        }

        if self.last_trade_reason:
            status['last_trade'] = self.last_trade_reason
            status['last_trade_data_source'] = self.last_trade_data_source
            if self.last_trade_signal_timestamp:
                status['last_trade_signal_timestamp'] = self.last_trade_signal_timestamp.strftime(
                    '%Y-%m-%d %H:%M:%S')

        if hasattr(self, 'df_ma') and not self.df_ma.empty:
            status['ma_difference'] = self.df_ma.iloc[-1]['Short_MA'] - \
                self.df_ma.iloc[-1]['Long_MA']
            if len(self.df_ma) >= 2:
                short_ma_slope = self.df_ma.iloc[-1]['Short_MA'] - \
                    self.df_ma.iloc[-2]['Short_MA']
                long_ma_slope = self.df_ma.iloc[-1]['Long_MA'] - \
                    self.df_ma.iloc[-2]['Long_MA']
                status['ma_slope_difference'] = short_ma_slope - long_ma_slope
                status['short_ma_momentum'] = 'Increasing' if short_ma_slope > 0 else 'Decreasing'
                status['long_ma_momentum'] = 'Increasing' if long_ma_slope > 0 else 'Decreasing'
                status['momentum_alignment'] = (
                    'Aligned' if (short_ma_slope > 0 and long_ma_slope > 0)
                    or (short_ma_slope < 0 and long_ma_slope < 0)
                    else 'Diverging'
                )

        if self.trades_executed > 0:
            status['average_fee_per_trade'] = self.total_fees_paid / \
                self.trades_executed
            status['risk_reward_ratio'] = (
                abs(self.total_profit_loss /
                    self.total_fees_paid) if self.total_fees_paid > 0 else 0
            )

        # Mark-to-market updates
        mtm_usd, mtm_btc = self.get_mark_to_market_values()
        status['mark_to_market_usd'] = mtm_usd
        status['mark_to_market_btc'] = mtm_btc

        if mtm_usd > self.max_mtm_usd:
            self.max_mtm_usd = mtm_usd
        if mtm_usd < self.min_mtm_usd:
            self.min_mtm_usd = mtm_usd

        status['max_balance_usd'] = self.max_balance_usd
        status['min_balance_usd'] = self.min_balance_usd
        status['max_balance_btc'] = self.max_balance_btc
        status['min_balance_btc'] = self.min_balance_btc
        status['max_mtm_usd'] = self.max_mtm_usd
        status['min_mtm_usd'] = self.min_mtm_usd

        # Build position_info
        position_info = {
            'current_price': self.data_manager.get_current_price(self.symbol) or 0.0,
            'entry_price': 0.0,
            'position_size_btc': 0.0,
            'position_size_usd': 0.0,
            'unrealized_pnl': 0.0,
        }

        # Fix position flag early if it's wrong
        if abs(self.balance_btc) < 1e-6 and self.balance_usd > 10000 and self.position == 0:
            self.position = -1  # We're short, holding USD
            self.logger.info(f"Corrected position flag to SHORT: holding ${self.balance_usd:.2f} USD, 0 BTC")
        elif self.balance_btc > 1e-6 and abs(self.balance_usd) < 1000 and self.position == 0:
            self.position = 1   # We're long, holding BTC
            self.logger.info(f"Corrected position flag to LONG: holding {self.balance_btc:.8f} BTC")

        cp = position_info['current_price']

        # Handle theoretical vs real trades properly
        if self.trades_executed == 0 and self.theoretical_trade:
            # THEORETICAL TRADE: Show what position should be worth
            entry_price = self.theoretical_trade.get('entry_price', 0.0)
            if self.theoretical_trade['direction'] == 'long':
                position_info['entry_price'] = entry_price
                position_info['position_size_btc'] = self.theoretical_trade['amount']
                position_info['position_size_usd'] = self.theoretical_trade['amount'] * cp
                position_info['unrealized_pnl'] = (cp - entry_price) * self.theoretical_trade['amount']
            else:  # short
                position_info['entry_price'] = entry_price  
                position_info['position_size_btc'] = 0.0
                position_info['position_size_usd'] = self.theoretical_trade['amount']
                btc_equivalent = self.theoretical_trade['amount'] / entry_price
                position_info['unrealized_pnl'] = (entry_price - cp) * btc_equivalent

        else:
            # REAL TRADES: Use actual balances and tracking
            # First try to get entry price from trades.json for consistency
            calculated_entry_price, _ = self.calculate_entry_price_from_trades()
            
            if self.position == 1:
                # Long position - holding BTC
                if calculated_entry_price is not None:
                    # Use the calculated entry price from trades.json
                    position_info['entry_price'] = calculated_entry_price
                    position_info['position_size_btc'] = self.balance_btc
                    position_info['position_size_usd'] = self.balance_btc * cp
                    position_info['unrealized_pnl'] = (self.balance_btc * cp) - (self.balance_btc * calculated_entry_price)
                    self.logger.info(f"[ENTRY_PRICE_DEBUG] Using trades.json entry price: ${calculated_entry_price:.2f}")
                elif self.position_size > 1e-8:
                    avg_entry_price = self.position_cost_basis / self.position_size
                    self.logger.info(f"[ENTRY_PRICE_DEBUG] Calculating entry price: cost_basis=${self.position_cost_basis:.2f} / size={self.position_size:.8f} = ${avg_entry_price:.2f}")
                    position_info['entry_price'] = avg_entry_price
                    position_info['position_size_btc'] = self.position_size
                    position_info['position_size_usd'] = self.position_size * cp
                    position_info['unrealized_pnl'] = (self.position_size * cp) - self.position_cost_basis
                else:
                    # Long position but position_size not tracked - use last trade price
                    if self.last_trade_price and self.last_trade_price > 0:
                        position_info['entry_price'] = self.last_trade_price
                        position_info['position_size_btc'] = self.balance_btc
                        position_info['position_size_usd'] = self.balance_btc * cp
                        # Calculate P&L based on last trade price
                        position_info['unrealized_pnl'] = (cp - self.last_trade_price) * self.balance_btc
                    else:
                        # Fallback: no entry price available
                        position_info['entry_price'] = 0.0
                        position_info['position_size_btc'] = self.balance_btc
                        position_info['position_size_usd'] = self.balance_btc * cp
                        position_info['unrealized_pnl'] = 0.0
                    
            elif self.position == -1:
                # Short position - properly calculate from stored position data
                position_info['position_size_btc'] = 0.0  # No BTC held
                position_info['position_size_usd'] = self.balance_usd  # USD from sale
                
                if calculated_entry_price is not None:
                    # Use the calculated entry price from trades.json
                    btc_sold = self.balance_usd / calculated_entry_price if calculated_entry_price > 0 else 0
                    position_info['entry_price'] = calculated_entry_price
                    position_info['unrealized_pnl'] = (calculated_entry_price - cp) * btc_sold
                    self.logger.info(f"[ENTRY_PRICE_DEBUG] Using trades.json entry price for SHORT: ${calculated_entry_price:.2f}")
                    
                    # Update position tracking if needed
                    if abs(self.position_size) < 1e-8 or self.position_cost_basis == 0:
                        self.logger.warning(f"[POSITION_DEBUG] Updating SHORT position tracking from trades.json: size={-btc_sold}, cost_basis={self.balance_usd}")
                        self.position_size = -btc_sold
                        self.position_cost_basis = self.balance_usd
                        self.last_trade_price = calculated_entry_price
                elif self.position_size < 0:  # We have a short position properly tracked
                    btc_sold = abs(self.position_size)
                    entry_price = self.position_cost_basis / btc_sold if btc_sold > 0 else 0
                    position_info['entry_price'] = entry_price
                    position_info['unrealized_pnl'] = (entry_price - cp) * btc_sold
                else:
                    # Fallback to last trade price
                    position_info['entry_price'] = self.last_trade_price or 0.0
                    position_info['unrealized_pnl'] = 0.0
            else:
                # This should never happen in this system
                self.logger.error("System in neutral position - this should not occur!")
                position_info['entry_price'] = 0.0
                position_info['position_size_btc'] = 0.0
                position_info['position_size_usd'] = 0.0
                position_info['unrealized_pnl'] = 0.0

        status['position_info'] = position_info

        ########################################################################
        # (A) Provide an MA-based measure of "how close" we are to crossing:
        ########################################################################
        if status['ma_difference'] is not None:
            short_val = self.df_ma.iloc[-1]['Short_MA']
            long_val = self.df_ma.iloc[-1]['Long_MA']
            avg_ma = (short_val + long_val) / \
                2.0 if (short_val + long_val) != 0 else 0.0
            if avg_ma != 0.0:
                status['ma_signal_proximity'] = abs(
                    short_val - long_val) / abs(avg_ma)
            else:
                status['ma_signal_proximity'] = None
        else:
            status['ma_signal_proximity'] = None
        ########################################################################

        return status

    def _log_successful_trade(self, trade_info):
        self.logger.info(
            f"Trade executed successfully: {trade_info.to_dict()}")

    def _log_failed_trade(self, trade_info):
        self.logger.info(f"Trade failed/canceled: {trade_info.to_dict()}")

    def track_trade_for_whipsaw(self, trade_type, price, timestamp):
        """Track trade and detect whipsaw patterns"""
        if not hasattr(self, 'whipsaw_tracker'):
            return
            
        # Add trade to history
        trade_entry = {
            'type': trade_type,
            'price': price,
            'timestamp': timestamp,
            'position': self.position
        }
        self.whipsaw_tracker['trades'].append(trade_entry)
        
        # Only keep recent trades (last 24 hours)
        cutoff_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') - timedelta(hours=24)
        self.whipsaw_tracker['trades'] = [
            t for t in self.whipsaw_tracker['trades'] 
            if datetime.strptime(t['timestamp'], '%Y-%m-%d %H:%M:%S') > cutoff_time
        ]
        
        # Detect whipsaws
        self._detect_whipsaws(timestamp)
        
    def _detect_whipsaws(self, current_timestamp):
        """Detect whipsaw patterns in recent trades"""
        trades = self.whipsaw_tracker['trades']
        if len(trades) < 3:
            return
        
        # Debug: Log trade sequence
        self.logger.debug(f"Checking {len(trades)} trades for whipsaws")
        for i, t in enumerate(trades[-5:]):  # Show last 5 trades
            self.logger.debug(f"  Trade {i}: {t['type']} at ${t['price']} ({t['timestamp']})")
            
        # Look for pattern: BUY -> SELL -> BUY or SELL -> BUY -> SELL
        # within detection window
        current_time = datetime.strptime(current_timestamp, '%Y-%m-%d %H:%M:%S')
        detection_window = timedelta(seconds=self.whipsaw_tracker['detection_window'])
        
        for i in range(len(trades) - 2):
            t1, t2, t3 = trades[i], trades[i+1], trades[i+2]
            
            # Check if trades form a whipsaw pattern
            if t1['type'] == t3['type'] and t1['type'] != t2['type']:
                t3_time = datetime.strptime(t3['timestamp'], '%Y-%m-%d %H:%M:%S')
                t1_time = datetime.strptime(t1['timestamp'], '%Y-%m-%d %H:%M:%S')
                
                if t3_time - t1_time <= detection_window:
                    # Calculate loss from whipsaw
                    if t1['type'] == 'buy':
                        # BUY -> SELL -> BUY: loss = (t1_price - t2_price) + (t3_price - t2_price)
                        loss = (t1['price'] - t2['price']) + (t3['price'] - t2['price'])
                    else:
                        # SELL -> BUY -> SELL: loss = (t2_price - t1_price) + (t2_price - t3_price)
                        loss = (t2['price'] - t1['price']) + (t2['price'] - t3['price'])
                    
                    whipsaw = {
                        'pattern': f"{t1['type']} -> {t2['type']} -> {t3['type']}",
                        'timestamps': [t1['timestamp'], t2['timestamp'], t3['timestamp']],
                        'prices': [t1['price'], t2['price'], t3['price']],
                        'loss': loss,
                        'duration': str(t3_time - t1_time)
                    }
                    
                    # Check if this whipsaw was already recorded
                    if not any(w['timestamps'] == whipsaw['timestamps'] for w in self.whipsaw_tracker['whipsaws']):
                        self.whipsaw_tracker['whipsaws'].append(whipsaw)
                        self.whipsaw_tracker['stats']['total_whipsaws'] += 1
                        self.whipsaw_tracker['stats']['whipsaw_losses'] += max(0, loss)
                        self.whipsaw_tracker['stats']['whipsaw_timeframes'].append((t3_time - t1_time).total_seconds())
                        
                        # Update average cost
                        if self.whipsaw_tracker['stats']['total_whipsaws'] > 0:
                            self.whipsaw_tracker['stats']['avg_whipsaw_cost'] = (
                                self.whipsaw_tracker['stats']['whipsaw_losses'] / 
                                self.whipsaw_tracker['stats']['total_whipsaws']
                            )
                        
                        self.logger.warning(f"⚡ Whipsaw detected: {whipsaw['pattern']} - Loss: ${loss:.2f}")
                        self.logger.info(f"  Timestamps: {whipsaw['timestamps']}")
                        self.logger.info(f"  Prices: ${t1['price']:.0f} → ${t2['price']:.0f} → ${t3['price']:.0f}")
    
    def force_pivot_recalculation(self):
        """Force immediate recalculation of pivot levels"""
        if hasattr(self, 'pivot_tracker') and self.pivot_tracker:
            self.pivot_tracker['levels_locked'] = False
            self.logger.info("🔓 Pivot levels unlocked for recalculation")
            return True
        return False
        
    def get_whipsaw_stats(self):
        """Get current whipsaw statistics"""
        if not hasattr(self, 'whipsaw_tracker'):
            return None
            
        stats = self.whipsaw_tracker['stats'].copy()
        
        # Add recent whipsaws
        recent_whipsaws = []
        cutoff = datetime.utcnow() - timedelta(hours=24)
        for w in self.whipsaw_tracker['whipsaws']:
            last_trade_time = datetime.strptime(w['timestamps'][-1], '%Y-%m-%d %H:%M:%S')
            if last_trade_time > cutoff:
                recent_whipsaws.append(w)
        
        stats['recent_whipsaws'] = recent_whipsaws
        stats['trades_last_24h'] = len(self.whipsaw_tracker['trades'])
        
        # Calculate whipsaw rate
        if len(self.whipsaw_tracker['trades']) >= 3:
            stats['whipsaw_rate'] = (stats['total_whipsaws'] * 3) / len(self.whipsaw_tracker['trades'])
        else:
            stats['whipsaw_rate'] = 0.0
            
        return stats
    
    def _load_recent_trades_for_whipsaw(self):
        """Load recent trades from trades.json for whipsaw tracking"""
        try:
            if os.path.exists(self.trade_log_file):
                all_trades = self._read_trades_jsonl()
                
                # Only load trades from last 24 hours
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                for trade in all_trades:
                    # Parse trade timestamp
                    trade_time_str = trade.get('timestamp', '')
                    if not trade_time_str:
                        continue
                        
                    try:
                        trade_time = datetime.strptime(trade_time_str, '%Y-%m-%d %H:%M:%S')
                    except:
                        continue
                    
                    # Only process recent trades
                    if trade_time > cutoff_time:
                        # Add to whipsaw tracker
                        trade_type = trade.get('type', '').lower()
                        price = float(trade.get('price', 0))
                        
                        if trade_type in ['buy', 'sell'] and price > 0:
                            self.track_trade_for_whipsaw(trade_type, price, trade_time_str)
                
                # Run detection on loaded trades
                if self.whipsaw_tracker['trades']:
                    self._detect_whipsaws(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
                    
                self.logger.info(f"Loaded {len(self.whipsaw_tracker['trades'])} recent trades for whipsaw tracking")
                
        except Exception as e:
            self.logger.error(f"Error loading trades for whipsaw tracking: {e}")


###############################################################################
class AdaptiveMultiStrategy(MACrossoverStrategy):
    """
    Adaptive strategy that switches between different trading approaches:

    TRENDING MARKETS: Uses MA crossovers (trend following)
    RANGING MARKETS: Uses mean reversion (RSI + Bollinger Bands)  
    VOLATILE MARKETS: Uses breakout strategies (MACD + volume)

    This way we're ALWAYS trading optimally instead of sitting out!
    """

    def __init__(self, *args, **kwargs):
        # Extract adaptive parameters
        self.regime_lookback = kwargs.pop('regime_lookback', 50)
        self.regime_switch_threshold = kwargs.pop(
            'regime_switch_threshold', 0.7)
        self.min_trade_gap_minutes = kwargs.pop('min_trade_gap_minutes', 30)
        self.rsi_oversold = kwargs.pop('rsi_oversold', 30)
        self.rsi_overbought = kwargs.pop('rsi_overbought', 70)
        self.bb_std_dev = kwargs.pop('bb_std_dev', 2.0)
        self.volume_threshold = kwargs.pop('volume_threshold', 1.5)
        self.macd_threshold = kwargs.pop('macd_threshold', 0.001)
        self.signal_confirmation_bars = kwargs.pop(
            'signal_confirmation_bars', 2)
        
        # Pivot protection parameters
        self.pivot_buffer = kwargs.pop('pivot_buffer', 100)  # Buffer zone in dollars
        self.pivot_lookback_hours = kwargs.pop('pivot_lookback_hours', 2)  # Hours to look back for pivots
        self.enable_pivot_protection = kwargs.pop('enable_pivot_protection', True)  # Can disable if needed
        
        # Trailing pivot parameters (new)
        self.enable_trailing_pivots = kwargs.pop('enable_trailing_pivots', False)
        self.pivot_profit_tiers = kwargs.pop('pivot_profit_tiers', [
            {'threshold': 0.05, 'protection_ratio': 0.70},
            {'threshold': 0.10, 'protection_ratio': 0.80},
            {'threshold': 0.15, 'protection_ratio': 0.85},
            {'threshold': 0.20, 'protection_ratio': 0.90}
        ])
        self.pivot_respect_technical_levels = kwargs.pop('pivot_respect_technical_levels', True)

        # Initialize parent class
        super().__init__(*args, **kwargs)

        # Add startup grace period
        self.startup_time = datetime.now()
        self.startup_grace_period_minutes = 5  # No trades for first 5 minutes

        # Log adaptive strategy initialization
        self.diagnostic_logger.log_event("ADAPTIVE_STRATEGY_INIT", {
            "strategy": "ADAPTIVE_MULTI",
            "parameters": {"regime_threshold": self.regime_switch_threshold,
                          "signal_confirmation_bars": self.signal_confirmation_bars,
                          "min_trade_gap_minutes": self.min_trade_gap_minutes}
        })

        # Adaptive state
        self.current_regime = "unknown"
        self.regime_confidence = 0.0
        self.active_strategy = "trending"  # trending, ranging, volatile
        self.strategy_switches_today = 0
        
        # Try to restore pivot tracker from resume file if available
        self._restore_pivot_tracker_from_resume()
        self.signal_history = []
        self.last_confirmed_signal = 0

        # Add time-based strategy switch constraint
        self.last_strategy_switch_time = None
        self.min_strategy_switch_minutes = 120  # 2 hours minimum between switches

        # Ensure critical attributes exist (fix AttributeError)
        if not hasattr(self, 'last_trade_time'):
            self.last_trade_time = None

        # Performance tracking by strategy
        self.strategy_performance = {
            "trending": {"trades": 0, "profit": 0.0},
            "ranging": {"trades": 0, "profit": 0.0},
            "volatile": {"trades": 0, "profit": 0.0}
        }

        self.logger.info(f"🎯 ADAPTIVE MULTI-STRATEGY INITIALIZED")
        self.logger.info(
            f"   Will switch between TRENDING → RANGING → VOLATILE strategies")
        self.logger.info(
            f"   Regime lookback: {self.regime_lookback}, Gap: {self.min_trade_gap_minutes}min")
        
        # Initialize System Verifier for continuous regression detection
        self.system_verifier = None
        self.last_verification_time = None
        self.verification_interval = 30  # seconds
        try:
            from tdr_core.system_verifier import SystemVerifier
            self.system_verifier = SystemVerifier(self, data_manager, logger)
            self.logger.info("✅ System Verifier initialized for continuous regression detection")
        except Exception as e:
            self.logger.warning(f"Could not initialize System Verifier: {e}")

        # BUG FIX: Use actual config parameters instead of hardcoded values
        self.logger.info(f"   Confidence threshold: {self.regime_switch_threshold:.1%}")
        self.logger.info(f"   Signal confirmation: {self.signal_confirmation_bars} bars")
        self.logger.info(f"   Position-aware switching: $50k+ requires higher confidence")

    def validate_position_tracking(self):
        """Validate and correct position tracking inconsistencies."""
        current_price = self.data_manager.get_current_price(self.symbol) or 0
        
        # Determine actual position from balances
        # CRITICAL: In a 100% flip system, we're either LONG (holding BTC) or SHORT (holding USD)
        actual_position = 0
        
        # Check BTC balance first - if we have BTC, we're LONG
        if self.balance_btc > 0.0001:  # More than dust amount
            actual_position = 1  # LONG
            # Even if we also have USD, having BTC means we're LONG
        elif self.balance_usd > 100 and self.balance_btc < 0.0001:  
            # Only SHORT if we have USD AND no BTC
            actual_position = -1  # SHORT
        else:
            # This should rarely happen in a 100% position system
            self.logger.warning(f"Unusual balance state: BTC={self.balance_btc}, USD={self.balance_usd}")
            actual_position = 0
        
        # Check if position flag matches actual holdings
        if self.position != actual_position:
            self.logger.warning(f"🚨 POSITION MISMATCH DETECTED!")
            self.logger.warning(f"  Position flag: {self.position} ({'LONG' if self.position == 1 else 'SHORT' if self.position == -1 else 'NEUTRAL'})")
            self.logger.warning(f"  Actual holdings: BTC={self.balance_btc:.8f}, USD=${self.balance_usd:.2f}")
            self.logger.warning(f"  Actual position: {actual_position} ({'LONG' if actual_position == 1 else 'SHORT' if actual_position == -1 else 'NEUTRAL'})")
            
            # Auto-correct the position flag
            old_position = self.position
            self.position = actual_position
            
            # Log the correction
            self.diagnostic_logger.log_position_anomaly(
                f"Auto-corrected position flag from {old_position} to {actual_position}",
                {
                    "balance_btc": self.balance_btc, 
                    "balance_usd": self.balance_usd,
                    "old_position": old_position,
                    "new_position": actual_position,
                    "position_size": self.position_size,
                    "cost_basis": self.position_cost_basis
                }
            )
            
            self.logger.info(f"✅ Position flag corrected from {old_position} to {actual_position}")
            
            # Also validate position size matches
            if actual_position == 1 and abs(self.position_size - self.balance_btc) > 0.0001:
                self.logger.warning(f"Position size mismatch: {self.position_size} vs actual {self.balance_btc}")
                self.position_size = self.balance_btc
            elif actual_position == -1 and self.position_size >= 0:
                # For SHORT positions, position_size should be negative
                # Don't change position_size if we're already tracking a short position
                if abs(self.position_size) < 0.0001:
                    # Only set to 0 if we truly have no position tracked
                    self.position_size = 0
                    self.logger.warning("SHORT position but no position size tracked - may need to validate from trades")
        
        # Additional validation: ensure position_size sign matches position
        if self.position == 1 and self.position_size < 0:
            self.logger.warning(f"LONG position but negative size: {self.position_size}")
            self.position_size = abs(self.position_size)
        elif self.position == -1 and self.position_size > 0:
            self.logger.warning(f"SHORT position but positive size: {self.position_size}")
            self.position_size = -abs(self.position_size)
            
        # Log current validated state
        self.logger.debug(f"Position validation complete: pos={self.position}, btc={self.balance_btc:.8f}, usd=${self.balance_usd:.2f}, size={self.position_size}")
        
        return actual_position
        
        # Validate short position tracking
        if self.position == -1 and self.position_size >= 0 and self.balance_usd > 50000:
            # Short position should have negative position_size or proper cost basis
            if self.position_cost_basis == 0 and self.last_trade_price:
                self.logger.warning("Short position missing cost basis, attempting to reconstruct")
                # Estimate based on USD balance and last trade price
                estimated_btc_sold = self.balance_usd / self.last_trade_price
                self.position_size = -estimated_btc_sold
                self.position_cost_basis = self.balance_usd
                self.diagnostic_logger.log_position_anomaly(
                    "Reconstructed short position tracking",
                    {"estimated_btc_sold": estimated_btc_sold, "entry_price": self.last_trade_price}
                )

    def detect_market_regime(self, df):
        """Detect if market is trending, ranging, or volatile."""
        if len(df) < self.regime_lookback:
            return "unknown", 0.0, {}

        recent_df = df.tail(self.regime_lookback).copy()

        # Calculate market metrics
        price_start = recent_df['close'].iloc[0]
        price_end = recent_df['close'].iloc[-1]
        price_high = recent_df['close'].max()
        price_low = recent_df['close'].min()

        total_return = abs(price_end - price_start) / price_start
        price_range = (price_high - price_low) / ((price_high + price_low) / 2)
        trend_strength = total_return / price_range if price_range > 0 else 0

        # Volatility
        returns = recent_df['close'].pct_change().dropna()
        volatility = returns.std() if len(returns) > 1 else 0

        # Whipsaw detection
        df_temp = add_moving_averages(
            recent_df.copy(), self.short_window, self.long_window, price_col='close')
        df_temp = generate_ma_signals(df_temp)
        signal_changes = (df_temp['MA_Signal'].diff() != 0).sum()
        whipsaw_ratio = (signal_changes / len(df_temp)) * 100

        # Range-bound detection
        ma_20 = recent_df['close'].rolling(20).mean()
        price_vs_ma = (recent_df['close'] - ma_20) / ma_20
        range_bound_score = 1.0 - \
            abs(price_vs_ma.mean()) if not price_vs_ma.empty else 0

        # Regime scoring
        regime_scores = {'trending': 0.0, 'ranging': 0.0, 'volatile': 0.0}

        # TRENDING indicators (more conservative)
        if trend_strength > 0.3:  # Reduced from 0.4 - more reasonable
            regime_scores['trending'] += 2.0
        if whipsaw_ratio < 2.0:  # Lower whipsaw tolerance
            regime_scores['trending'] += 2.0
        if volatility < 0.02:  # Increased from 0.015 - more reasonable
            regime_scores['trending'] += 1.5
        # Require sustained directional movement
        if len(recent_df) >= 10:
            recent_closes = recent_df['close'].tail(10)
            directional_moves = 0
            for i in range(1, len(recent_closes)):
                if (recent_closes.iloc[i] > recent_closes.iloc[i-1] and price_end > price_start) or \
                   (recent_closes.iloc[i] < recent_closes.iloc[i-1] and price_end < price_start):
                    directional_moves += 1
            if directional_moves >= 6:  # Reduced from 7 - 60% directional consistency
                regime_scores['trending'] += 1.0

        # RANGING indicators
        if range_bound_score > 0.8:
            regime_scores['ranging'] += 2.0
        if whipsaw_ratio > 8.0:  # Increased from 4.0 to prevent false ranging detection
            regime_scores['ranging'] += 2.0
        if trend_strength < 0.15:  # Tightened from 0.2
            regime_scores['ranging'] += 1.0

        # VOLATILE indicators
        if volatility > 0.035:
            regime_scores['volatile'] += 2.0
        if price_range > 0.08:
            regime_scores['volatile'] += 1.0

        # Determine regime
        best_regime = max(regime_scores.items(), key=lambda x: x[1])
        regime = best_regime[0]
        max_score = best_regime[1]
        # Normalize to max possible score
        confidence = min(0.95, max_score / 6.0)

        # Create metrics dict first
        metrics = {'whipsaw_ratio': whipsaw_ratio, 'trend_strength': trend_strength, 'volatility': volatility}
        
        self.logger.debug(
            f"📊 Regime Scores: TRENDING={regime_scores['trending']:.1f}, RANGING={regime_scores['ranging']:.1f}, VOLATILE={regime_scores['volatile']:.1f}")
        self.logger.debug(f"📈 Market Metrics: whipsaw={metrics.get('whipsaw_ratio', 0):.1f}%, trend_strength={metrics.get('trend_strength', 0):.3f}, volatility={volatility:.4f}")
        self.logger.debug(f"🎯 Final: {regime.upper()} (confidence: {confidence:.1%})")

        return regime, confidence, metrics

    def generate_ranging_signal(self, df):
        """Generate mean reversion signals for ranging markets."""
        if len(df) < 20:
            return 0, "Insufficient data for ranging strategy"

        # Calculate RSI
        df_rsi = calculate_rsi(df.copy(), window=14, price_col='close')
        current_rsi = df_rsi.iloc[-1]['RSI']

        # Calculate Bollinger Bands
        df_bb = calculate_bollinger_bands(
            df.copy(), window=20, num_std=self.bb_std_dev, price_col='close')
        current_price = df_bb.iloc[-1]['close']
        bb_upper = df_bb.iloc[-1]['BB_Upper']
        bb_lower = df_bb.iloc[-1]['BB_Lower']
        bb_middle = df_bb.iloc[-1]['BB_MA']

        signal = 0
        reason = "No mean reversion signal"

        # ENHANCEMENT: Add volatility-based position sizing for ranging markets
        current_price = df.iloc[-1]['close']
        volatility = df['close'].pct_change().tail(10).std()
        
        # Dynamic thresholds based on recent volatility
        dynamic_oversold = self.rsi_oversold + (10 * volatility) if volatility > 0.02 else self.rsi_oversold
        dynamic_overbought = self.rsi_overbought - (10 * volatility) if volatility > 0.02 else self.rsi_overbought
        
        # ENHANCEMENT: Add price momentum filter
        price_momentum = (current_price - df['close'].rolling(5).mean().iloc[-1]) / df['close'].rolling(5).mean().iloc[-1]
        strong_momentum = abs(price_momentum) > 0.01  # 1% momentum threshold
        
        self.logger.debug(f"Ranging strategy: RSI={current_rsi:.1f}, dynamic_oversold={dynamic_oversold:.1f}, dynamic_overbought={dynamic_overbought:.1f}")
        self.logger.debug(f"Price momentum: {price_momentum:.3f}, strong={strong_momentum}")

        # Mean reversion logic - BUY OVERSOLD, SELL OVERBOUGHT
        if current_rsi < self.rsi_oversold and current_price < bb_lower:
            signal = 1  # Oversold = BUY
            reason = f"Mean Reversion BUY: RSI {current_rsi:.1f} oversold + below BB"
        elif current_rsi > self.rsi_overbought and current_price > bb_upper:
            signal = -1  # Overbought = SELL
            reason = f"Mean Reversion SELL: RSI {current_rsi:.1f} overbought + above BB"
        # ENHANCEMENT: Additional entry with dynamic thresholds and momentum filter
        elif current_rsi < dynamic_oversold and not strong_momentum:
            signal = 1
            reason = f"Dynamic Mean Reversion BUY: RSI {current_rsi:.1f} < {dynamic_oversold:.1f}, low momentum"
        elif current_rsi > dynamic_overbought and not strong_momentum:
            signal = -1
            reason = f"Dynamic Mean Reversion SELL: RSI {current_rsi:.1f} > {dynamic_overbought:.1f}, low momentum"
        elif current_rsi < 40 and current_price < bb_middle * 0.99:  # Additional entry condition
            signal = 1
            reason = f"Mean Reversion BUY: RSI {current_rsi:.1f} low + below BB middle"
        elif self.position != 0:
            # Exit positions when price reaches the OPPOSITE band for maximum profit
            if self.position == 1:  # Currently LONG (bought at bottom)
                # SELL when price reaches upper band or RSI is overbought
                if current_price >= bb_upper * 0.995:  # Near upper band
                    signal = -1
                    reason = f"Mean Reversion SELL: Price at upper BB (${current_price:.0f})"
                elif current_rsi > self.rsi_overbought:
                    signal = -1
                    reason = f"Mean Reversion SELL: RSI {current_rsi:.1f} overbought"
            elif self.position == -1:  # Currently SHORT (sold at top)
                # BUY when price reaches lower band or RSI is oversold
                if current_price <= bb_lower * 1.005:  # Near lower band
                    signal = 1
                    reason = f"Mean Reversion BUY: Price at lower BB (${current_price:.0f})"
                elif current_rsi < self.rsi_oversold:
                    signal = 1
                    reason = f"Mean Reversion BUY: RSI {current_rsi:.1f} oversold"

        return signal, reason

    def generate_trending_signal(self, df):
        """Generate MA crossover signals for trending markets."""
        if len(df) < self.long_window:
            return 0, "Insufficient data for MA"

        df_ma = add_moving_averages(
            df.copy(), self.short_window, self.long_window, price_col='close')
        df_ma = generate_ma_signals(df_ma)

        signal = df_ma.iloc[-1]['MA_Signal']
        reason = f"MA Crossover: {'LONG' if signal == 1 else 'SHORT' if signal == -1 else 'NEUTRAL'}"

        return signal, reason

    def generate_volatile_signal(self, df):
        """Generate breakout signals for volatile markets."""
        if len(df) < 26:
            return 0, "Insufficient data for volatile strategy"

        df_macd = calculate_macd(
            df.copy(), fast=12, slow=26, signal=9, price_col='close')
        df_macd = generate_macd_signals(df_macd)

        macd_signal = df_macd.iloc[-1]['MACD_Signal']
        current_macd = df_macd.iloc[-1]['MACD']

        signal = 0
        reason = "No breakout signal"

        # Breakout logic with MACD
        if macd_signal == 1 and abs(current_macd) > self.macd_threshold:
            signal = 1
            reason = f"Breakout LONG: MACD bullish crossover"
        elif macd_signal == -1 and abs(current_macd) > self.macd_threshold:
            signal = -1
            reason = f"Breakout SHORT: MACD bearish crossover"

        return signal, reason

    def confirm_signal(self, current_signal):
        """Enhanced signal confirmation with position-aware requirements."""

        # Clear history if signal changes
        if self.signal_history and self.signal_history[-1] != current_signal:
            self.signal_history = []  # Reset on signal change

        self.signal_history.append(current_signal)

        # Keep only recent signals (max 5)
        if len(self.signal_history) > 5:
            self.signal_history = self.signal_history[-5:]

        # Require more confirmation when holding positions
        current_position_value = abs(self.balance_btc * (self.data_manager.get_current_price(self.symbol) or 0)) + self.balance_usd

        # OPTIMIZATION: Adaptive confirmation based on market regime and position
        base_bars = self.signal_confirmation_bars
        
        # Reduce confirmation in trending markets, increase in ranging
        if self.current_regime == "trending":
            required_bars = max(1, base_bars - 1)  # Faster in trends
        elif self.current_regime == "ranging":
            required_bars = base_bars + 1  # More cautious in ranging
        else:
            required_bars = base_bars
        
        if len(self.signal_history) < required_bars:
            return False

        recent_signals = self.signal_history[-required_bars:]

        # All signals must be consistent
        if not all(s == recent_signals[0] for s in recent_signals):
            return False

        # Don't re-confirm the same signal
        if recent_signals[0] == 0:  # No signal
            return False

        # Check if we've already confirmed this signal recently
        if recent_signals[0] == getattr(self, 'last_confirmed_signal', None):
            return False

        self.last_confirmed_signal = recent_signals[0]
        return True

    def check_trade_gap(self):
        """Check if enough time passed since last trade."""
        if self.last_trade_time is None:
            return True

        minutes_since = (datetime.now() -
                         self.last_trade_time).total_seconds() / 60
        return minutes_since >= self.min_trade_gap_minutes

    def run_strategy_loop(self):
        """Main adaptive strategy loop."""
        evaluation_count = 0
        last_evaluation_log = datetime.now()
        last_successful_eval = datetime.now()
        
        while self.running:
            evaluation_count += 1
            current_time = datetime.now()
            
            # Log evaluation frequency every 5 evaluations or every 5 minutes
            if evaluation_count % 5 == 0 or (current_time - last_evaluation_log).total_seconds() > 300:
                mins_since_last = (current_time - last_successful_eval).total_seconds() / 60
                self.logger.info(f"📊 Adaptive strategy evaluation #{evaluation_count} at {current_time.strftime('%Y-%m-%d %H:%M:%S')} (last successful: {mins_since_last:.1f}min ago)")
                last_evaluation_log = current_time
            
            # Add position validation and save resume state every 10 minutes
            if not hasattr(self, '_last_validation') or \
               (datetime.now() - self._last_validation).total_seconds() > 600:
                self.validate_position_tracking()
                self._last_validation = datetime.now()
                
                # Save resume state periodically
                try:
                    self.save_resume_state()
                except Exception as e:
                    self.logger.error(f"Failed to save resume state: {e}")
            
            # Run System Verifier checks every 30 seconds
            if self.system_verifier and (self.last_verification_time is None or 
                                        (current_time - self.last_verification_time).total_seconds() >= self.verification_interval):
                try:
                    verification_results = self.system_verifier.run_all_checks()
                    self.last_verification_time = current_time
                    
                    # Log errors if any found
                    if verification_results.get('errors'):
                        self.logger.error(f"🚨 SYSTEM VERIFIER DETECTED {len(verification_results['errors'])} ISSUES!")
                        for error in verification_results['errors']:
                            self.logger.error(f"  ❌ {error}")
                except Exception as e:
                    self.logger.error(f"System Verifier failed: {e}")

            df = self.data_manager.get_price_dataframe(self.symbol)
            if not df.empty:
                try:
                    df = ensure_datetime_index(df)
                    df_resampled = df.resample('1H').agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                        'volume': 'sum', 'trades': 'sum', 'timestamp': 'last', 'source': 'last'
                    }).dropna()

                    if len(df_resampled) >= self.long_window:
                        # 0. Check for emergency exit conditions
                        status = self.get_status()
                        position_info = status.get('position_info', {})
                        unrealized_pnl = position_info.get('unrealized_pnl', 0)
                        
                        # Emergency exit if losing more than $2000
                        if self.position != 0 and unrealized_pnl < -2000:
                            self.logger.warning(f"🚨 EMERGENCY EXIT: Unrealized loss ${unrealized_pnl:.2f} exceeds threshold")
                            emergency_signal = -self.position  # Flip position
                            emergency_reason = f"Emergency exit: loss ${unrealized_pnl:.2f}"
                            
                            # Force immediate execution by bypassing confirmation
                            current_price = df_resampled.iloc[-1]['close']
                            signal_time = df_resampled.index[-1]
                            self.signal_history = [emergency_signal] * self.signal_confirmation_bars  # Force confirmation
                            self.check_for_signals(emergency_signal, current_price, signal_time)
                            continue
                        
                        # 0.5 Check for pivot-based quick flips (Support/Resistance)
                        if self.enable_pivot_protection:
                            current_price = self.data_manager.get_current_price(self.symbol) or df_resampled.iloc[-1]['close']
                            
                            # Skip pivot protection during startup grace period to prevent phantom trades
                            grace_period_active = False
                            if hasattr(self, 'startup_time') and hasattr(self, 'startup_grace_period_minutes'):
                                if (datetime.now() - self.startup_time).total_seconds() < (self.startup_grace_period_minutes * 60):
                                    grace_period_active = True
                                    self.logger.debug("Skipping pivot protection during startup grace period")
                            
                            # Initialize pivot tracking if not exists
                            if not hasattr(self, 'pivot_tracker'):
                                self.pivot_tracker = {
                                    'recent_high': current_price,
                                    'recent_low': current_price,
                                    'last_update': datetime.now(),
                                    'support_level': None,
                                    'resistance_level': None,
                                    'buffer_zone': self.pivot_buffer,  # Configurable buffer
                                    'levels_locked': False,  # Track if levels are established
                                    'last_position_flip': None  # Track when we last flipped position
                                }
                        
                            # Skip all pivot operations during grace period
                            if not grace_period_active:
                                # Update pivot levels (look at configurable hours of data)
                                lookback_hours = min(self.pivot_lookback_hours, len(df_resampled))
                                recent_data = df_resampled.iloc[-lookback_hours:] if lookback_hours > 0 else df_resampled
                                recent_high = recent_data['high'].max()
                                recent_low = recent_data['low'].min()
                                
                                # Store in tracker for status display
                                self.pivot_tracker['recent_high'] = recent_high
                                self.pivot_tracker['recent_low'] = recent_low
                                
                                # Update support/resistance based on recent price action
                                if self.position == 1:  # LONG position
                                    # Check if we need to establish new levels (after position flip or first time)
                                    if (not self.pivot_tracker['levels_locked'] or 
                                        self.pivot_tracker['last_position_flip'] != self.position):
                                        # Set sticky support level based on recent low
                                        calculated_support = recent_low - (self.pivot_buffer / 2)
                                        
                                        # For profit protection: ensure support is above entry price if we're profitable
                                        entry_price = self.position_cost_basis / abs(self.position_size) if self.position_size != 0 else 0
                                        if entry_price > 0 and current_price > entry_price:
                                            # We're profitable - ensure we lock in at least break-even
                                            # Calculate profit buffer based on position size
                                            position_value = abs(self.position_size) * current_price
                                            # Use 0.5% of position value or $200, whichever is larger
                                            min_profit_buffer = max(200, position_value * 0.005)
                                            profit_support = entry_price + min_profit_buffer
                                            self.pivot_tracker['support_level'] = max(calculated_support, profit_support)
                                            
                                            # Log if we're using profit protection
                                            if self.pivot_tracker['support_level'] == profit_support:
                                                self.logger.info(f"💰 Profit Protection Active: Support raised to ${profit_support:.0f} (entry: ${entry_price:.0f})")
                                        else:
                                            self.pivot_tracker['support_level'] = calculated_support
                                            
                                        self.pivot_tracker['resistance_level'] = recent_high + (self.pivot_buffer / 2)
                                        self.pivot_tracker['levels_locked'] = True
                                        self.pivot_tracker['last_position_flip'] = self.position
                                        self.logger.info(f"📍 LONG Pivot Levels Established - Support: ${self.pivot_tracker['support_level']:.0f}, "
                                                       f"Resistance: ${self.pivot_tracker['resistance_level']:.0f}")
                                    
                                    # Check if current levels protect profit - if not, recalculate
                                    entry_price = self.position_cost_basis / abs(self.position_size) if self.position_size != 0 else 0
                                    if entry_price > 0 and current_price > entry_price:
                                        # We're profitable - ensure support protects profit
                                        # Calculate profit buffer based on position size
                                        position_value = abs(self.position_size) * current_price
                                        # Use 0.5% of position value or $200, whichever is larger
                                        min_profit_buffer = max(200, position_value * 0.005)
                                        min_support = entry_price + min_profit_buffer
                                        if self.pivot_tracker['support_level'] < min_support:
                                            self.logger.warning(f"⚠️ Current support ${self.pivot_tracker['support_level']:.0f} doesn't protect profit!")
                                            self.logger.info(f"💰 Raising support to ${min_support:.0f} to protect ${min_profit_buffer} profit")
                                            self.pivot_tracker['support_level'] = min_support
                                            # CRITICAL FIX: Unlock levels so they can be recalculated
                                            self.pivot_tracker['levels_locked'] = False
                                    
                                    # Check if price broke below support
                                    if current_price < self.pivot_tracker['support_level']:
                                        # Check if we're still in startup grace period
                                        if hasattr(self, 'startup_time') and hasattr(self, 'startup_grace_period_minutes'):
                                            if (datetime.now() - self.startup_time).total_seconds() < (self.startup_grace_period_minutes * 60):
                                                self.logger.info(f"⏳ Pivot break detected during grace period - IGNORING to prevent phantom trade")
                                                continue
                                        
                                        self.logger.warning(f"🚨 PIVOT BREAK: Price ${current_price:.0f} broke support ${self.pivot_tracker['support_level']:.0f}")
                                        pivot_signal = -1  # Flip to SHORT
                                        pivot_reason = f"Pivot break: below support ${self.pivot_tracker['support_level']:.0f}"
                                        
                                        # Reset levels for next position
                                        self.pivot_tracker['levels_locked'] = False
                                        
                                        # Force immediate execution
                                        signal_time = df_resampled.index[-1]
                                        self.signal_history = [pivot_signal] * self.signal_confirmation_bars
                                        self.last_trade_reason = pivot_reason
                                        self.logger.warning(f"🎯 PIVOT PROTECTION TRIGGERED: {pivot_reason}")
                                        self.check_for_signals(pivot_signal, current_price, signal_time)
                                        continue
                                    
                                elif self.position == -1:  # SHORT position
                                    # Check if we need to establish new levels (after position flip or first time)
                                    if (not self.pivot_tracker['levels_locked'] or 
                                        self.pivot_tracker['last_position_flip'] != self.position):
                                        # Set sticky levels based on recent high/low
                                        self.pivot_tracker['support_level'] = recent_low - (self.pivot_buffer / 2)
                                        calculated_resistance = recent_high + (self.pivot_buffer / 2)
                                        
                                        # For profit protection: ensure resistance is below entry price if we're profitable
                                        entry_price = self.last_trade_price  # For SHORT, entry is the SELL price
                                        if entry_price > 0 and current_price < entry_price:
                                            # We're profitable - ensure we lock in at least break-even
                                            # Calculate profit buffer based on position size
                                            position_value = abs(self.position_size) * current_price
                                            # Use 0.5% of position value or $200, whichever is larger
                                            min_profit_buffer = max(200, position_value * 0.005)
                                            profit_resistance = entry_price - min_profit_buffer
                                            self.pivot_tracker['resistance_level'] = min(calculated_resistance, profit_resistance)
                                            
                                            # Log if we're using profit protection
                                            if self.pivot_tracker['resistance_level'] == profit_resistance:
                                                self.logger.info(f"💰 Profit Protection Active: Resistance lowered to ${profit_resistance:.0f} (entry: ${entry_price:.0f})")
                                        else:
                                            self.pivot_tracker['resistance_level'] = calculated_resistance
                                            
                                        self.pivot_tracker['levels_locked'] = True
                                        self.pivot_tracker['last_position_flip'] = self.position
                                        self.logger.info(f"📍 SHORT Pivot Levels Established - Support: ${self.pivot_tracker['support_level']:.0f}, "
                                                       f"Resistance: ${self.pivot_tracker['resistance_level']:.0f}")
                                    
                                    # Check if current levels protect profit - if not, recalculate
                                    entry_price = self.last_trade_price  # For SHORT, entry is the SELL price
                                    if entry_price > 0 and current_price < entry_price:
                                        # We're profitable - ensure resistance protects profit
                                        # Calculate profit buffer based on position size
                                        position_value = abs(self.position_size) * current_price
                                        # Use 0.5% of position value or $200, whichever is larger
                                        min_profit_buffer = max(200, position_value * 0.005)
                                        max_resistance = entry_price - min_profit_buffer
                                        if self.pivot_tracker['resistance_level'] > max_resistance:
                                            self.logger.warning(f"⚠️ Current resistance ${self.pivot_tracker['resistance_level']:.0f} doesn't protect profit!")
                                            self.logger.info(f"💰 Lowering resistance to ${max_resistance:.0f} to protect ${min_profit_buffer} profit")
                                            self.pivot_tracker['resistance_level'] = max_resistance
                                    
                                    # Check if price broke above resistance
                                    if current_price > self.pivot_tracker['resistance_level']:
                                        # Check if we're still in startup grace period
                                        if hasattr(self, 'startup_time') and hasattr(self, 'startup_grace_period_minutes'):
                                            if (datetime.now() - self.startup_time).total_seconds() < (self.startup_grace_period_minutes * 60):
                                                self.logger.info(f"⏳ Pivot break detected during grace period - IGNORING to prevent phantom trade")
                                                continue
                                        
                                        self.logger.warning(f"🚨 PIVOT BREAK: Price ${current_price:.0f} broke resistance ${self.pivot_tracker['resistance_level']:.0f}")
                                        pivot_signal = 1  # Flip to LONG
                                        pivot_reason = f"Pivot break: above resistance ${self.pivot_tracker['resistance_level']:.0f}"
                                        
                                        # Reset levels for next position
                                        self.pivot_tracker['levels_locked'] = False
                                        
                                        # Force immediate execution
                                        signal_time = df_resampled.index[-1]
                                        self.signal_history = [pivot_signal] * self.signal_confirmation_bars
                                        self.last_trade_reason = pivot_reason
                                        self.logger.warning(f"🎯 PIVOT PROTECTION TRIGGERED: {pivot_reason}")
                                        self.check_for_signals(pivot_signal, current_price, signal_time)
                                        continue
                            
                                # Update trailing pivot protection if enabled
                                if getattr(self, 'enable_trailing_pivots', True):
                                    self.update_trailing_pivot_protection(current_price)
                                
                                # Log pivot levels periodically
                                if not hasattr(self, '_last_pivot_log') or \
                                   (datetime.now() - self._last_pivot_log).total_seconds() > 300:
                                    status = "LOCKED" if self.pivot_tracker.get('levels_locked', False) else "UPDATING"
                                    profit_info = ""
                                    if hasattr(self.pivot_tracker, 'profit_locked') and self.pivot_tracker.get('profit_locked'):
                                        profit_info = f" (Profit Locked: ${self.pivot_tracker['profit_locked']:.0f})"
                                    self.logger.info(f"📊 Pivot Levels ({status}) - Support: ${self.pivot_tracker['support_level']:.0f}, "
                                                   f"Resistance: ${self.pivot_tracker['resistance_level']:.0f}, "
                                                   f"Current: ${current_price:.0f}{profit_info}")
                                    self._last_pivot_log = datetime.now()
                        
                        # 1. Detect market regime
                        regime, confidence, metrics = self.detect_market_regime(
                            df_resampled)

                        # 2. BUG FIX: Use config threshold instead of hardcoded values
                        current_position_value = abs(self.balance_btc * (self.data_manager.get_current_price(self.symbol) or 0)) + self.balance_usd
                        has_significant_position = current_position_value > 50000  # $50k+ position
                        


                        # EMERGENCY FIX: Much more responsive thresholds for losing positions
                        unrealized_pnl = self._calculate_unrealized_pnl()
                        is_losing_position = unrealized_pnl < -5000  # Losing more than $5k

                        if regime == "trending":
                            if is_losing_position:
                                required_confidence = 0.40  # EMERGENCY: Very low threshold for losing positions
                            else:
                                required_confidence = min(0.65, self.regime_switch_threshold - 0.15)  # Much lower for trending
                        elif regime == "ranging":
                            # Changed from max(0.60, ...) to allow lower thresholds via tuning
                            # But still maintain a minimum of 0.30 to prevent too frequent switches
                            required_confidence = max(0.30, self.regime_switch_threshold - 0.15)  # Respect tunable threshold
                        else:  # volatile
                            required_confidence = self.regime_switch_threshold
                         
                        # Smaller increase for significant positions to maintain responsiveness
                        if has_significant_position:
                           if is_losing_position:
                               required_confidence = min(0.50, required_confidence + 0.05)  # Emergency override
                           else:
                               required_confidence = min(0.75, required_confidence + 0.05)  # Reduced from 0.80
 
                         # Only switch if confidence exceeds threshold
                        if confidence >= required_confidence:
                            new_strategy = regime
                            # Check time constraint for strategy switching
                            can_switch_time = True
                            if self.last_strategy_switch_time:
                                mins_since_switch = (datetime.now() - self.last_strategy_switch_time).total_seconds() / 60
                                can_switch_time = mins_since_switch >= self.min_strategy_switch_minutes

                            if new_strategy != self.active_strategy and can_switch_time:
                                mins_str = f" (last switch {mins_since_switch:.0f}min ago)" if self.last_strategy_switch_time else ""
                                self.logger.info(f"✅ Switching to {regime} strategy - confidence {confidence:.1%} >= {required_confidence:.1%} required{mins_str}")
                            elif new_strategy != self.active_strategy and not can_switch_time:
                                self.logger.info(f"⏳ Would switch to {regime} but only {mins_since_switch:.0f}min since last switch (need {self.min_strategy_switch_minutes}min)")
                                new_strategy = self.active_strategy  # Don't switch yet
                        else:
                            new_strategy = self.active_strategy  # Stay with current strategy
                            self.logger.info(f"⏸️  Staying with {self.active_strategy} strategy - {regime} confidence {confidence:.1%} < {required_confidence:.1%} required (position: ${current_position_value:,.0f})")

                        # 3. Check for strategy switch
                        if new_strategy != self.active_strategy:
                            self.logger.info(f"🔄 STRATEGY SWITCH: {self.active_strategy} → {new_strategy} (confidence: {confidence:.1%})")
                            self.active_strategy = new_strategy
                            self.strategy_switches_today += 1
                            self.signal_history = []  # Reset signal history on strategy switch
                            self.last_strategy_switch_time = datetime.now()
                        else:
                            self.logger.debug(f"Keeping {self.active_strategy} strategy (regime: {regime}, confidence: {confidence:.1%})")

                        # 3.5 CRITICAL: Validate position before generating signals
                        self.validate_position_tracking()
                        
                        # 4. Generate signal using active strategy
                        if self.active_strategy == "trending":
                            signal, signal_reason = self.generate_trending_signal(
                                df_resampled)
                        elif self.active_strategy == "ranging":
                            signal, signal_reason = self.generate_ranging_signal(
                                df_resampled)
                        elif self.active_strategy == "volatile":
                            signal, signal_reason = self.generate_volatile_signal(
                                df_resampled)
                        else:
                            signal, signal_reason = 0, "Unknown strategy"

                        # Log diagnostic snapshot if needed
                        if self.diagnostic_logger.should_snapshot():
                            self._log_diagnostic_snapshot()
                            
                        # Log signal evaluation (with deduplication)
                        # Store confirmation result to avoid double-calling
                        signal_confirmed = self.confirm_signal(signal) if signal != 0 else False
                        trade_gap_ok = self.check_trade_gap()
                        
                        will_trade = signal != 0 and signal_confirmed and trade_gap_ok
                        why_not = []
                        if signal == 0:
                            why_not.append("No signal")
                        elif not signal_confirmed:
                            why_not.append(f"Signal not confirmed: {len(self.signal_history)}/{self.signal_confirmation_bars}")
                        elif not trade_gap_ok:
                            mins_since = (datetime.now() - self.last_trade_time).total_seconds() / 60 if self.last_trade_time else 999
                            why_not.append(f"Trade gap: {mins_since:.0f}min < {self.min_trade_gap_minutes}min")
                        elif self.trade_count_today >= self.max_trades_per_day:
                            why_not.append(f"Daily limit: {self.trade_count_today}/{self.max_trades_per_day}")
                        elif signal == self.position:
                            why_not.append(f"Signal matches position")
                            why_not.append(f"Trade gap constraint")
                            
                        # Only log interesting evaluations or periodic updates
                        should_log = will_trade or regime != self.current_regime or self.diagnostic_logger.should_snapshot()
                        if should_log:
                            self.diagnostic_logger.log_signal_evaluation(
                                signal_type=f"{self.active_strategy.upper()}_SIGNAL",
                                signal_value=signal,
                                reason=signal_reason,
                                will_trade=will_trade,
                                why_not=why_not if why_not else None
                            )

                        # Log regime change if it occurred
                        if regime != self.current_regime and hasattr(self, '_last_logged_regime'):
                            self.diagnostic_logger.log_regime_change(
                                old_regime=self._last_logged_regime,
                                new_regime=regime,
                                confidence=confidence,
                                metrics=metrics
                            )
                        self._last_logged_regime = regime

                        # 5. Execute if signal confirmed (use stored result)
                        if signal != 0 and signal_confirmed:
                            current_price = df_resampled.iloc[-1]['close']
                            signal_time = df_resampled.index[-1]
                            self.logger.info(
                                f"📊 {self.active_strategy.upper()}: {signal_reason}")
                            
                            # Additional check for signal validity
                            if signal == self.position:
                                self.logger.info(f"📊 {self.active_strategy.upper()}: {signal_reason} - but already in position")
                                self.signal_history = []  # Clear history since we can't act on this
                                continue
                            if not trade_gap_ok:
                                self.signal_history = []  # Clear if we can't trade yet
                                continue
                                
                            self.check_for_signals(signal, current_price, signal_time)

                        # Store regime info
                        self.current_regime = regime
                        self.regime_confidence = confidence
                        
                        # Mark successful evaluation
                        last_successful_eval = current_time
                    
                    else:
                        # Log insufficient data warning
                        if not hasattr(self, '_last_insufficient_data_warning') or \
                           (datetime.now() - self._last_insufficient_data_warning).seconds > 300:
                            self.logger.warning(f"Insufficient data for signal evaluation: {len(df_resampled)}/{self.long_window} hours available")
                            self._last_insufficient_data_warning = datetime.now()

                except Exception as e:
                    self.logger.error(f"Error in adaptive strategy loop: {e}")
                    self.diagnostic_logger.log_error(f"Adaptive strategy error: {e}")
            else:
                # Log when no data is available
                if not hasattr(self, '_last_no_data_warning') or \
                   (datetime.now() - self._last_no_data_warning).seconds > 300:
                    self.logger.warning("No price data available for evaluation")
                    self._last_no_data_warning = datetime.now()
            
            # Reduce sleep time to 30 seconds for more responsive evaluations
            time.sleep(30)

    def check_for_signals(self, latest_signal, current_price, signal_time):
        """Execute trades with adaptive strategy logic."""
        
        # Always validate position first
        actual_position = self.validate_position_tracking()
        
        # Log signal evaluation details
        self.logger.info(f"📊 Signal Evaluation: signal={latest_signal}, position={self.position}, btc={self.balance_btc:.8f}, usd=${self.balance_usd:.2f}")
        
        # Check startup grace period
        if (datetime.now() - self.startup_time).total_seconds() < (self.startup_grace_period_minutes * 60):
            mins_remaining = self.startup_grace_period_minutes - ((datetime.now() - self.startup_time).total_seconds() / 60)
            self.logger.debug(f"🚫 STARTUP GRACE PERIOD: {mins_remaining:.1f} minutes remaining before trading")
            return

        today = datetime.utcnow().date()
        if today != self.current_day:
            self.current_day = today
            self.trade_count_today = 0
            self.strategy_switches_today = 0

        if self.last_signal_time == signal_time:
            return

        # Check constraints
        if not self.check_trade_gap():
            mins = (datetime.now() - self.last_trade_time).total_seconds() / \
                60 if self.last_trade_time else 0
            self.logger.info(
                f"⏱️  Trade gap: {mins:.1f}min < {self.min_trade_gap_minutes}min required")
            return

        if self.trade_count_today >= self.max_trades_per_day:
            self.logger.info(
                f"📈 Daily limit: {self.trade_count_today}/{self.max_trades_per_day}")
            return

        # Execute trade
        if latest_signal == 1 and self.position <= 0:
            self.logger.info(
                f"🟢 {self.active_strategy.upper()} LONG at ${current_price}")
            
            # Log position before trade
            position_before = {
                "btc": self.balance_btc,
                "usd": self.balance_usd,
                "position": self.position
            }
            
            # Don't overwrite pivot protection reasons
            if "Pivot break:" not in self.last_trade_reason:
                self.last_trade_reason = f"Adaptive {self.active_strategy}: confirmed long"
            self.last_trade_time = datetime.now()
            
            # Execute trade FIRST
            self.buy_in_three_parts(current_price, datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S'), signal_time)
            
            # Update position AFTER successful trade execution
            self.position = 1
            self.trade_count_today += 1
            self.last_signal_time = signal_time
            self.strategy_performance[self.active_strategy]["trades"] += 1
            
            # Log trade execution
            self.diagnostic_logger.log_trade_execution(
                trade_type="BUY",
                price=current_price,
                amount=self.position_size,
                position_before=position_before,
                position_after={"btc": self.balance_btc, "usd": self.balance_usd, "position": self.position},
                pnl=self.total_profit_loss
            )
            
            # Log full status after trade
            self._log_trade_status()
            
        elif latest_signal == -1 and self.position >= 0:
            self.logger.info(
                f"🔴 {self.active_strategy.upper()} SHORT at ${current_price}")
            
            # Log position before trade
            position_before = {
                "btc": self.balance_btc,
                "usd": self.balance_usd,
                "position": self.position
            }
            
            # Don't overwrite pivot protection reasons
            if "Pivot break:" not in self.last_trade_reason:
                self.last_trade_reason = f"Adaptive {self.active_strategy}: confirmed short"
            self.last_trade_time = datetime.now()
            trade_btc = round(self.balance_btc, 8)
            
            # Only execute if we have BTC to sell
            if trade_btc > 1e-8:
                # Execute trade FIRST
                self.execute_trade("sell", current_price, datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S'), signal_time, trade_btc)
                
                # Update position AFTER successful trade execution
                self.position = -1
            else:
                self.logger.warning(f"Cannot sell - insufficient BTC balance: {trade_btc}")
                return
                
            self.trade_count_today += 1
            self.last_signal_time = signal_time
            self.strategy_performance[self.active_strategy]["trades"] += 1
            
            # Log trade execution
            self.diagnostic_logger.log_trade_execution(
                trade_type="SELL",
                price=current_price,
                amount=trade_btc,
                position_before=position_before,
                position_after={"btc": self.balance_btc, "usd": self.balance_usd, "position": self.position},
                pnl=self.total_profit_loss
            )
            
            # Log full status after trade
            self._log_trade_status()

    def get_status(self):
        """Enhanced status with adaptive metrics."""
        status = super().get_status()

        # Add adaptive strategy metrics
        status.update({
            'adaptive_strategy': {
                'current_regime': self.current_regime,
                'regime_confidence': f"{self.regime_confidence:.1%}",
                'active_strategy': self.active_strategy,
                'strategy_switches_today': self.strategy_switches_today,
                'strategy_performance': self.strategy_performance,
                'position_value': abs(self.balance_btc * (self.data_manager.get_current_price(self.symbol) or 0)) + self.balance_usd,
                'confidence_required': self.regime_switch_threshold,  # Use actual config value
            },
            'signal_confirmation': {
                'signals_recorded': len(self.signal_history),
                'confirmation_required': self.signal_confirmation_bars,
                'last_confirmed_signal': self.last_confirmed_signal
            },
            'trade_timing': {
                'min_gap_minutes': self.min_trade_gap_minutes,
                'time_since_last_trade_minutes': (
                    (datetime.now() - self.last_trade_time).total_seconds() / 60
                    if self.last_trade_time else None
                ),
                'can_trade_now': self.check_trade_gap()
            }
        })

        return status
