#!/usr/bin/env python3
"""
System Verification Module
Continuously checks system health and detects regressions
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class SystemVerifier:
    """
    Verifies system behavior hasn't regressed.
    Runs continuous checks and alerts on anomalies.
    """
    
    def __init__(self, strategy, logger=None):
        self.strategy = strategy
        self.logger = logger or logging.getLogger(__name__)
        self.verification_history = []
        self.max_history = 1000
        self.last_verification = None
        self.error_counts = {}
        
        # Baseline expectations
        self.baselines = {
            'max_entry_price_change': 0.20,  # 20% max change
            'max_position_size_change': 0.0001,  # Tiny changes only
            'signal_eval_frequency': (25, 35),  # 25-35 seconds
            'max_cost_basis_discrepancy': 100,  # $100 tolerance
        }
        
    def verify_all(self) -> Dict[str, any]:
        """Run all verification checks."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # Run each verification
        checks = [
            ('position_consistency', self.verify_position_consistency),
            ('entry_price_sanity', self.verify_entry_price_sanity),
            ('balance_consistency', self.verify_balance_consistency),
            ('signal_frequency', self.verify_signal_frequency),
            ('cost_basis_integrity', self.verify_cost_basis_integrity),
            ('trade_count_accuracy', self.verify_trade_count_accuracy),
        ]
        
        for check_name, check_func in checks:
            try:
                errors = check_func()
                results['checks'][check_name] = {
                    'passed': len(errors) == 0,
                    'errors': errors
                }
                
                # Track error counts
                if errors:
                    self.error_counts[check_name] = self.error_counts.get(check_name, 0) + 1
                    
            except Exception as e:
                results['checks'][check_name] = {
                    'passed': False,
                    'errors': [f'Check failed with exception: {str(e)}']
                }
        
        # Store in history
        self.verification_history.append(results)
        if len(self.verification_history) > self.max_history:
            self.verification_history.pop(0)
            
        self.last_verification = datetime.now()
        
        # Log summary
        failed_checks = [k for k, v in results['checks'].items() if not v['passed']]
        if failed_checks:
            self.logger.warning(f"⚠️ System verification failed {len(failed_checks)} checks: {failed_checks}")
        else:
            self.logger.debug("✅ All system verification checks passed")
            
        return results
    
    def verify_position_consistency(self) -> List[str]:
        """Verify position tracking is internally consistent."""
        errors = []
        
        # Check 1: Position direction matches balances
        if hasattr(self.strategy, 'position'):
            if self.strategy.position == 1:  # LONG
                if not hasattr(self.strategy, 'balance_btc') or self.strategy.balance_btc <= 0:
                    errors.append(f"LONG position but BTC balance is {getattr(self.strategy, 'balance_btc', 0)}")
                if hasattr(self.strategy, 'balance_usd') and self.strategy.balance_usd > 100:
                    errors.append(f"LONG position but USD balance is ${self.strategy.balance_usd:.2f}")
                    
            elif self.strategy.position == -1:  # SHORT
                if not hasattr(self.strategy, 'balance_usd') or self.strategy.balance_usd <= 0:
                    errors.append(f"SHORT position but USD balance is ${getattr(self.strategy, 'balance_usd', 0)}")
                if hasattr(self.strategy, 'balance_btc') and self.strategy.balance_btc > 0.001:
                    errors.append(f"SHORT position but BTC balance is {self.strategy.balance_btc}")
        
        # Check 2: Position size sign matches position direction
        if hasattr(self.strategy, 'position_size') and hasattr(self.strategy, 'position'):
            if self.strategy.position == 1 and self.strategy.position_size < 0:
                errors.append(f"LONG position but negative position_size: {self.strategy.position_size}")
            elif self.strategy.position == -1 and self.strategy.position_size > 0:
                errors.append(f"SHORT position but positive position_size: {self.strategy.position_size}")
                
        return errors
    
    def verify_entry_price_sanity(self) -> List[str]:
        """Verify entry price is reasonable."""
        errors = []
        
        if hasattr(self.strategy, 'position') and self.strategy.position != 0:
            # Get entry price
            entry_price = None
            if hasattr(self.strategy, 'get_entry_price'):
                entry_price = self.strategy.get_entry_price()
            elif hasattr(self.strategy, 'position_cost_basis') and hasattr(self.strategy, 'position_size'):
                if self.strategy.position_size != 0:
                    entry_price = abs(self.strategy.position_cost_basis / self.strategy.position_size)
            
            if entry_price:
                # Get current price
                current_price = None
                if hasattr(self.strategy, 'data_manager'):
                    current_price = self.strategy.data_manager.get_current_price('btcusd')
                
                if current_price and entry_price:
                    # Check if entry price is reasonable (within 20% of current)
                    price_diff_pct = abs(entry_price - current_price) / current_price
                    if price_diff_pct > self.baselines['max_entry_price_change']:
                        errors.append(f"Entry price ${entry_price:.2f} is {price_diff_pct:.1%} different from current ${current_price:.2f}")
                
                # Check if entry price is in reasonable range
                if entry_price < 10000 or entry_price > 500000:
                    errors.append(f"Entry price ${entry_price:.2f} is outside reasonable range ($10k-$500k)")
                    
        return errors
    
    def verify_balance_consistency(self) -> List[str]:
        """Verify balances are consistent with position."""
        errors = []
        
        # Check total value is reasonable
        if hasattr(self.strategy, 'balance_btc') and hasattr(self.strategy, 'balance_usd'):
            current_price = None
            if hasattr(self.strategy, 'data_manager'):
                current_price = self.strategy.data_manager.get_current_price('btcusd')
                
            if current_price:
                total_value = self.strategy.balance_usd + (self.strategy.balance_btc * current_price)
                
                # Check if total value is reasonable (should be 100k-300k for this system)
                if total_value < 50000:
                    errors.append(f"Total portfolio value too low: ${total_value:.2f}")
                elif total_value > 500000:
                    errors.append(f"Total portfolio value suspiciously high: ${total_value:.2f}")
                    
        return errors
    
    def verify_signal_frequency(self) -> List[str]:
        """Verify signals are being evaluated at expected frequency."""
        errors = []
        
        # This check needs access to recent signal times
        # Would need to be implemented with access to signal history
        
        return errors
    
    def verify_cost_basis_integrity(self) -> List[str]:
        """Verify cost basis calculations are consistent."""
        errors = []
        
        if hasattr(self.strategy, 'position_cost_basis') and hasattr(self.strategy, 'position_size'):
            if self.strategy.position_size != 0:
                # Calculate implied entry price
                implied_entry = abs(self.strategy.position_cost_basis / self.strategy.position_size)
                
                # Compare with stated entry price if available
                if hasattr(self.strategy, 'get_entry_price'):
                    stated_entry = self.strategy.get_entry_price()
                    if stated_entry and abs(implied_entry - stated_entry) > self.baselines['max_cost_basis_discrepancy']:
                        errors.append(f"Cost basis implies entry ${implied_entry:.2f} but stated entry is ${stated_entry:.2f}")
                        
        return errors
    
    def verify_trade_count_accuracy(self) -> List[str]:
        """Verify trade counts are accurate."""
        errors = []
        
        if hasattr(self.strategy, 'trades_executed') and hasattr(self.strategy, 'trade_count_today'):
            if self.strategy.trade_count_today > self.strategy.trades_executed:
                errors.append(f"Daily trade count {self.strategy.trade_count_today} exceeds total trades {self.strategy.trades_executed}")
                
        return errors
    
    def get_verification_summary(self) -> Dict[str, any]:
        """Get summary of recent verifications."""
        if not self.verification_history:
            return {'status': 'No verifications run yet'}
            
        recent = self.verification_history[-10:]  # Last 10 checks
        
        total_checks = sum(len(v['checks']) for v in recent)
        failed_checks = sum(1 for v in recent for check in v['checks'].values() if not check['passed'])
        
        return {
            'total_verifications': len(self.verification_history),
            'recent_checks': total_checks,
            'recent_failures': failed_checks,
            'failure_rate': failed_checks / total_checks if total_checks > 0 else 0,
            'error_counts': self.error_counts,
            'last_verification': self.last_verification.isoformat() if self.last_verification else None
        }
    
    def should_alert(self, check_name: str) -> bool:
        """Determine if we should alert for this check failure."""
        # Alert on first failure or every 10th failure
        count = self.error_counts.get(check_name, 0)
        return count == 1 or count % 10 == 0


class PositionStateRecorder:
    """
    Records position state changes for regression detection.
    """
    
    def __init__(self, filename='position_state_history.json'):
        self.filename = filename
        self.history = self.load_history()
        
    def load_history(self) -> List[Dict]:
        """Load historical position states."""
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except:
            return []
            
    def save_history(self):
        """Save position state history."""
        try:
            # Keep last 10000 records
            if len(self.history) > 10000:
                self.history = self.history[-10000:]
                
            with open(self.filename, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save position history: {e}")
            
    def record_state(self, strategy):
        """Record current position state."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'position': getattr(strategy, 'position', None),
            'position_size': getattr(strategy, 'position_size', None),
            'position_cost_basis': getattr(strategy, 'position_cost_basis', None),
            'balance_btc': getattr(strategy, 'balance_btc', None),
            'balance_usd': getattr(strategy, 'balance_usd', None),
            'trades_executed': getattr(strategy, 'trades_executed', None),
            'entry_price': strategy.get_entry_price() if hasattr(strategy, 'get_entry_price') else None
        }
        
        self.history.append(state)
        
        # Save periodically
        if len(self.history) % 100 == 0:
            self.save_history()
            
    def detect_anomaly(self, current_state: Dict) -> List[str]:
        """Detect anomalies in current state compared to history."""
        anomalies = []
        
        if len(self.history) < 10:
            return anomalies  # Not enough history
            
        # Check for sudden position size changes
        recent = self.history[-10:]
        recent_sizes = [h.get('position_size', 0) for h in recent if h.get('position_size') is not None]
        
        if recent_sizes and current_state.get('position_size') is not None:
            avg_size = sum(recent_sizes) / len(recent_sizes)
            current_size = current_state['position_size']
            
            if avg_size != 0 and abs(current_size - avg_size) / abs(avg_size) > 0.1:  # 10% change
                anomalies.append(f"Position size changed significantly: {avg_size:.4f} -> {current_size:.4f}")
                
        return anomalies


def integrate_verifier(strategy):
    """
    Integrate verifier into existing strategy.
    Call this after strategy initialization.
    """
    verifier = SystemVerifier(strategy, strategy.logger)
    recorder = PositionStateRecorder()
    
    # Monkey patch the strategy to add verification
    original_check_for_signals = strategy.check_for_signals
    
    def verified_check_for_signals(*args, **kwargs):
        # Run original
        result = original_check_for_signals(*args, **kwargs)
        
        # Run verification
        verification_results = verifier.verify_all()
        
        # Record state
        recorder.record_state(strategy)
        
        # Alert on failures
        for check_name, check_result in verification_results['checks'].items():
            if not check_result['passed'] and verifier.should_alert(check_name):
                strategy.logger.error(f"🚨 VERIFICATION FAILED - {check_name}: {check_result['errors']}")
                
        return result
        
    strategy.check_for_signals = verified_check_for_signals
    strategy.verifier = verifier
    strategy.state_recorder = recorder
    
    return verifier