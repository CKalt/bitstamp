"""
System Verifier - Continuous regression detection for TDR trading system
Runs checks every 30 seconds to catch bugs and inconsistencies in real-time
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import os


class SystemVerifier:
    """Continuous system verification to catch regressions early."""
    
    def __init__(self, strategy, data_manager, logger=None):
        self.strategy = strategy
        self.data_manager = data_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Track verification history
        self.check_history = []
        self.error_count = 0
        self.last_check_time = None
        
        # Configuration
        self.enabled = True
        self.check_interval = 30  # seconds
        self.max_price_deviation = 0.5  # 50% max difference between entry and current
        self.max_history_size = 100
        
    def run_all_checks(self) -> Dict[str, any]:
        """Run all verification checks and return results."""
        if not self.enabled:
            return {"enabled": False}
            
        results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "errors": [],
            "warnings": []
        }
        
        # Run each check
        checks = [
            ("position_consistency", self.verify_position_consistency),
            ("entry_price_sanity", self.verify_entry_price_sanity),
            ("balance_integrity", self.verify_balance_integrity),
            ("cost_basis_accuracy", self.verify_cost_basis_accuracy),
            ("trade_execution", self.verify_trade_execution),
            ("data_synchronization", self.verify_data_synchronization)
        ]
        
        for check_name, check_func in checks:
            try:
                check_errors = check_func()
                results["checks"][check_name] = {
                    "passed": len(check_errors) == 0,
                    "errors": check_errors
                }
                results["errors"].extend(check_errors)
            except Exception as e:
                error_msg = f"Check {check_name} failed with exception: {e}"
                results["checks"][check_name] = {
                    "passed": False,
                    "errors": [error_msg]
                }
                results["errors"].append(error_msg)
                self.logger.error(error_msg)
        
        # Update history
        self.check_history.append(results)
        if len(self.check_history) > self.max_history_size:
            self.check_history.pop(0)
            
        self.last_check_time = datetime.now()
        self.error_count = len(results["errors"])
        
        # Log results
        if results["errors"]:
            self.logger.error(f"🚨 SYSTEM VERIFIER: {len(results['errors'])} errors detected!")
            for error in results["errors"]:
                self.logger.error(f"  ❌ {error}")
        else:
            self.logger.debug("✅ SYSTEM VERIFIER: All checks passed")
            
        return results
    
    def verify_position_consistency(self) -> List[str]:
        """Verify position tracking is internally consistent."""
        errors = []
        
        # Check 1: Position direction matches balances
        if self.strategy.position == 1:  # LONG
            if not hasattr(self.strategy, 'balance_btc') or self.strategy.balance_btc <= 0:
                errors.append(f"LONG position but BTC balance is {getattr(self.strategy, 'balance_btc', 0)}")
        elif self.strategy.position == -1:  # SHORT
            if not hasattr(self.strategy, 'balance_usd') or self.strategy.balance_usd <= 0:
                errors.append(f"SHORT position but USD balance is {getattr(self.strategy, 'balance_usd', 0)}")
                
        # Check 2: Position size matches position direction
        if hasattr(self.strategy, 'position_size'):
            if self.strategy.position == 1 and self.strategy.position_size <= 0:
                errors.append(f"LONG position but position_size is {self.strategy.position_size}")
            elif self.strategy.position == -1 and self.strategy.position_size >= 0:
                errors.append(f"SHORT position but position_size is {self.strategy.position_size}")
                
        return errors
    
    def verify_entry_price_sanity(self) -> List[str]:
        """Verify entry price is reasonable."""
        errors = []
        
        if self.strategy.position == 0:
            return errors  # No position, no entry price to check
            
        current_price = self.data_manager.get_current_price('btcusd')
        if not current_price:
            errors.append("Cannot verify entry price - no current price available")
            return errors
            
        # Get entry price from multiple sources
        entry_prices = []
        
        # Source 1: Direct calculation
        if hasattr(self.strategy, 'position_size') and self.strategy.position_size != 0:
            if hasattr(self.strategy, 'position_cost_basis'):
                calculated_entry = abs(self.strategy.position_cost_basis / self.strategy.position_size)
                entry_prices.append(("calculated", calculated_entry))
        
        # Source 2: Stored entry price
        if hasattr(self.strategy, 'entry_price') and self.strategy.entry_price > 0:
            entry_prices.append(("stored", self.strategy.entry_price))
            
        # Source 3: From trades
        try:
            trades_entry, _ = self.strategy.calculate_entry_price_from_trades()
            if trades_entry and trades_entry > 0:
                entry_prices.append(("trades", trades_entry))
        except:
            pass
            
        # Check all entry prices are reasonable
        for source, entry_price in entry_prices:
            deviation = abs(entry_price - current_price) / current_price
            if deviation > self.max_price_deviation:
                errors.append(
                    f"Entry price from {source} (${entry_price:.2f}) deviates "
                    f"{deviation:.1%} from current price (${current_price:.2f})"
                )
                
        # Check all sources agree (within 1%)
        if len(entry_prices) > 1:
            prices = [p[1] for p in entry_prices]
            min_price, max_price = min(prices), max(prices)
            if (max_price - min_price) / min_price > 0.01:
                price_details = ", ".join([f"{s}: ${p:.2f}" for s, p in entry_prices])
                errors.append(f"Entry price mismatch between sources: {price_details}")
                
        return errors
    
    def verify_balance_integrity(self) -> List[str]:
        """Verify balances are internally consistent."""
        errors = []
        
        # Check for negative balances
        if hasattr(self.strategy, 'balance_btc') and self.strategy.balance_btc < 0:
            errors.append(f"Negative BTC balance: {self.strategy.balance_btc}")
            
        if hasattr(self.strategy, 'balance_usd') and self.strategy.balance_usd < 0:
            errors.append(f"Negative USD balance: {self.strategy.balance_usd}")
            
        # Check total value consistency
        if hasattr(self.strategy, 'balance_btc') and hasattr(self.strategy, 'balance_usd'):
            current_price = self.data_manager.get_current_price('btcusd')
            if current_price:
                total_usd = self.strategy.balance_usd + (self.strategy.balance_btc * current_price)
                
                # Check against expected range (should be close to initial capital)
                # Allow for some profit/loss but flag extreme values
                if total_usd < 1000:  # Less than $1k seems wrong
                    errors.append(f"Total portfolio value too low: ${total_usd:.2f}")
                elif total_usd > 1000000:  # More than $1M for test account
                    errors.append(f"Total portfolio value suspiciously high: ${total_usd:.2f}")
                    
        return errors
    
    def verify_cost_basis_accuracy(self) -> List[str]:
        """Verify cost basis tracking is accurate."""
        errors = []
        
        if not hasattr(self.strategy, 'position_cost_basis'):
            return errors
            
        # For LONG positions
        if self.strategy.position == 1:
            if hasattr(self.strategy, 'position_size') and self.strategy.position_size > 0:
                # Cost basis should be positive for long
                if self.strategy.position_cost_basis <= 0:
                    errors.append(
                        f"LONG position with size {self.strategy.position_size} "
                        f"but cost basis is {self.strategy.position_cost_basis}"
                    )
                    
        # For SHORT positions  
        elif self.strategy.position == -1:
            if hasattr(self.strategy, 'position_size') and self.strategy.position_size < 0:
                # Cost basis should be positive (absolute value)
                if self.strategy.position_cost_basis <= 0:
                    errors.append(
                        f"SHORT position with size {self.strategy.position_size} "
                        f"but cost basis is {self.strategy.position_cost_basis}"
                    )
                    
        return errors
    
    def verify_trade_execution(self) -> List[str]:
        """Verify trades are executing when they should."""
        errors = []
        
        # This check needs access to recent signal evaluations
        # For now, just check if we're stuck in a position too long
        if hasattr(self.strategy, 'last_trade_time') and self.strategy.last_trade_time:
            time_since_trade = datetime.now() - self.strategy.last_trade_time
            if time_since_trade > timedelta(days=7):
                errors.append(
                    f"No trades executed in {time_since_trade.days} days - "
                    "possible signal evaluation issue"
                )
                
        return errors
    
    def verify_data_synchronization(self) -> List[str]:
        """Verify data_manager and strategy are in sync."""
        errors = []
        
        # Check if data_manager has position tracking
        if hasattr(self.data_manager, 'position'):
            if self.data_manager.position != self.strategy.position:
                errors.append(
                    f"Position mismatch: strategy={self.strategy.position}, "
                    f"data_manager={self.data_manager.position}"
                )
                
        # Check position sizes match
        if hasattr(self.data_manager, 'position_size') and hasattr(self.strategy, 'position_size'):
            if abs(self.data_manager.position_size - self.strategy.position_size) > 0.00000001:
                errors.append(
                    f"Position size mismatch: strategy={self.strategy.position_size}, "
                    f"data_manager={self.data_manager.position_size}"
                )
                
        return errors
    
    def get_summary(self) -> Dict:
        """Get summary of verification status."""
        return {
            "enabled": self.enabled,
            "last_check": self.last_check_time.isoformat() if self.last_check_time else None,
            "error_count": self.error_count,
            "checks_run": len(self.check_history),
            "recent_errors": self.get_recent_errors()
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict]:
        """Get recent errors from check history."""
        recent_errors = []
        for check in reversed(self.check_history):
            if check["errors"]:
                recent_errors.append({
                    "timestamp": check["timestamp"],
                    "errors": check["errors"]
                })
                if len(recent_errors) >= limit:
                    break
        return recent_errors