#!/usr/bin/env python3
"""
Comprehensive tests for Early Warning Monitor.
Run these BEFORE any integration to ensure no bugs.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tdr_core.early_warning_monitor import EarlyWarningMonitor

class TestEarlyWarningMonitor(unittest.TestCase):
    """Test the Early Warning Monitor for safety and correctness."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_data_manager = Mock()
        self.mock_logger = Mock()
        self.monitor = EarlyWarningMonitor(self.mock_data_manager, self.mock_logger)
        
    def test_initialization(self):
        """Test monitor initializes correctly."""
        self.assertTrue(self.monitor.enabled)
        self.assertEqual(self.monitor.ma_short_period, 6)
        self.assertEqual(self.monitor.ma_long_period, 34)
        self.assertEqual(self.monitor.warning_threshold, 0.5)
        
    def test_disable_enable(self):
        """Test monitor can be disabled/enabled."""
        self.monitor.disable()
        self.assertFalse(self.monitor.enabled)
        
        self.monitor.enable()
        self.assertTrue(self.monitor.enabled)
        
    def test_no_operation_when_disabled(self):
        """Test monitor does nothing when disabled."""
        self.monitor.disable()
        result = self.monitor.check_5min_mas()
        self.assertIsNone(result)
        
    def test_insufficient_data_handling(self):
        """Test handling of insufficient candle data."""
        # Mock insufficient data
        self.mock_data_manager.get_recent_candles.return_value = [
            {'close': 100} for _ in range(20)  # Less than 34 needed
        ]
        
        result = self.monitor.check_5min_mas()
        self.assertIsNone(result)
        
    def test_ma_calculation(self):
        """Test MA calculation is correct."""
        # Create test data
        test_candles = []
        for i in range(50):
            test_candles.append({'close': 100 + i})  # Ascending prices
            
        self.mock_data_manager.get_recent_candles.return_value = test_candles
        
        result = self.monitor.check_5min_mas()
        
        self.assertIsNotNone(result)
        self.assertIn('ma_short', result)
        self.assertIn('ma_long', result)
        self.assertIn('proximity', result)
        self.assertIn('signal', result)
        
        # MA_short should be higher than MA_long with ascending prices
        self.assertGreater(result['ma_short'], result['ma_long'])
        self.assertEqual(result['signal'], 1)  # Should be LONG signal
        
    def test_proximity_calculation(self):
        """Test proximity calculation is correct."""
        # Create flat price data
        test_candles = [{'close': 100} for _ in range(50)]
        self.mock_data_manager.get_recent_candles.return_value = test_candles
        
        result = self.monitor.check_5min_mas()
        
        # With flat prices, MAs should be equal
        self.assertAlmostEqual(result['proximity'], 0, places=5)
        
    def test_rate_limiting(self):
        """Test warning rate limiting works."""
        # Set up data that would trigger warnings
        self.monitor.last_5min_signal = -1
        self.monitor.consecutive_signals = 3
        
        warning_data = {
            'proximity': 0.2,
            'signal': 1,
            'ma_short': 101,
            'ma_long': 100
        }
        
        # Issue max warnings
        for i in range(self.monitor.max_warnings_per_hour):
            warning = self.monitor.check_warning_conditions(warning_data, -1)
            self.assertIsNotNone(warning)
            
        # Next warning should be rate limited
        warning = self.monitor.check_warning_conditions(warning_data, -1)
        self.assertIsNone(warning)
        
    def test_consecutive_signal_requirement(self):
        """Test that warnings require consecutive signals."""
        warning_data = {
            'proximity': 0.2,
            'signal': 1,
            'ma_short': 101,
            'ma_long': 100
        }
        
        # First signal - no warning yet
        self.monitor.last_5min_signal = -1
        self.monitor.consecutive_signals = 1
        warning = self.monitor.check_warning_conditions(warning_data, -1)
        self.assertIsNone(warning)
        
        # Second consecutive - should warn for approaching
        self.monitor.consecutive_signals = 2
        warning = self.monitor.check_warning_conditions(warning_data, -1)
        self.assertIsNotNone(warning)
        self.assertIn("approaching", warning)
        
        # Third consecutive - should warn for crossover
        self.monitor.consecutive_signals = 3
        self.monitor.warnings_this_hour = []  # Reset rate limit
        warning = self.monitor.check_warning_conditions(warning_data, -1)
        self.assertIsNotNone(warning)
        self.assertIn("CROSSOVER", warning)
        
    def test_no_warning_same_position(self):
        """Test no warning when signal matches position."""
        warning_data = {
            'proximity': 0.1,  # Very close
            'signal': 1,       # LONG signal
            'ma_short': 101,
            'ma_long': 100
        }
        
        self.monitor.consecutive_signals = 5
        
        # Position already LONG, no warning needed
        warning = self.monitor.check_warning_conditions(warning_data, 1)
        self.assertIsNone(warning)
        
    def test_error_handling(self):
        """Test error handling doesn't crash system."""
        # Mock data manager to raise exception
        self.mock_data_manager.get_recent_candles.side_effect = Exception("API Error")
        
        # Should return None, not crash
        result = self.monitor.check_5min_mas()
        self.assertIsNone(result)
        
        # Should log error
        self.mock_logger.error.assert_called()
        
    def test_monitor_step_integration(self):
        """Test full monitor step execution."""
        # Set up successful data
        test_candles = [{'close': 100 + i * 0.01} for i in range(50)]
        self.mock_data_manager.get_recent_candles.return_value = test_candles
        
        # Run monitor step
        self.monitor.monitor_step(hourly_position=-1)
        
        # Should log debug info
        self.mock_logger.debug.assert_called()
        
    def test_cannot_affect_trading(self):
        """Verify monitor has no access to trading functions."""
        # Monitor should not have any trading methods
        self.assertFalse(hasattr(self.monitor, 'place_order'))
        self.assertFalse(hasattr(self.monitor, 'execute_trade'))
        self.assertFalse(hasattr(self.monitor, 'order_placer'))
        
        # Data manager should be used read-only
        self.monitor.check_5min_mas()
        self.mock_data_manager.get_recent_candles.assert_called()
        # Verify no write operations
        self.assertEqual(self.mock_data_manager.method_calls, 
                        [('get_recent_candles', (), {'symbol': 'btcusd', 'timeframe': '5m', 'limit': 50})])


class TestEarlyWarningIntegration(unittest.TestCase):
    """Test integration scenarios."""
    
    def test_parallel_operation(self):
        """Test monitor doesn't interfere with main system."""
        # This would be tested in actual integration
        # For now, verify the design is parallel-safe
        pass
        
    def test_performance_impact(self):
        """Test monitor doesn't slow down main loop."""
        mock_dm = Mock()
        mock_dm.get_recent_candles.return_value = [{'close': 100} for _ in range(50)]
        
        monitor = EarlyWarningMonitor(mock_dm)
        
        start = datetime.now()
        for _ in range(100):
            monitor.monitor_step(1)
        duration = (datetime.now() - start).total_seconds()
        
        # Should complete 100 iterations quickly
        self.assertLess(duration, 1.0)  # Less than 1 second


def run_all_tests():
    """Run all tests and report results."""
    print("Running Early Warning Monitor Tests...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED - Safe to proceed with integration")
    else:
        print("❌ TESTS FAILED - DO NOT INTEGRATE")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)