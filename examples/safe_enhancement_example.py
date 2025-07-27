#!/usr/bin/env python3
"""
Example: How to Safely Add Enhancements Without Breaking Production

This demonstrates the recommended development process for adding
new features while preventing regression bugs.
"""

import json
import logging
from datetime import datetime

# Example: Adding a new entry price calculation method
# GOAL: Improve entry price accuracy without breaking existing calculation

class FeatureFlags:
    """Simple feature flag system."""
    
    def __init__(self, config_file='feature_flags.json'):
        self.config_file = config_file
        self.flags = self.load_flags()
        
    def load_flags(self):
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except:
            return {}
            
    def get(self, flag_name, default=False):
        return self.flags.get(flag_name, default)
        
    def set(self, flag_name, value):
        self.flags[flag_name] = value
        with open(self.config_file, 'w') as f:
            json.dump(self.flags, f, indent=2)


class EnhancedStrategy:
    """Example strategy with safe enhancement pattern."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_flags = FeatureFlags()
        self.position_size = 1.5
        self.position_cost_basis = 175000
        
        # Metrics for comparing old vs new
        self.comparison_metrics = {
            'old_calculations': [],
            'new_calculations': [],
            'discrepancies': []
        }
        
    def calculate_entry_price(self):
        """
        Safe enhancement pattern:
        1. Keep original code unchanged
        2. Add new implementation behind feature flag
        3. Run both and compare in shadow mode
        4. Gradually roll out new implementation
        """
        
        # Original calculation - NEVER MODIFY THIS
        original_result = self._calculate_entry_price_v1()
        
        # Check if we should use new calculation
        if self.feature_flags.get('use_enhanced_entry_calculation'):
            # New calculation is active
            return self._calculate_entry_price_v2()
            
        # Check if we should run in shadow mode
        elif self.feature_flags.get('shadow_enhanced_entry_calculation'):
            # Run new calculation but don't use it
            try:
                new_result = self._calculate_entry_price_v2()
                
                # Compare results
                self._compare_calculations(original_result, new_result)
                
            except Exception as e:
                self.logger.error(f"Shadow calculation failed: {e}")
                
        # Always return original result unless new is explicitly enabled
        return original_result
        
    def _calculate_entry_price_v1(self):
        """Original entry price calculation - DO NOT MODIFY."""
        if self.position_size == 0:
            return 0
        return abs(self.position_cost_basis / self.position_size)
        
    def _calculate_entry_price_v2(self):
        """
        Enhanced entry price calculation.
        Includes additional validation and edge case handling.
        """
        # New implementation with improvements
        if abs(self.position_size) < 0.00001:  # Handle near-zero
            self.logger.warning("Position size near zero, returning 0")
            return 0
            
        # Additional validation
        if self.position_cost_basis < 0 and self.position_size > 0:
            self.logger.warning("Cost basis negative for long position")
            
        # Enhanced calculation with rounding
        raw_price = abs(self.position_cost_basis / self.position_size)
        
        # Sanity check
        if raw_price < 1000 or raw_price > 1000000:
            self.logger.warning(f"Entry price {raw_price} outside normal range")
            
        return round(raw_price, 2)
        
    def _compare_calculations(self, old_result, new_result):
        """Compare old vs new calculations for discrepancies."""
        
        # Record both results
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'old': old_result,
            'new': new_result,
            'difference': abs(old_result - new_result) if old_result else 0,
            'position_size': self.position_size,
            'cost_basis': self.position_cost_basis
        }
        
        self.comparison_metrics['old_calculations'].append(old_result)
        self.comparison_metrics['new_calculations'].append(new_result)
        
        # Check for significant discrepancy
        if old_result > 0:
            pct_diff = abs(old_result - new_result) / old_result
            if pct_diff > 0.001:  # 0.1% difference
                comparison['significant'] = True
                self.comparison_metrics['discrepancies'].append(comparison)
                self.logger.warning(f"Entry price discrepancy: v1=${old_result:.2f} v2=${new_result:.2f} ({pct_diff:.1%} diff)")
                
        # Log for analysis
        with open('calculation_comparisons.jsonl', 'a') as f:
            f.write(json.dumps(comparison) + '\n')


def demonstrate_safe_rollout():
    """
    Demonstrate the safe rollout process for a new feature.
    """
    
    print("Safe Enhancement Rollout Demo")
    print("=" * 50)
    
    strategy = EnhancedStrategy()
    
    # Stage 1: Production with no changes
    print("\nStage 1: Production (no flags)")
    strategy.feature_flags.set('shadow_enhanced_entry_calculation', False)
    strategy.feature_flags.set('use_enhanced_entry_calculation', False)
    
    result = strategy.calculate_entry_price()
    print(f"Entry price: ${result:.2f} (using original calculation)")
    
    # Stage 2: Shadow mode
    print("\nStage 2: Shadow Mode (comparing but not using)")
    strategy.feature_flags.set('shadow_enhanced_entry_calculation', True)
    strategy.feature_flags.set('use_enhanced_entry_calculation', False)
    
    result = strategy.calculate_entry_price()
    print(f"Entry price: ${result:.2f} (using original, but comparing with new)")
    
    # Simulate some calculations
    test_cases = [
        (1.5, 150000),
        (0.001, 100),
        (2.0, 200000),
        (0.0000001, 10),  # Edge case
    ]
    
    for size, basis in test_cases:
        strategy.position_size = size
        strategy.position_cost_basis = basis
        strategy.calculate_entry_price()
        
    # Check for discrepancies
    if strategy.comparison_metrics['discrepancies']:
        print(f"\n⚠️  Found {len(strategy.comparison_metrics['discrepancies'])} discrepancies!")
        for disc in strategy.comparison_metrics['discrepancies']:
            print(f"  Position {disc['position_size']}: v1=${disc['old']:.2f} vs v2=${disc['new']:.2f}")
    else:
        print("\n✅ No significant discrepancies found")
        
    # Stage 3: Gradual rollout
    print("\nStage 3: Gradual Rollout")
    print("Would now enable for 10% -> 25% -> 50% -> 100% of calculations")
    print("Monitoring error rates and discrepancies at each stage")
    
    # Stage 4: Full deployment
    print("\nStage 4: Full Deployment (after verification)")
    strategy.feature_flags.set('shadow_enhanced_entry_calculation', False)
    strategy.feature_flags.set('use_enhanced_entry_calculation', True)
    
    strategy.position_size = 1.5
    strategy.position_cost_basis = 175000
    result = strategy.calculate_entry_price()
    print(f"Entry price: ${result:.2f} (using new enhanced calculation)")
    
    print("\n" + "=" * 50)
    print("Summary: This approach ensures:")
    print("1. Original code remains unchanged")
    print("2. New code is tested in production without risk")
    print("3. Discrepancies are detected before deployment")
    print("4. Rollback is instant (just flip feature flag)")
    print("5. Gradual rollout minimizes risk")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    demonstrate_safe_rollout()