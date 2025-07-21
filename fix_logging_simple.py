#!/usr/bin/env python3
"""
Simple fix to reduce logging spam in strategies.py
"""
import os
import re

def fix_logging_spam():
    """Comment out or reduce spammy log lines"""
    strategies_file = "src/tdr_core/strategies.py"
    
    if not os.path.exists(strategies_file):
        print(f"❌ Error: {strategies_file} not found")
        return
    
    # Backup the file
    with open(strategies_file, 'r') as f:
        content = f.read()
    
    backup = f"{strategies_file}.backup"
    with open(backup, 'w') as f:
        f.write(content)
    print(f"✅ Backed up {strategies_file}")
    
    # Fix 1: Change signal evaluation from info to debug
    content = re.sub(
        r'self\.logger\.info\(f"📊 Signal Evaluation:',
        r'self.logger.debug(f"📊 Signal Evaluation:',
        content
    )
    
    # Fix 2: Change "Using last SELL price" from info to debug
    content = re.sub(
        r'self\.logger\.info\(f"Using last SELL price',
        r'self.logger.debug(f"Using last SELL price',
        content
    )
    
    # Fix 3: Change emergency loss threshold to -10000 (from -2000)
    content = re.sub(
        r'if self\.position != 0 and unrealized_pnl < -2000:',
        r'if self.position != 0 and unrealized_pnl < -10000:',
        content
    )
    
    # Fix 4: Comment out the ENTRY_PRICE_DEBUG log
    content = re.sub(
        r'(self\.logger\.info\(f"\[ENTRY_PRICE_DEBUG\])',
        r'# \1',
        content
    )
    
    # Fix 5: Reduce adaptive strategy evaluation logging frequency
    content = re.sub(
        r'if evaluation_count % 5 == 0:',
        r'if evaluation_count % 100 == 0:',  # Only log every 100 evaluations
        content
    )
    
    with open(strategies_file, 'w') as f:
        f.write(content)
    
    print("✅ Fixed logging spam in strategies.py")
    print("   - Signal evaluation: info → debug")
    print("   - SELL price logs: info → debug")
    print("   - Emergency threshold: -$2000 → -$10000")
    print("   - Commented out ENTRY_PRICE_DEBUG")
    print("   - Reduced evaluation logging frequency")

if __name__ == "__main__":
    fix_logging_spam()