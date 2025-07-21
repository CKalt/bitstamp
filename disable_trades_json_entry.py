#!/usr/bin/env python3
"""
Disable trades.json entry price calculation in strategies.py
"""
import re

strategies_file = "src/tdr_core/strategies.py"

# Read the file
with open(strategies_file, 'r') as f:
    content = f.read()

# Backup
with open(f"{strategies_file}.backup_entry", 'w') as f:
    f.write(content)

# Find and comment out the lines that use trades.json entry price
# Pattern 1: For LONG positions
content = re.sub(
    r'self\.logger\.info\(f"\[ENTRY_PRICE_DEBUG\] Using trades\.json entry price: \$\{calculated_entry_price:\.2f\}"\)',
    r'# self.logger.info(f"[ENTRY_PRICE_DEBUG] Using trades.json entry price: ${calculated_entry_price:.2f}")',
    content
)

# Pattern 2: Override the calculated_entry_price usage
content = re.sub(
    r'if calculated_entry_price is not None:',
    r'if False:  # DISABLED: calculated_entry_price is not None:',
    content
)

# Write back
with open(strategies_file, 'w') as f:
    f.write(content)

print("✅ Disabled trades.json entry price calculation")
print("The system will now use the actual position tracking entry price")