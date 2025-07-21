#!/usr/bin/env python3
"""
Fix for the progress display showing 100% repeatedly
"""
import os
import re

def fix_progress_calculation():
    """Fix the progress calculation in loader.py"""
    loader_file = "src/data/loader.py"
    
    if not os.path.exists(loader_file):
        print(f"❌ Error: {loader_file} not found")
        return
    
    # Backup the file
    backup = f"{loader_file}.backup"
    with open(loader_file, 'r') as f:
        content = f.read()
    
    with open(backup, 'w') as f:
        f.write(content)
    print(f"✅ Backed up {loader_file}")
    
    # Fix the progress calculation
    # Look for the progress percentage calculation
    fixed_content = re.sub(
        r'progress_pct = min\(100, int\(\(lines_processed / lines_to_process\) \* 100\)\)',
        'progress_pct = int((line_count / total_lines) * 100) if total_lines > 0 else 0',
        content
    )
    
    # Also fix the status message to show actual vs total
    fixed_content = re.sub(
        r'f"Status: Reading historical data - Line {line_count:,} \({lines_processed:,}/{lines_to_process:,} processed, {progress_pct}%\)',
        'f"Status: Reading historical data - Line {line_count:,} ({lines_processed:,} processed, {progress_pct}% of file)"',
        fixed_content
    )
    
    if fixed_content != content:
        with open(loader_file, 'w') as f:
            f.write(fixed_content)
        print(f"✅ Fixed progress calculation in {loader_file}")
    else:
        print(f"ℹ️  No changes needed in {loader_file}")

if __name__ == "__main__":
    fix_progress_calculation()