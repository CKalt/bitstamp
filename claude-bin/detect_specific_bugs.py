#!/usr/bin/env python3
"""
Detect specific bugs in paper trading logs
"""
import re
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict

def run_ssh_command(cmd):
    """Run command on remote server"""
    try:
        result = subprocess.run(f'ssh ck \'{cmd}\'', shell=True, capture_output=True, text=True)
        return result.stdout
    except:
        return ""

def analyze_logs():
    """Analyze logs for specific bug patterns"""
    print("🐛 AUTOMATED BUG DETECTION")
    print("=" * 50)
    
    bugs_found = []
    
    # Get last 1000 lines of log
    log_data = run_ssh_command('tail -1000 /home/chris/projects/bitstamp-testing/logs/tdr_server.log')
    lines = log_data.strip().split('\n')
    
    # 1. Check candle timing consistency
    print("\n1. Checking 1-minute candle timing...")
    candle_times = []
    for line in lines:
        if "NEW 1-MIN CANDLE:" in line:
            match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if match:
                candle_times.append(datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S'))
    
    if len(candle_times) > 1:
        gaps = []
        for i in range(1, len(candle_times)):
            gap = (candle_times[i] - candle_times[i-1]).total_seconds()
            gaps.append(gap)
        
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        max_gap = max(gaps) if gaps else 0
        
        print(f"   Candles found: {len(candle_times)}")
        print(f"   Average gap: {avg_gap:.1f} seconds")
        print(f"   Max gap: {max_gap:.1f} seconds")
        
        if max_gap > 90:  # More than 1.5 minutes
            bugs_found.append(f"TIMING BUG: Gap of {max_gap}s between candles")
    
    # 2. Check for duplicate trades
    print("\n2. Checking for duplicate trades...")
    trade_signatures = defaultdict(int)
    for line in lines:
        if "PAPER TRADE:" in line:
            # Extract amount and price
            match = re.search(r'Would (\w+) ([\d.]+) BTC @ \$([\d,]+)', line)
            if match:
                sig = f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
                trade_signatures[sig] += 1
    
    for sig, count in trade_signatures.items():
        if count > 3:  # More than 3 parts
            bugs_found.append(f"DUPLICATE TRADE BUG: {sig} executed {count} times")
    
    # 3. Check position consistency
    print("\n3. Checking position consistency...")
    positions = []
    for line in lines:
        if "Pos=" in line and "SIGNAL_EVAL" in line:
            match = re.search(r'Pos=([-\d]+)', line)
            if match:
                positions.append(int(match.group(1)))
    
    if len(positions) > 10:
        # Check for impossible position changes
        for i in range(1, len(positions)):
            if abs(positions[i] - positions[i-1]) > 1:
                bugs_found.append(f"POSITION JUMP BUG: {positions[i-1]} -> {positions[i]}")
    
    # 4. Check proximity threshold violations
    print("\n4. Checking proximity threshold...")
    for line in lines:
        if "SIGNAL_EVAL" in line:
            prox_match = re.search(r'Prox=([\d.]+)%', line)
            action_match = re.search(r'Action=(\w+)', line)
            
            if prox_match and action_match:
                proximity = float(prox_match.group(1))
                action = action_match.group(1)
                
                if proximity < 0.5 and action in ['WILL_BUY', 'WILL_SELL']:
                    bugs_found.append(f"THRESHOLD VIOLATION: Trade at {proximity}% proximity")
    
    # 5. Check for error patterns
    print("\n5. Checking for error patterns...")
    error_types = defaultdict(int)
    for line in lines:
        if "ERROR" in line or "Exception" in line:
            # Categorize errors
            if "connection" in line.lower():
                error_types["Connection"] += 1
            elif "balance" in line.lower():
                error_types["Balance"] += 1
            elif "order" in line.lower():
                error_types["Order"] += 1
            else:
                error_types["Other"] += 1
    
    for error_type, count in error_types.items():
        if count > 0:
            bugs_found.append(f"ERROR PATTERN: {error_type} errors ({count} occurrences)")
    
    # Report findings
    print("\n" + "=" * 50)
    print("🔍 BUG DETECTION RESULTS:")
    print("=" * 50)
    
    if bugs_found:
        print(f"\n❌ Found {len(bugs_found)} potential bugs:\n")
        for i, bug in enumerate(bugs_found, 1):
            print(f"   {i}. {bug}")
    else:
        print("\n✅ No bugs detected in recent logs!")
    
    # Additional statistics
    print("\n📊 STATISTICS:")
    print(f"   Total lines analyzed: {len(lines)}")
    print(f"   Candle transitions: {len(candle_times)}")
    print(f"   Paper trades: {sum(trade_signatures.values())}")
    print(f"   Errors found: {sum(error_types.values())}")

if __name__ == "__main__":
    analyze_logs()