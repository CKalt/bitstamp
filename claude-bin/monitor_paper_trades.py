#!/usr/bin/env python3
"""
Monitor paper trading performance and proximity threshold effectiveness
"""
import json
import sys
from datetime import datetime, timedelta
from collections import defaultdict

def analyze_logs(log_file):
    """Analyze paper trading logs for threshold effectiveness"""
    
    proximity_blocks = 0
    would_have_traded = 0
    signal_evals = 0
    proximity_values = []
    
    print("📊 PAPER TRADING ANALYSIS")
    print("=" * 60)
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'SIGNAL_EVAL v2:' in line:
                signal_evals += 1
                
                # Extract proximity value
                if 'Prox=' in line:
                    try:
                        prox_str = line.split('Prox=')[1].split('%')[0]
                        prox = float(prox_str)
                        proximity_values.append(prox)
                    except:
                        pass
                
                # Check for proximity blocks
                if 'NO_TRADE_PROXIMITY' in line:
                    proximity_blocks += 1
                    print(f"🚫 Proximity block at: {line.split()[0]} {line.split()[1]}")
                
                # Check for would-have trades
                elif 'WOULD EXECUTE' in line or 'WILL_BUY' in line or 'WILL_SELL' in line:
                    would_have_traded += 1
                    print(f"📈 Would trade at: {line.split()[0]} {line.split()[1]}")
    
    # Analysis
    print(f"\n📊 SUMMARY:")
    print(f"- Total signal evaluations: {signal_evals}")
    print(f"- Trades blocked by proximity: {proximity_blocks}")
    print(f"- Trades that would execute: {would_have_traded}")
    
    if proximity_values:
        avg_prox = sum(proximity_values) / len(proximity_values)
        print(f"\n📏 Proximity Statistics:")
        print(f"- Average proximity: {avg_prox:.3f}%")
        print(f"- Min proximity: {min(proximity_values):.3f}%")
        print(f"- Max proximity: {max(proximity_values):.3f}%")
        
        # Show distribution
        under_threshold = sum(1 for p in proximity_values if p <= 0.5)
        print(f"- Under 0.5% threshold: {under_threshold}/{len(proximity_values)} ({under_threshold/len(proximity_values)*100:.1f}%)")
    
    print(f"\n💰 ESTIMATED SAVINGS:")
    print(f"- Flips prevented: {proximity_blocks}")
    print(f"- Est. savings: ${proximity_blocks * 400:.2f} (assuming $400/flip cost)")
    
    return {
        'proximity_blocks': proximity_blocks,
        'would_have_traded': would_have_traded,
        'signal_evals': signal_evals,
        'proximity_values': proximity_values
    }

if __name__ == "__main__":
    # Default to server log
    log_file = sys.argv[1] if len(sys.argv) > 1 else "/home/chris/projects/bitstamp/logs/tdr_server.log"
    analyze_logs(log_file)