#!/usr/bin/env python3
"""
Quick signal monitoring check script.
Shows signal evaluation status and alerts for any issues.
"""

import json
import os
import sys
from datetime import datetime, timedelta
import subprocess

def check_signal_health():
    """Check if signals are being evaluated regularly."""
    
    print("🔍 Signal Monitoring Health Check")
    print("=" * 50)
    
    # Check local API
    try:
        import requests
        response = requests.get("http://localhost:4000/api/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: Connected")
            print(f"📊 Price: ${data.get('last_price', 0):.0f}")
            print(f"📍 Position: {data.get('position', {}).get('position', 'Unknown')}")
        else:
            print(f"⚠️  API returned status code: {response.status_code}")
    except Exception as e:
        print(f"❌ API Error: {e}")
    
    print()
    
    # Check signal logs on server
    try:
        # Get last heartbeat
        cmd = 'ssh ck "grep HEARTBEAT ~/projects/bitstamp/logs/tdr_server.log | tail -1"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            # Extract timestamp from log
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                log_time_str = f"{parts[0]} {parts[1]}"
                print(f"💓 Last Heartbeat: {log_time_str}")
                
                # Check if recent
                try:
                    # Simple time comparison - just check if it's today
                    today = datetime.now().strftime("%Y-%m-%d")
                    if today in log_time_str:
                        print("✅ Heartbeat is recent (today)")
                    else:
                        print("⚠️  WARNING: Heartbeat is old!")
                except:
                    pass
        else:
            print("❌ No heartbeats found!")
            
        print()
        
        # Get last signal evaluation
        cmd = 'ssh ck "grep SIGNAL_EVAL ~/projects/bitstamp/logs/tdr_server.log | tail -1"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                log_time_str = f"{parts[0]} {parts[1]}"
                print(f"📊 Last Signal Eval: {log_time_str}")
                
                # Extract signal details
                if "Sig=" in result.stdout:
                    import re
                    sig_match = re.search(r'Sig=(-?\d+)', result.stdout)
                    pos_match = re.search(r'Pos=(-?\d+)', result.stdout)
                    action_match = re.search(r'Action=(\w+)', result.stdout)
                    
                    if sig_match and pos_match:
                        signal = int(sig_match.group(1))
                        position = int(pos_match.group(1))
                        action = action_match.group(1) if action_match else "Unknown"
                        
                        print(f"   Signal: {signal}, Position: {position}, Action: {action}")
                        
                        # Check if signal would trigger trade
                        if (signal == 1 and position <= 0) or (signal == -1 and position >= 0):
                            print("   🎯 Signal would trigger trade (if not blocked)")
                        else:
                            print("   ⏸️  Signal matches position (no trade)")
        else:
            print("❌ No signal evaluations found!")
            
        print()
        
        # Count evaluations in last hour
        cmd = 'ssh ck "grep SIGNAL_EVAL ~/projects/bitstamp/logs/tdr_server.log | grep \\"$(date +%Y-%m-%d)\\" | tail -20 | wc -l"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            count = int(result.stdout.strip())
            print(f"📈 Recent Activity:")
            print(f"   Evaluations in last 20 entries: {count}")
            
            if count == 0:
                print("   ⚠️  WARNING: No recent evaluations!")
            elif count < 10:
                print("   ⚠️  Low evaluation count")
            else:
                print("   ✅ Good evaluation frequency")
                
    except Exception as e:
        print(f"❌ SSH Error: {e}")
    
    print()
    print("💡 Tips:")
    print("   - Heartbeats should appear every 30 seconds")
    print("   - Signal evaluations happen every 30 seconds")
    print("   - Check logs if no recent activity")
    print("   - Run 'tail -f logs/tdr_server.log' to monitor live")

if __name__ == "__main__":
    check_signal_health()