#!/usr/bin/env python3
"""
Check current trading activity on the server
"""
import requests
import json
from datetime import datetime

SERVER_URL = "http://localhost:4000"

# Get recent trades
print("=== RECENT TRADES ===")
try:
    response = requests.get(f"{SERVER_URL}/api/trades?limit=10")
    if response.ok:
        data = response.json()
        trades = data.get('trades', [])
        if trades:
            for trade in trades:
                timestamp = trade.get('timestamp', 'N/A')
                if timestamp != 'N/A':
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    time_str = timestamp
                print(f"{time_str} - {trade.get('action')} {trade.get('amount', 0):.8f} BTC @ ${trade.get('price', 0):,.2f}")
        else:
            print("No recent trades")
except Exception as e:
    print(f"Error getting trades: {e}")

# Get system status
print("\n=== SYSTEM STATUS ===")
try:
    response = requests.get(f"{SERVER_URL}/api/status")
    if response.ok:
        data = response.json()
        status = data.get('result', {})
        print(f"Auto-trader: {status.get('auto_trader_status', 'Unknown')}")
        print(f"Position: {status.get('position', 'Unknown')}")
        print(f"Current Price: ${status.get('current_price', 0):,.2f}")
        
        # Check flip distance
        flip_response = requests.get(f"{SERVER_URL}/api/strategy/flip_distance")
        if flip_response.ok:
            flip_data = flip_response.json()
            print(f"\nSignal Analysis:")
            print(f"  Current MA Signal: {flip_data.get('signal_direction', 'N/A')}")
            print(f"  Distance to Flip: ${flip_data.get('distance_to_flip', 0):.2f} ({flip_data.get('percentage_to_flip', 0):.2f}%)")
except Exception as e:
    print(f"Error getting status: {e}")

# Get recent logs to see what's happening
print("\n=== RECENT SERVER ACTIVITY ===")
try:
    response = requests.get(f"{SERVER_URL}/api/logs?lines=20&type=auto_trader")
    if response.ok:
        data = response.json()
        logs = data.get('logs', [])
        print("Last 20 auto-trader log entries:")
        for log in logs:
            print(f"  {log}")
except Exception as e:
    print(f"Error getting logs: {e}")

# Check diagnostics for signal evaluations
print("\n=== RECENT SIGNAL EVALUATIONS ===")
try:
    response = requests.get(f"{SERVER_URL}/api/diagnostics?event_type=SIGNAL_EVAL&count=5")
    if response.ok:
        data = response.json()
        events = data.get('events', [])
        if events:
            print("Last 5 signal evaluations:")
            for event in events:
                print(f"  {event.get('timestamp')} - {event.get('details', {}).get('reason', 'N/A')}")
        else:
            print("No recent signal evaluations")
except Exception as e:
    print(f"Error getting diagnostics: {e}")