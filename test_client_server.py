#!/usr/bin/env python
"""
Test script for TDR client-server functionality
"""

import requests
import json
import time
import sys

def test_server(server_url="http://localhost:4000"):
    """Test basic server functionality"""
    
    print(f"Testing TDR server at {server_url}")
    print("=" * 50)
    
    # Test 1: Ping
    print("\n1. Testing ping endpoint...")
    try:
        response = requests.get(f"{server_url}/api/ping", timeout=5)
        if response.status_code == 200:
            print("✓ Ping successful:", response.json())
        else:
            print("✗ Ping failed:", response.status_code)
    except Exception as e:
        print("✗ Cannot connect to server:", e)
        return False
    
    # Test 2: Status
    print("\n2. Testing status endpoint...")
    try:
        response = requests.get(f"{server_url}/api/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print("✓ Status retrieved:")
            print(f"  - Server: {status.get('server')}")
            print(f"  - WebSocket: {status.get('websocket')}")
            print(f"  - Live Trading: {status.get('live_trading')}")
            if 'position' in status:
                pos = status['position']
                print(f"  - Position: BTC={pos['btc_balance']:.8f}, USD=${pos['usd_balance']:.2f}")
        else:
            print("✗ Status failed:", response.status_code)
    except Exception as e:
        print("✗ Status error:", e)
    
    # Test 3: Price
    print("\n3. Testing price endpoint...")
    try:
        response = requests.get(f"{server_url}/api/price/btcusd", timeout=5)
        if response.status_code == 200:
            price_data = response.json()
            print(f"✓ BTC/USD Price: ${price_data['price']:.2f}")
        else:
            print("✗ Price failed:", response.status_code)
    except Exception as e:
        print("✗ Price error:", e)
    
    # Test 4: Command execution
    print("\n4. Testing command execution...")
    try:
        response = requests.post(
            f"{server_url}/api/command",
            json={"command": "help"},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print("✓ Command executed successfully")
            # Show first few lines of output
            output_lines = result.get('output', '').split('\n')[:5]
            for line in output_lines:
                if line.strip():
                    print(f"  {line}")
            if len(result.get('output', '').split('\n')) > 5:
                print("  ...")
        else:
            print("✗ Command failed:", response.status_code)
    except Exception as e:
        print("✗ Command error:", e)
    
    # Test 5: Data endpoint
    print("\n5. Testing data endpoint...")
    try:
        response = requests.get(
            f"{server_url}/api/data/btcusd",
            params={"limit": 5, "frequency": "1H"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print("✓ Data retrieved:")
            if data.get('data'):
                print(f"  Got {len(data['data'])} data points")
                latest = data['data'][-1] if data['data'] else None
                if latest:
                    print(f"  Latest: {latest.get('time', 'N/A')} - Close: ${latest.get('close', 0):.2f}")
            else:
                print("  No data available yet")
        else:
            print("✗ Data failed:", response.status_code)
    except Exception as e:
        print("✗ Data error:", e)
    
    print("\n" + "=" * 50)
    print("Testing complete!")
    return True

if __name__ == "__main__":
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:4000"
    test_server(server_url)