#!/usr/bin/env python3
"""
Update TDR server configuration via HTTP API
"""
import requests
import json
import sys

# Server URL (via SSH tunnel)
SERVER_URL = "http://localhost:4000"

def get_current_config():
    """Get current server configuration"""
    try:
        response = requests.get(f"{SERVER_URL}/api/best_strategy")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting configuration: {e}")
        return None

def update_config(config):
    """Update server configuration"""
    try:
        response = requests.post(
            f"{SERVER_URL}/api/best_strategy",
            json=config,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error updating configuration: {e}")
        return None

def check_server_status():
    """Check if server is reachable"""
    try:
        response = requests.get(f"{SERVER_URL}/api/ping")
        response.raise_for_status()
        return True
    except:
        return False

def main():
    print("TDR Server Configuration Updater")
    print("=" * 50)
    
    # Check server connectivity
    print("\n1. Checking server connection...")
    if not check_server_status():
        print("❌ Cannot connect to server at localhost:4000")
        print("   Make sure your SSH tunnel is running:")
        print("   ssh -L 4000:localhost:4000 chriskoin")
        sys.exit(1)
    print("✅ Server is reachable")
    
    # Get current configuration
    print("\n2. Getting current configuration...")
    config = get_current_config()
    if not config:
        sys.exit(1)
    
    print(f"✅ Current configuration:")
    print(f"   Short_Window: {config.get('Short_Window', 'N/A')}")
    print(f"   Long_Window: {config.get('Long_Window', 'N/A')}")
    print(f"   Average_Trades_Per_Day: {config.get('Average_Trades_Per_Day', 'N/A')}")
    
    # Check if already updated
    if config.get('Short_Window') == 4 and config.get('Long_Window') == 20:
        print("\n✅ Configuration is already updated to optimal values!")
        return
    
    # Update configuration
    print("\n3. Updating configuration...")
    print("   Setting Short_Window: 6 → 4")
    print("   Setting Long_Window: 34 → 20")
    
    config['Short_Window'] = 4
    config['Long_Window'] = 20
    
    # Send update
    result = update_config(config)
    if result and result.get('success'):
        print("\n✅ Configuration updated successfully!")
        print("   The server will use the new parameters immediately.")
        print("\n📊 Expected results based on backtest:")
        print("   - Trade frequency: ~1.43 trades per day")
        print("   - Better returns than previous configuration")
        
        # Get detailed status to confirm
        try:
            status_response = requests.get(f"{SERVER_URL}/api/status/detailed")
            if status_response.ok:
                status = status_response.json()
                strategy_config = status.get('strategy', {}).get('config', {})
                print(f"\n🔍 Confirming active strategy parameters:")
                print(f"   Short MA: {strategy_config.get('short_window', 'N/A')}")
                print(f"   Long MA: {strategy_config.get('long_window', 'N/A')}")
        except:
            pass
            
    else:
        print("❌ Failed to update configuration")
        if result:
            print(f"   Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()