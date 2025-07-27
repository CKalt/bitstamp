#!/usr/bin/env python3
"""
View TDR server configuration and status via HTTP API
"""
import requests
import json
from datetime import datetime

# Server URL (via SSH tunnel)
SERVER_URL = "http://localhost:4000"

def format_json(data):
    """Pretty print JSON data"""
    return json.dumps(data, indent=2)

def get_api_data(endpoint):
    """Get data from API endpoint"""
    try:
        response = requests.get(f"{SERVER_URL}{endpoint}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error accessing {endpoint}: {e}")
        return None

def main():
    print("TDR Server Configuration Viewer")
    print("=" * 50)
    
    # Check server connectivity
    print("\n1. Checking server connection...")
    if not get_api_data("/api/ping"):
        print("❌ Cannot connect to server at localhost:4000")
        print("   Make sure your SSH tunnel is running:")
        print("   ssh -L 4000:localhost:4000 chriskoin")
        return
    print("✅ Server is reachable")
    
    # Get best_strategy.json
    print("\n2. Current Strategy Configuration (best_strategy.json):")
    print("-" * 50)
    strategy = get_api_data("/api/best_strategy")
    if strategy:
        print(f"Strategy: {strategy.get('Strategy', 'N/A')}")
        print(f"Short MA Window: {strategy.get('Short_Window', 'N/A')}")
        print(f"Long MA Window: {strategy.get('Long_Window', 'N/A')}")
        print(f"Frequency: {strategy.get('Frequency', 'N/A')}")
        print(f"Live Trading: {strategy.get('do_live_trades', False)}")
        avg_trades = strategy.get('Average_Trades_Per_Day', 'N/A')
        if isinstance(avg_trades, (int, float)):
            print(f"Average Trades/Day: {avg_trades:.2f}")
        else:
            print(f"Average Trades/Day: {avg_trades}")
        print(f"Max Trades/Day: {strategy.get('max_trades_per_day', 'N/A')}")
        print(f"Min Time Between Trades: {strategy.get('min_time_between_trades_minutes', 'N/A')} minutes")
    
    # Get detailed status
    print("\n3. Current System Status:")
    print("-" * 50)
    status = get_api_data("/api/status/detailed")
    if status:
        position = status.get('position', {})
        print(f"Position: {position.get('side', 'N/A')}")
        print(f"Amount: {position.get('amount', 0):.8f} BTC")
        print(f"Entry Price: ${position.get('entry_price', 0):,.2f}")
        print(f"Current Price: ${status.get('current_price', 0):,.2f}")
        print(f"Unrealized P&L: ${position.get('unrealized_pnl', 0):,.2f}")
        
        strategy_info = status.get('strategy', {})
        print(f"\nActive Strategy: {strategy_info.get('name', 'N/A')}")
        print(f"Market Regime: {strategy_info.get('regime', 'N/A')}")
        print(f"Auto-trader: {status.get('auto_trader_status', 'N/A')}")
    
    # Get flip distance
    print("\n4. Position Flip Analysis:")
    print("-" * 50)
    flip = get_api_data("/api/strategy/flip_distance")
    if flip:
        print(f"Signal Direction: {flip.get('signal_direction', 'N/A')}")
        print(f"Distance to Flip: ${flip.get('distance_to_flip', 0):.2f} ({flip.get('percentage_to_flip', 0):.2f}%)")
        print(f"Next Action Price: ${flip.get('next_action_price', 0):,.2f}")
    
    # Get recent trades
    print("\n5. Recent Trades:")
    print("-" * 50)
    trades = get_api_data("/api/trades?limit=5")
    if trades and trades.get('trades'):
        for trade in trades['trades']:
            timestamp = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
            print(f"{timestamp.strftime('%Y-%m-%d %H:%M')} - {trade['action']} {trade.get('amount', 0):.8f} BTC @ ${trade['price']:,.2f}")
    else:
        print("No recent trades")
    
    # Get server config
    print("\n6. Full Server Configuration:")
    print("-" * 50)
    config = get_api_data("/api/config")
    if config:
        print("Key configuration values:")
        if 'strategy' in config:
            print(f"  Strategy Type: {config['strategy'].get('strategy_type', 'N/A')}")
        if 'data' in config:
            print(f"  Data Window: {config['data'].get('data_window_days', 'N/A')} days")
        print("\nFor full configuration, check the JSON output above.")

if __name__ == "__main__":
    main()