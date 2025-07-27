#!/usr/bin/env python3
"""
Configure server for pure MA trading as backtested
"""
import requests
import json

SERVER_URL = "http://localhost:4000"

print("Configuring Pure MA Strategy (as backtested)")
print("=" * 50)

# Get current configuration
response = requests.get(f"{SERVER_URL}/api/best_strategy")
if not response.ok:
    print("❌ Failed to connect to server")
    exit(1)

current = response.json()['best_strategy']
print(f"\n✅ Connected to server")
print(f"Current: MA {current.get('Short_Window')}/{current.get('Long_Window')}")

# Update configuration for pure MA trading
updates = {
    # Core MA parameters from backtest
    "Strategy": "MA",
    "Short_Window": 4,
    "Long_Window": 20,
    "Frequency": "1H",
    
    # Disable all adaptive features
    "enable_pivot_protection": False,
    "enable_regime_detection": False,
    "enable_adaptive_strategy": False,
    "strategy_type": "MA",
    
    # Keep essential safety parameters
    "do_live_trades": True,
    "max_trades_per_day": 5,
    "min_time_between_trades_minutes": 30,  # Reduced from 120 to allow ~1.4 trades/day
    
    # Disable other features that weren't in backtest
    "use_dynamic_pivots": False,
    "volatility_adjusted_pivots": False,
    "reduce_size_after_whipsaw": False,
    "require_confirmation_bars": 0,  # Immediate execution like backtest
}

# Merge with current config
for key, value in updates.items():
    current[key] = value

# Send update
print("\n📤 Sending configuration update...")
response = requests.post(
    f"{SERVER_URL}/api/best_strategy",
    json=current,
    headers={'Content-Type': 'application/json'}
)

if response.ok and response.json().get('success'):
    print("✅ Configuration updated successfully!")
    print("\n🎯 Pure MA Strategy Active:")
    print(f"   • MA {updates['Short_Window']}/{updates['Long_Window']} crossover")
    print(f"   • No pivot protection")
    print(f"   • No adaptive switching") 
    print(f"   • No regime detection")
    print(f"   • Immediate signal execution")
    print(f"\nThis matches the backtested strategy that showed 9.61% returns.")
else:
    print("❌ Failed to update configuration")
    print(response.text)