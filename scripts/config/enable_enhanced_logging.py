#!/usr/bin/env python3
"""
Enable enhanced logging via API without restarting server
This will help us see exactly what happens when we hit the trigger
"""
import requests
import json

server_url = "http://localhost:4000"

print("🔧 Enabling Enhanced Logging...")

# Update configuration to enable verbose logging
config_updates = {
    "log_signal_evaluation": True,
    "verbose_logging": True
}

# Send config update command
for key, value in config_updates.items():
    cmd = f"config {key} {str(value).lower()}"
    response = requests.post(
        f"{server_url}/api/command",
        json={"command": cmd}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Set {key} = {value}")
        if result.get('output'):
            print(f"   {result['output'].strip()}")
    else:
        print(f"❌ Failed to set {key}")

print("\n📋 Current Configuration:")
# Get current config
response = requests.post(
    f"{server_url}/api/command",
    json={"command": "config"}
)

if response.status_code == 200:
    result = response.json()
    output = result.get('output', '')
    # Parse relevant settings
    for line in output.split('\n'):
        if any(x in line for x in ['log_signal', 'verbose', 'threshold', 'trades_per']):
            print(f"   {line.strip()}")

print("\n✅ Enhanced logging enabled!")
print("📌 Logs will now show:")
print("   - Every signal evaluation (every 30 seconds)")
print("   - MA values and proximity calculations")
print("   - Trade decision logic")
print("\nMonitor logs with: tail -f logs/tdr_server.log | grep SIGNAL_EVAL")