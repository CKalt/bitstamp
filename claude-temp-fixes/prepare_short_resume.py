#!/usr/bin/env python3
"""
Prepare system to resume with a SHORT position at specified entry price
This is for when you want to manually override to a SHORT position
"""
import json
import os
import sys
from datetime import datetime

def save_json_file(filepath, data):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {filepath}")

def prepare_short_position(usd_amount, entry_price):
    """Prepare resume files for SHORT position."""
    
    # Create resume data for SHORT position
    resume_data = {
        "position": "SHORT",
        "amount": usd_amount,
        "unit": "usd",
        "entry_price": entry_price,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "command": f"resume_auto_trade {usd_amount:.2f}usd short {int(entry_price)}",
        "strategy": "MA",
        "ma_short": 4,
        "ma_long": 20,
        "source": "manual_override"
    }
    
    return resume_data

def update_best_strategy(entry_price):
    """Update best_strategy.json for SHORT position."""
    # Load existing config
    config = {}
    if os.path.exists('best_strategy.json'):
        with open('best_strategy.json', 'r') as f:
            config = json.load(f)
    
    config['Last_Trade_Price'] = entry_price
    config['Last_Trade_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config['auto_resume'] = True
    
    return config

def main():
    """Main function."""
    print("=" * 60)
    print("PREPARE SHORT POSITION RESUME")
    print("=" * 60)
    
    # Get parameters
    if len(sys.argv) != 3:
        print("\nUsage: python prepare_short_resume.py <usd_amount> <entry_price>")
        print("Example: python prepare_short_resume.py 172000 117987")
        sys.exit(1)
    
    try:
        usd_amount = float(sys.argv[1])
        entry_price = float(sys.argv[2])
    except ValueError:
        print("❌ Invalid parameters. Please provide numeric values.")
        sys.exit(1)
    
    # Change to server directory
    os.chdir('/home/chris/projects/bitstamp')
    
    print(f"\n📋 Preparing SHORT position:")
    print(f"   USD Amount: ${usd_amount:,.2f}")
    print(f"   Entry Price: ${entry_price:,.2f}")
    
    # Backup existing files
    print("\n📦 Creating backups...")
    for filename in ['resume-auto-trade.json', 'best_strategy.json']:
        if os.path.exists(filename):
            backup_name = f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(filename, backup_name)
            print(f"   Backed up {filename} → {backup_name}")
    
    # Create resume file
    print("\n🔧 Creating resume-auto-trade.json...")
    resume_data = prepare_short_position(usd_amount, entry_price)
    save_json_file('resume-auto-trade.json', resume_data)
    print(f"   Command: {resume_data['command']}")
    
    # Update best_strategy.json
    print("\n🔧 Updating best_strategy.json...")
    config = update_best_strategy(entry_price)
    save_json_file('best_strategy.json', config)
    
    print("\n" + "=" * 60)
    print("✅ SHORT POSITION PREPARED!")
    print("=" * 60)
    print("\n⚠️  IMPORTANT: This sets up tracking for a SHORT position but does NOT sell your BTC!")
    print("\nNext steps:")
    print("1. If you have BTC, you need to manually SELL it first")
    print("2. Start the server: python src/tdr.py --server")
    print("3. Wait for history to load (~3-5 minutes)")
    print("4. The server should auto-resume with the SHORT position")
    print("\nManual resume command if needed:")
    print(f"   {resume_data['command']}")

if __name__ == "__main__":
    main()