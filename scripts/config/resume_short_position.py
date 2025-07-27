#!/usr/bin/env python3
"""
Helper script to calculate and show resume command for SHORT position
"""

# Your position details
usd_holdings = 170577  # USD amount you're holding
entry_price = 117564   # Price at which you sold BTC

# Calculate how much BTC you sold
# USD = BTC * Price * (1 - fee)
# BTC = USD / (Price * (1 - fee))
fee_rate = 0.0012  # Bitstamp fee
btc_sold = usd_holdings / (entry_price * (1 - fee_rate))

print("📊 SHORT Position Resume Details")
print("=" * 50)
print(f"Entry Price (SELL): ${entry_price:,.2f}")
print(f"USD Holdings: ${usd_holdings:,.2f}")
print(f"Estimated BTC Sold: {btc_sold:.8f} BTC")
print(f"Fee Rate: {fee_rate * 100}%")

print("\n🔧 Resume Commands:")
print("-" * 50)

# Method 1: Using USD amount
print("Method 1 - Using USD amount:")
print(f"resume_auto_trade {usd_holdings}usd short {entry_price}")

# Method 2: Using BTC amount
print(f"\nMethod 2 - Using BTC amount sold:")
print(f"resume_auto_trade {btc_sold:.8f}btc short {entry_price}")

print("\n📝 Full TDR Client Commands:")
print("-" * 50)
print("1. Start TDR client:")
print("   cd /Users/chris/projects/python/btc")
print("   source env/bin/activate")
print("   python src/tdr.py")

print("\n2. Enable commands:")
print("   tdr> enable_commands")

print("\n3. Resume auto-trading (choose one):")
print(f"   tdr> resume_auto_trade {usd_holdings}usd short {entry_price}")
print("   OR")
print(f"   tdr> resume_auto_trade {btc_sold:.8f}btc short {entry_price}")

print("\n⚠️  Important Notes:")
print("- You're SHORT (holding USD, no BTC)")
print("- Entry price is where you SOLD BTC")
print("- System will track this as a short position")
print("- Will BUY BTC when MA signal turns LONG")