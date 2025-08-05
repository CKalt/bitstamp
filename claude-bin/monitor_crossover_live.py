#!/usr/bin/env python3
"""
Monitor MA crossover proximity in real-time
Shows exactly when the system will flip from SHORT to LONG
"""
import requests
import time
import json
from datetime import datetime

def monitor_crossover(interval=10):
    """Monitor MA crossover proximity until it triggers"""
    server_url = "http://localhost:4000"
    
    print("📊 MONITORING MA CROSSOVER IN REAL-TIME")
    print("=" * 60)
    
    # Get threshold from config
    with open('best_strategy.json', 'r') as f:
        config = json.load(f)
    threshold = config.get('ma_separation_threshold', 0.3)
    
    print(f"Threshold for triggering: {threshold}%")
    print(f"Monitoring every {interval} seconds...")
    print("\nPress Ctrl+C to stop")
    print("-" * 60)
    
    last_proximity = None
    trigger_predicted = False
    
    try:
        while True:
            # Get current status
            response = requests.get(f"{server_url}/api/status")
            if response.status_code == 200:
                status = response.json()
                auto_trading = status.get('auto_trading', {})
                
                # Extract key values
                position = auto_trading.get('position', 'UNKNOWN')
                current_price = status.get('current_price', {}).get('btcusd', 0)
                
                # Get MA values and proximity
                indicators = status.get('indicators', {}).get('btcusd', {})
                ma4 = indicators.get('MA_4', 0)
                ma20 = indicators.get('MA_20', 0)
                
                # Calculate proximity manually to verify
                if ma20 > 0:
                    proximity_calc = abs((ma4 - ma20) / ma20) * 100
                else:
                    proximity_calc = 0
                
                # Get reported proximity
                diag_data = auto_trading.get('diagnostic_data', {})
                proximity_str = diag_data.get('ma_proximity', '0%')
                proximity = float(proximity_str.strip('%')) if proximity_str else proximity_calc
                
                # Display update
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{timestamp}] Position: {position} | Price: ${current_price:,.0f}")
                print(f"  MA4:  ${ma4:,.2f}")
                print(f"  MA20: ${ma20:,.2f}")
                print(f"  Diff: ${ma4 - ma20:,.2f}")
                print(f"  Proximity: {proximity:.3f}% (threshold: {threshold}%)")
                
                # Show trend
                if last_proximity is not None:
                    if proximity < last_proximity:
                        trend = "↓ APPROACHING TRIGGER"
                    elif proximity > last_proximity:
                        trend = "↑ Moving away"
                    else:
                        trend = "→ Stable"
                    print(f"  Trend: {trend}")
                
                # Check if close to trigger
                if proximity <= threshold and position == "SHORT":
                    print("  🎯 TRIGGER ZONE! Trade should execute soon!")
                    trigger_predicted = True
                elif proximity <= threshold + 0.05:  # Within 0.05% of threshold
                    remaining = proximity - threshold
                    print(f"  ⚠️  VERY CLOSE! Only {remaining:.3f}% to go!")
                
                # Check if trade happened
                if trigger_predicted and position == "LONG":
                    print("\n🎉 TRADE EXECUTED! Position flipped to LONG!")
                    break
                
                last_proximity = proximity
                
            else:
                print(f"\n❌ Failed to get status: {response.status_code}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Last proximity: {proximity:.3f}%")
    print(f"Threshold: {threshold}%")
    if proximity > threshold:
        print(f"Needs to drop by: {proximity - threshold:.3f}% to trigger")
    else:
        print("Should be trading! Check logs for issues.")

if __name__ == "__main__":
    # Run with 10 second intervals by default
    monitor_crossover(10)