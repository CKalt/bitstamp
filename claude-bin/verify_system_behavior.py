#!/usr/bin/env python3
"""
Verify system behavior by monitoring signal evaluations and checking decisions
"""
import time
import json
import requests
from datetime import datetime
import subprocess

class SystemVerifier:
    def __init__(self, server_url="http://localhost:4000"):
        self.server_url = server_url
        self.last_eval_time = None
        self.last_proximity = None
        self.last_position = None
        self.expected_trade = False
        
    def get_status(self):
        """Get current system status"""
        try:
            response = requests.get(f"{self.server_url}/api/status")
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def get_ma_status(self):
        """Get MA crossover details"""
        try:
            response = requests.post(
                f"{self.server_url}/api/command",
                json={"command": "status"}
            )
            if response.status_code == 200:
                output = response.json().get('output', '')
                # Extract proximity
                if "MA Crossover Proximity:" in output:
                    prox_line = [l for l in output.split('\n') if 'MA Crossover Proximity:' in l][0]
                    proximity = float(prox_line.split(':')[1].strip().rstrip('%'))
                    return proximity
        except:
            pass
        return None
    
    def get_recent_logs(self, lines=50):
        """Get recent log entries"""
        try:
            # Read log file directly
            result = subprocess.run(
                ['tail', '-n', str(lines), 'logs/tdr_server.log'],
                capture_output=True, text=True
            )
            return result.stdout.split('\n') if result.returncode == 0 else []
        except:
            return []
    
    def check_signal_evaluation(self):
        """Verify signal is being evaluated every 30 seconds"""
        logs = self.get_recent_logs(100)
        eval_logs = [l for l in logs if 'SIGNAL_EVAL' in l]
        
        if eval_logs:
            # Parse most recent evaluation
            latest = eval_logs[-1]
            timestamp_str = latest.split(' - ')[0]
            
            # Extract MA values if present
            if 'MA4=' in latest and 'MA20=' in latest:
                ma4 = float(latest.split('MA4=')[1].split()[0])
                ma20 = float(latest.split('MA20=')[1].split()[0])
                
                # Verify proximity calculation
                calculated_prox = abs(ma4 - ma20) / ma20 * 100
                
                if 'Prox=' in latest:
                    logged_prox = float(latest.split('Prox=')[1].split('%')[0])
                    
                    if abs(calculated_prox - logged_prox) > 0.01:
                        print(f"⚠️  PROXIMITY MISMATCH: Calculated {calculated_prox:.3f}% vs Logged {logged_prox:.3f}%")
                        return False
                        
            return True
        return False
    
    def verify_trade_decision(self, position, signal, proximity, threshold=0.3):
        """Verify if trade decision is correct"""
        should_trade = (position != signal and proximity <= threshold)
        
        if should_trade:
            # Check for blocking conditions in logs
            logs = self.get_recent_logs(20)
            blockers = []
            
            for log in logs:
                if "Reached daily trade limit" in log:
                    blockers.append("Daily limit reached")
                elif "grace period" in log:
                    blockers.append("In startup grace period")
                elif "min time between trades" in log:
                    blockers.append("Too soon after last trade")
                    
            return should_trade, blockers
        
        return should_trade, []
    
    def monitor(self):
        """Main monitoring loop"""
        print("🔍 SYSTEM BEHAVIOR VERIFICATION")
        print("=" * 60)
        print("Monitoring signal evaluations and trade decisions...")
        print("Press Ctrl+C to stop\n")
        
        # Get initial config
        with open('best_strategy.json', 'r') as f:
            config = json.load(f)
        threshold = config.get('ma_separation_threshold', 0.3)
        
        print(f"Configuration:")
        print(f"  MA Threshold: {threshold}%")
        print(f"  Max trades/day: {config.get('max_trades_per_day', 5)}")
        print(f"  Max trades/hour: {config.get('max_trades_per_hour', 2)}")
        print(f"  Live trading: {config.get('do_live_trades', False)}")
        print("-" * 60)
        
        last_check = time.time()
        
        try:
            while True:
                current_time = time.time()
                
                # Check every 35 seconds (slightly after evaluation)
                if current_time - last_check >= 35:
                    status = self.get_status()
                    proximity = self.get_ma_status()
                    
                    if status and proximity is not None:
                        position = status['position']['position']
                        pos_name = "LONG" if position == 1 else "SHORT" if position == -1 else "NEUTRAL"
                        
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Check:")
                        print(f"  Position: {pos_name}")
                        print(f"  Proximity: {proximity:.3f}%")
                        
                        # Check if signal evaluation happened
                        if self.check_signal_evaluation():
                            print("  ✅ Signal evaluated")
                        else:
                            print("  ⚠️  No recent signal evaluation!")
                        
                        # Determine expected signal
                        if proximity <= threshold:
                            print(f"  🎯 IN TRIGGER ZONE! (proximity {proximity:.3f}% <= {threshold}%)")
                            
                            # Check if trade should happen
                            logs = self.get_recent_logs(10)
                            trade_logs = [l for l in logs if 'Executing trade' in l or 'Buy signal triggered' in l or 'Sell signal triggered' in l]
                            
                            if trade_logs:
                                print("  ✅ Trade executed/triggered")
                            else:
                                print("  ⚠️  No trade despite being in trigger zone - checking why...")
                                
                                # Look for blockers
                                blocker_logs = [l for l in logs if any(x in l for x in ['limit', 'grace', 'time between'])]
                                if blocker_logs:
                                    print(f"  Blocked by: {blocker_logs[-1]}")
                        
                        # Track proximity trend
                        if self.last_proximity is not None:
                            trend = "↓" if proximity < self.last_proximity else "↑" if proximity > self.last_proximity else "→"
                            print(f"  Trend: {trend} (was {self.last_proximity:.3f}%)")
                        
                        self.last_proximity = proximity
                        self.last_position = position
                    
                    last_check = current_time
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    verifier = SystemVerifier()
    verifier.monitor()