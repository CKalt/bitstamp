#!/usr/bin/env python3
"""
Dry-run test to force signal changes and verify trading functionality
This test is designed with minimal risk - it uses a separate test configuration
and does NOT modify any production files or settings.
"""
import os
import sys
import json
import time
import shutil
import requests
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DryRunTradingTest:
    def __init__(self, server_url="http://localhost:4000"):
        self.server_url = server_url
        self.test_dir = "test_dry_run"
        self.original_best_strategy = "best_strategy.json"
        self.test_best_strategy = os.path.join(self.test_dir, "best_strategy.json")
        self.test_btcusd_log = os.path.join(self.test_dir, "btcusd.log")
        
    def setup_test_environment(self):
        """Create isolated test environment"""
        print("\n🔧 Setting up test environment...")
        
        # Create test directory
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Copy best_strategy.json and modify for dry-run
        with open(self.original_best_strategy, 'r') as f:
            config = json.load(f)
        
        # Force dry-run mode
        config['do_live_trades'] = False
        config['strategy_type'] = 'MA'
        config['enable_adaptive_strategy'] = False
        config['Short_Window'] = 4  # Use original MA4
        config['Long_Window'] = 20   # Use original MA20
        
        with open(self.test_best_strategy, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Created test configuration with do_live_trades=False")
        
    def create_test_price_data(self, current_price, signal_type):
        """Create synthetic price data to force specific signals"""
        print(f"\n📊 Creating test price data for {signal_type} signal...")
        
        now = datetime.now()
        timestamps = []
        prices = []
        
        # Generate 50 data points (enough for MA20 + buffer)
        for i in range(50):
            timestamp = now - timedelta(hours=(49-i))
            timestamps.append(int(timestamp.timestamp()))
            
            if signal_type == "LONG":
                # Create uptrend: MA4 > MA20
                if i < 30:
                    # Historical prices - downtrend
                    price = current_price - 1000 + (i * 10)
                else:
                    # Recent prices - strong uptrend
                    price = current_price - 500 + ((i - 30) * 50)
            else:  # SHORT
                # Create downtrend: MA4 < MA20
                if i < 30:
                    # Historical prices - uptrend
                    price = current_price + 1000 - (i * 10)
                else:
                    # Recent prices - strong downtrend
                    price = current_price + 500 - ((i - 30) * 50)
            
            prices.append(price)
        
        # Write to test log file
        with open(self.test_btcusd_log, 'w') as f:
            for ts, price in zip(timestamps, prices):
                f.write(f"{ts},{price:.2f},0.01\n")
        
        # Calculate expected MAs
        ma4 = sum(prices[-4:]) / 4
        ma20 = sum(prices[-20:]) / 20
        
        print(f"✅ Created test data: MA4={ma4:.0f}, MA20={ma20:.0f}, Diff={ma4-ma20:.0f}")
        return ma4, ma20
    
    def check_server_status(self):
        """Check if server is running and get current status"""
        try:
            response = requests.get(f"{self.server_url}/api/status")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Server returned status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Could not connect to server: {e}")
            return None
    
    def initialize_test_server(self):
        """Initialize server with test configuration"""
        print("\n🚀 Initializing server with test configuration...")
        
        # Read test configuration
        with open(self.test_best_strategy, 'r') as f:
            test_config = json.load(f)
        
        # Prepare initialization payload
        init_payload = {
            "best_strategy": test_config,
            "verbose": True,
            "test_mode": True,  # Force test mode
            "auto_resume": False,  # Don't auto-resume in test
            "btcusd_log_path": self.test_btcusd_log  # Use test data
        }
        
        try:
            response = requests.post(f"{self.server_url}/api/init", json=init_payload)
            if response.status_code == 200:
                print("✅ Server initialized with test configuration")
                return True
            else:
                print(f"❌ Failed to initialize: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error initializing server: {e}")
            return False
    
    def start_auto_trading(self, position="LONG", amount=0.01, entry_price=100000):
        """Start auto-trading with specific position"""
        print(f"\n🤖 Starting auto-trading in {position} position...")
        
        auto_trade_params = {
            "resume_position": position,
            "resume_amount": amount,
            "resume_entry_price": entry_price
        }
        
        try:
            response = requests.post(f"{self.server_url}/api/autotrade/start", json=auto_trade_params)
            if response.status_code == 200:
                print("✅ Auto-trading started")
                return True
            else:
                print(f"❌ Failed to start auto-trading: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error starting auto-trading: {e}")
            return False
    
    def force_signal_evaluation(self):
        """Force immediate signal evaluation"""
        try:
            response = requests.post(f"{self.server_url}/api/signal/evaluate")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to evaluate signal: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Error evaluating signal: {e}")
            return None
    
    def monitor_trades(self, duration=30):
        """Monitor for trade execution"""
        print(f"\n👀 Monitoring for trades over {duration} seconds...")
        
        start_time = time.time()
        last_status = None
        trade_detected = False
        
        while time.time() - start_time < duration:
            status = self.check_server_status()
            if status and status.get('auto_trading', {}).get('active'):
                current_position = status['auto_trading'].get('position')
                current_signal = status['auto_trading'].get('latest_signal')
                
                # Check if position changed
                if last_status and last_status.get('position') != current_position:
                    print(f"\n🎯 TRADE DETECTED! Position flipped from {last_status.get('position')} to {current_position}")
                    trade_detected = True
                
                # Display current state
                print(f"\r⏱️  {int(time.time() - start_time)}s - Position: {current_position}, Signal: {current_signal}", end="", flush=True)
                
                last_status = {
                    'position': current_position,
                    'signal': current_signal
                }
            
            time.sleep(1)
        
        print("\n")
        return trade_detected
    
    def run_test_scenario(self, initial_position, target_signal):
        """Run a complete test scenario"""
        print(f"\n{'='*60}")
        print(f"TEST SCENARIO: {initial_position} position → {target_signal} signal")
        print(f"{'='*60}")
        
        # Create price data for target signal
        current_price = 119000
        ma4, ma20 = self.create_test_price_data(current_price, target_signal)
        
        # Initialize server
        if not self.initialize_test_server():
            return False
        
        # Start auto-trading with initial position
        if not self.start_auto_trading(position=initial_position):
            return False
        
        # Wait for initialization
        time.sleep(2)
        
        # Force signal evaluation
        print("\n📡 Forcing signal evaluation...")
        eval_result = self.force_signal_evaluation()
        if eval_result:
            print(f"Signal evaluation result: {eval_result}")
        
        # Monitor for trades
        trade_detected = self.monitor_trades(duration=20)
        
        if trade_detected:
            print("✅ TEST PASSED: Trade executed successfully!")
            return True
        else:
            print("❌ TEST FAILED: No trade detected")
            
            # Get final status for debugging
            final_status = self.check_server_status()
            if final_status:
                print("\nFinal server status:")
                print(json.dumps(final_status.get('auto_trading', {}), indent=2))
            
            return False
    
    def cleanup(self):
        """Clean up test environment"""
        print("\n🧹 Cleaning up test environment...")
        
        # Stop auto-trading
        try:
            requests.post(f"{self.server_url}/api/autotrade/stop")
        except:
            pass
        
        # Remove test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print("✅ Test directory removed")
    
    def run_all_tests(self):
        """Run all test scenarios"""
        print("\n🧪 DRY-RUN TRADING TEST SUITE")
        print("This test verifies trading functionality without risking real money")
        print("="*60)
        
        # Check server is running
        status = self.check_server_status()
        if not status:
            print("\n❌ ERROR: Server is not running!")
            print("Please start the server with: python src/tdr.py --server")
            return
        
        print(f"✅ Server is running at {self.server_url}")
        
        try:
            # Setup test environment
            self.setup_test_environment()
            
            # Test 1: SHORT position should flip to LONG when MA4 > MA20
            test1_passed = self.run_test_scenario("SHORT", "LONG")
            
            # Small delay between tests
            time.sleep(5)
            
            # Test 2: LONG position should flip to SHORT when MA4 < MA20
            test2_passed = self.run_test_scenario("LONG", "SHORT")
            
            # Summary
            print("\n" + "="*60)
            print("TEST SUMMARY:")
            print(f"  Test 1 (SHORT→LONG): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
            print(f"  Test 2 (LONG→SHORT): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
            print("="*60)
            
            if test1_passed and test2_passed:
                print("\n✅ ALL TESTS PASSED! Trading system is working correctly.")
                print("\n📌 IMPORTANT: These tests used do_live_trades=False (dry-run mode)")
                print("Your production configuration remains unchanged.")
            else:
                print("\n❌ SOME TESTS FAILED! Please check the logs for details.")
                print("\nPossible issues:")
                print("- Signal evaluation might not be triggering trades")
                print("- Position flip logic might have additional conditions")
                print("- Check server logs for detailed error messages")
        
        finally:
            # Always cleanup
            self.cleanup()
            
            # Restore server to normal state
            print("\n⚠️  Server needs to be reinitialized for normal operation")
            print("Restart the server or reinitialize with production config")


if __name__ == "__main__":
    # Run the test
    test = DryRunTradingTest()
    test.run_all_tests()