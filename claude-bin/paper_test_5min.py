#!/usr/bin/env python3
"""
Safe 5-minute paper trading test
Runs independently without modifying core strategy files
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time

class PaperTradingTest:
    def __init__(self, config):
        self.config = config
        self.position = 0  # -1 SHORT, 0 NEUTRAL, 1 LONG
        self.btc_balance = 0
        self.usd_balance = 10000
        self.trades = []
        self.proximity_blocks = 0
        self.signals_evaluated = 0
        
    def calculate_ma(self, prices, window):
        """Calculate simple moving average"""
        if len(prices) < window:
            return None
        return sum(prices[-window:]) / window
    
    def evaluate_signal(self, current_price, prices):
        """Evaluate MA crossover signal with proximity threshold"""
        self.signals_evaluated += 1
        
        # Calculate MAs
        ma_short = self.calculate_ma(prices, self.config['Short_Window'])
        ma_long = self.calculate_ma(prices, self.config['Long_Window'])
        
        if ma_short is None or ma_long is None:
            return None, "Insufficient data"
        
        # Calculate proximity
        ma_diff = ma_short - ma_long
        proximity = abs(ma_diff) / ma_long * 100
        
        # Determine signal
        signal = 1 if ma_short > ma_long else -1
        
        # Apply proximity threshold
        threshold = self.config.get('proximity_threshold', 0.5)
        
        result = {
            'timestamp': datetime.now(),
            'price': current_price,
            'ma_short': ma_short,
            'ma_long': ma_long,
            'proximity': proximity,
            'signal': signal,
            'position': self.position,
            'threshold': threshold
        }
        
        # Check if we should trade
        if proximity <= threshold:
            self.proximity_blocks += 1
            result['action'] = 'NO_TRADE_PROXIMITY'
            result['reason'] = f'MAs too close: {proximity:.2f}% <= {threshold}%'
        elif signal != self.position and self.position != 0:
            result['action'] = 'WOULD_TRADE'
            result['reason'] = f'Signal change: {self.position} -> {signal}'
            # Simulate the trade
            self.execute_paper_trade(signal, current_price)
        else:
            result['action'] = 'NO_TRADE'
            result['reason'] = 'No signal change or neutral position'
        
        return result
    
    def execute_paper_trade(self, signal, price):
        """Simulate a trade execution"""
        trade = {
            'timestamp': datetime.now(),
            'type': 'BUY' if signal == 1 else 'SELL',
            'price': price,
            'position_before': self.position,
            'position_after': signal
        }
        
        # Calculate theoretical P&L
        if self.position == 1 and signal == -1:
            # Selling BTC
            trade['btc_amount'] = self.btc_balance
            trade['usd_received'] = self.btc_balance * price * 0.9988  # 0.12% fee
            trade['pnl'] = trade['usd_received'] - self.usd_balance
            self.usd_balance = trade['usd_received']
            self.btc_balance = 0
        elif self.position == -1 and signal == 1:
            # Buying BTC
            trade['usd_spent'] = self.usd_balance
            trade['btc_amount'] = (self.usd_balance * 0.9988) / price
            self.btc_balance = trade['btc_amount']
            self.usd_balance = 0
        
        self.position = signal
        self.trades.append(trade)
        
        return trade
    
    def print_summary(self):
        """Print test summary"""
        print("\n📊 PAPER TRADING TEST SUMMARY")
        print("=" * 50)
        print(f"Signals evaluated: {self.signals_evaluated}")
        print(f"Proximity blocks: {self.proximity_blocks}")
        print(f"Trades executed: {len(self.trades)}")
        print(f"Block rate: {self.proximity_blocks/self.signals_evaluated*100:.1f}%")
        
        if self.trades:
            print(f"\n💰 Trading Activity:")
            for t in self.trades[-5:]:  # Last 5 trades
                print(f"  {t['timestamp'].strftime('%H:%M')} - {t['type']} @ ${t['price']:,.0f}")
        
        print(f"\n💡 Estimated savings from blocks: ${self.proximity_blocks * 400:,.0f}")

def run_5min_test():
    """Run 5-minute paper trading test"""
    print("🔬 Starting 5-Minute Paper Trading Test")
    print("=" * 50)
    
    # Load config
    with open('/tmp/stage1_5min_test.json', 'r') as f:
        config = json.load(f)
    
    print(f"Config: MA {config['Short_Window']}/{config['Long_Window']}")
    print(f"Proximity threshold: {config['proximity_threshold']}%")
    print(f"Mode: PAPER TRADING ONLY\n")
    
    # Initialize test
    test = PaperTradingTest(config)
    
    # Simulate price history (you'd get this from real data)
    prices = []
    base_price = 112000
    
    print("Running simulation...")
    
    # Run for specified duration
    for i in range(288):  # 288 5-min candles = 24 hours
        # Simulate price movement
        volatility = 0.001  # 0.1% volatility
        change = np.random.normal(0, volatility)
        new_price = base_price * (1 + change)
        prices.append(new_price)
        
        # Only evaluate after we have enough data
        if len(prices) >= config['Long_Window']:
            result = test.evaluate_signal(new_price, prices)
            
            if result and (i % 12 == 0 or result['action'] != 'NO_TRADE'):  # Log every hour or on action
                print(f"\n[{result['timestamp'].strftime('%H:%M')}] "
                      f"Price: ${result['price']:,.0f} | "
                      f"MA{config['Short_Window']}: ${result['ma_short']:,.0f} | "
                      f"MA{config['Long_Window']}: ${result['ma_long']:,.0f}")
                print(f"  Proximity: {result['proximity']:.2f}% | "
                      f"Signal: {result['signal']} | "
                      f"Action: {result['action']}")
                print(f"  Reason: {result['reason']}")
        
        # Update base price for next iteration
        base_price = new_price
        
        # Small delay to simulate real-time
        time.sleep(0.1)
    
    # Print summary
    test.print_summary()

if __name__ == "__main__":
    # First create the test config
    import subprocess
    subprocess.run(['python3', '/Users/chris/projects/python/btc/claude-bin/staged_testing_plan.py'])
    
    # Then run the test
    run_5min_test()