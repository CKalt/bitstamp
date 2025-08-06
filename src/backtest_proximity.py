#!/usr/bin/env python3
"""
Backtest the proximity threshold strategy to verify it matches live behavior
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loader import parse_log_file
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

class ProximityBacktester:
    def __init__(self, short_window=4, long_window=20, proximity_threshold=0.3):
        self.short_window = short_window
        self.long_window = long_window
        self.proximity_threshold = proximity_threshold
        self.position = 0  # -1 short, 0 neutral, 1 long
        self.trades = []
        self.blocked_trades = []
        self.initial_balance = 10000
        self.balance_usd = self.initial_balance
        self.balance_btc = 0
        self.entry_price = 0
        
    def calculate_moving_averages(self, prices, window):
        """Calculate simple moving average"""
        return prices.rolling(window=window).mean()
    
    def run_backtest(self, df, candle_interval='1h'):
        """Run backtest with proximity threshold"""
        
        # Resample data based on candle interval
        if candle_interval == '1min':
            df_resampled = df.resample('1T').last().dropna()
        elif candle_interval == '5min':
            df_resampled = df.resample('5T').last().dropna()
        elif candle_interval == '15min':
            df_resampled = df.resample('15T').last().dropna()
        else:  # Default to hourly
            df_resampled = df.resample('1H').last().dropna()
        
        # Calculate MAs
        df_resampled['MA_short'] = self.calculate_moving_averages(df_resampled['price'], self.short_window)
        df_resampled['MA_long'] = self.calculate_moving_averages(df_resampled['price'], self.long_window)
        
        # Drop NaN values
        df_resampled = df_resampled.dropna()
        
        print(f"\n📊 BACKTESTING WITH PROXIMITY THRESHOLD")
        print(f"========================================")
        print(f"Candle Interval: {candle_interval}")
        print(f"Proximity Threshold: {self.proximity_threshold}%")
        print(f"MA Windows: {self.short_window}/{self.long_window}")
        
        if len(df_resampled) == 0:
            print("ERROR: No data after resampling. Check input data.")
            return [], []
            
        print(f"Data Period: {df_resampled.index[0]} to {df_resampled.index[-1]}")
        print(f"Total Candles: {len(df_resampled)}\n")
        
        signals_evaluated = 0
        
        for idx, row in df_resampled.iterrows():
            price = row['price']
            ma_short = row['MA_short']
            ma_long = row['MA_long']
            
            # Calculate signal
            signal = 1 if ma_short > ma_long else -1
            
            # Calculate proximity
            ma_proximity = abs((ma_short - ma_long) / ma_long * 100) if ma_long != 0 else 0
            
            # Check if we should trade
            should_trade = False
            reason = ""
            
            signals_evaluated += 1
            
            # PROXIMITY CHECK - This is the critical part
            if ma_proximity <= self.proximity_threshold:
                # Block trade due to proximity
                reason = f"MAs too close: {ma_proximity:.2f}% <= {self.proximity_threshold}%"
                self.blocked_trades.append({
                    'time': idx,
                    'signal': signal,
                    'position': self.position,
                    'proximity': ma_proximity,
                    'reason': reason,
                    'ma_short': ma_short,
                    'ma_long': ma_long,
                    'price': price
                })
            elif signal == 1 and self.position <= 0:
                # BUY signal and we're short or neutral
                should_trade = True
                reason = "BUY: MA crossover up"
            elif signal == -1 and self.position >= 0:
                # SELL signal and we're long or neutral
                should_trade = True
                reason = "SELL: MA crossover down"
            
            # Execute trade if needed
            if should_trade:
                self.execute_trade(idx, price, signal, ma_proximity, reason)
        
        # Final position close
        if self.position != 0:
            final_price = df_resampled.iloc[-1]['price']
            self.close_position(df_resampled.index[-1], final_price)
        
        # Calculate statistics
        self.print_results(signals_evaluated)
        
        return self.trades, self.blocked_trades
    
    def execute_trade(self, timestamp, price, signal, proximity, reason):
        """Execute a trade"""
        trade = {
            'time': timestamp,
            'price': price,
            'signal': signal,
            'proximity': proximity,
            'reason': reason,
            'position_before': self.position
        }
        
        if signal == 1:  # BUY
            if self.position == -1:  # Close short
                pnl = (self.entry_price - price) * self.balance_btc
                self.balance_usd += self.balance_btc * price + pnl
                self.balance_btc = 0
            
            # Open long
            self.balance_btc = self.balance_usd / price * 0.998  # 0.2% fee
            self.balance_usd = 0
            self.entry_price = price
            self.position = 1
            trade['action'] = 'BUY'
            
        elif signal == -1:  # SELL
            if self.position == 1:  # Close long
                self.balance_usd = self.balance_btc * price * 0.998  # 0.2% fee
                self.balance_btc = 0
            else:  # Open short
                self.balance_btc = -self.balance_usd / price
                self.entry_price = price
            
            self.position = -1
            trade['action'] = 'SELL'
        
        trade['position_after'] = self.position
        self.trades.append(trade)
    
    def close_position(self, timestamp, price):
        """Close final position"""
        if self.position == 1:
            self.balance_usd = self.balance_btc * price * 0.998
        elif self.position == -1:
            pnl = (self.entry_price - price) * abs(self.balance_btc)
            self.balance_usd = self.balance_usd + pnl
        
        self.position = 0
        self.balance_btc = 0
    
    def print_results(self, signals_evaluated):
        """Print backtest results"""
        final_balance = self.balance_usd
        total_return = (final_balance - self.initial_balance) / self.initial_balance * 100
        
        print(f"\n📈 BACKTEST RESULTS")
        print(f"==================")
        print(f"Signals Evaluated: {signals_evaluated}")
        print(f"Trades Executed: {len(self.trades)}")
        print(f"Trades Blocked: {len(self.blocked_trades)}")
        print(f"Block Rate: {len(self.blocked_trades) / signals_evaluated * 100:.1f}%")
        print(f"")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${final_balance:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"")
        
        if self.trades:
            print(f"🔄 TRADE SEQUENCE:")
            for i, trade in enumerate(self.trades[:10], 1):  # Show first 10
                print(f"  {i}. {trade['time'].strftime('%Y-%m-%d %H:%M')} - "
                      f"{trade['action']} @ ${trade['price']:,.0f} "
                      f"(Proximity: {trade['proximity']:.2f}%)")
        
        if self.blocked_trades:
            print(f"\n🚫 SAMPLE BLOCKED TRADES:")
            for block in self.blocked_trades[:5]:  # Show first 5
                print(f"  {block['time'].strftime('%Y-%m-%d %H:%M')} - "
                      f"Proximity: {block['proximity']:.2f}% - {block['reason']}")

def main():
    """Main function to run proximity backtest"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest proximity threshold strategy')
    parser.add_argument('--days', type=int, default=30, help='Days to backtest')
    parser.add_argument('--threshold', type=float, default=0.3, help='Proximity threshold %')
    parser.add_argument('--interval', type=str, default='1h', 
                       choices=['1min', '5min', '15min', '1h'],
                       help='Candle interval')
    parser.add_argument('--log-file', type=str, default='btcusd.log',
                       help='Log file to parse')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.log_file}...")
    df = parse_log_file(args.log_file)
    
    # Convert timestamp to datetime index (timestamps are in seconds)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('timestamp')
    
    # Filter to requested days
    end_date = df.index[-1]
    start_date = end_date - timedelta(days=args.days)
    df_filtered = df[df.index >= start_date]
    
    print(f"Data loaded: {len(df_filtered)} records")
    
    # Run backtest
    backtester = ProximityBacktester(
        short_window=4,
        long_window=20,
        proximity_threshold=args.threshold
    )
    
    trades, blocked = backtester.run_backtest(df_filtered, args.interval)
    
    # Save results for comparison with live
    results = {
        'trades': [{'time': t['time'].isoformat(), 
                   'price': t['price'],
                   'action': t['action'],
                   'proximity': t['proximity']} for t in trades],
        'blocked_count': len(blocked),
        'settings': {
            'threshold': args.threshold,
            'interval': args.interval,
            'ma_windows': '4/20'
        }
    }
    
    with open('backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to backtest_results.json")
    print(f"   Compare with live trading logs to verify matching behavior")

if __name__ == '__main__':
    main()