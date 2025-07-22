#!/usr/bin/env python3
"""
Realistic backtesting with actual market frictions
Based on live trading observations
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import math

class RealisticBacktest:
    def __init__(self, data, initial_balance_usd=10000):
        self.data = data
        self.initial_balance = initial_balance_usd
        self.balance_usd = initial_balance_usd
        self.balance_btc = 0
        self.position = 0  # 1 = LONG, -1 = SHORT, 0 = FLAT
        self.trades = []
        self.last_trade_time = None
        
        # Realistic parameters based on live trading
        self.base_slippage = 0.0005  # 0.05% base slippage
        self.multi_part_slippage = 0.0002  # Additional 0.02% per part
        self.min_flip_hours = 2  # Minimum hours between position flips
        self.max_btc_per_order = 0.9  # Bitstamp limit
        
        # Fee structure (adjust based on your tier)
        self.fee_rate = 0.0025  # 0.25% for < $20k volume
        
    def calculate_multi_part_orders(self, btc_amount):
        """Calculate how many parts needed for order"""
        return math.ceil(btc_amount / self.max_btc_per_order)
    
    def calculate_realistic_fill_price(self, signal_price, trade_type, btc_amount):
        """Calculate realistic fill price with slippage"""
        num_parts = self.calculate_multi_part_orders(btc_amount)
        
        # Base slippage + additional for multi-part
        total_slippage = self.base_slippage + (self.multi_part_slippage * (num_parts - 1))
        
        if trade_type == 'BUY':
            # Buying pushes price up
            return signal_price * (1 + total_slippage)
        else:  # SELL
            # Selling pushes price down
            return signal_price * (1 - total_slippage)
    
    def calculate_fees(self, trade_value, num_parts):
        """Calculate total fees for multi-part order"""
        # Each part incurs a fee
        return trade_value * self.fee_rate * num_parts
    
    def check_whipsaw_filter(self, current_time):
        """Prevent rapid position flips"""
        if self.last_trade_time is None:
            return True
        
        time_since_last = (current_time - self.last_trade_time).total_seconds() / 3600
        return time_since_last >= self.min_flip_hours
    
    def execute_trade(self, signal, timestamp, signal_price, reason=""):
        """Execute trade with realistic modeling"""
        # Check whipsaw filter
        if not self.check_whipsaw_filter(timestamp):
            return None
        
        # Skip if already in desired position
        if signal == self.position:
            return None
        
        trade = {
            'timestamp': timestamp,
            'signal_price': signal_price,
            'reason': reason
        }
        
        if signal == 1 and self.position <= 0:  # Buy signal
            # Calculate BTC to buy
            btc_amount = self.balance_usd / signal_price
            
            # Realistic fill price
            fill_price = self.calculate_realistic_fill_price(signal_price, 'BUY', btc_amount)
            actual_btc = self.balance_usd / fill_price
            
            # Calculate fees
            num_parts = self.calculate_multi_part_orders(actual_btc)
            fees = self.calculate_fees(self.balance_usd, num_parts)
            
            # Update balances
            self.balance_btc = actual_btc
            self.balance_usd = 0
            
            trade.update({
                'type': 'BUY',
                'amount': actual_btc,
                'fill_price': fill_price,
                'slippage': fill_price - signal_price,
                'fees': fees,
                'num_parts': num_parts
            })
            
            self.position = 1
            
        elif signal == -1 and self.position >= 0:  # Sell signal
            btc_to_sell = self.balance_btc
            
            # Realistic fill price
            fill_price = self.calculate_realistic_fill_price(signal_price, 'SELL', btc_to_sell)
            usd_received = btc_to_sell * fill_price
            
            # Fees on single SELL (no multi-part for sells)
            fees = self.calculate_fees(usd_received, 1)
            
            # Update balances
            self.balance_usd = usd_received - fees
            self.balance_btc = 0
            
            trade.update({
                'type': 'SELL',
                'amount': btc_to_sell,
                'fill_price': fill_price,
                'slippage': signal_price - fill_price,
                'fees': fees,
                'num_parts': 1
            })
            
            self.position = -1
        
        if 'type' in trade:
            self.trades.append(trade)
            self.last_trade_time = timestamp
            return trade
        
        return None
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {}
        
        # Final balance
        final_value = self.balance_usd
        if self.balance_btc > 0:
            final_value = self.balance_btc * self.data.iloc[-1]['close']
        
        total_return = (final_value - self.initial_balance) / self.initial_balance * 100
        
        # Fee analysis
        total_fees = sum(t['fees'] for t in self.trades)
        avg_fee_per_trade = total_fees / len(self.trades)
        
        # Slippage analysis
        total_slippage_cost = sum(
            abs(t['slippage']) * t['amount'] for t in self.trades
        )
        
        # Multi-part analysis
        multi_part_trades = [t for t in self.trades if t['num_parts'] > 1]
        
        return {
            'total_trades': len(self.trades),
            'final_balance': final_value,
            'total_return_%': total_return,
            'total_fees': total_fees,
            'avg_fee_per_trade': avg_fee_per_trade,
            'total_slippage_cost': total_slippage_cost,
            'multi_part_trades': len(multi_part_trades),
            'avg_parts_per_buy': np.mean([t['num_parts'] for t in self.trades if t['type'] == 'BUY'])
        }

def run_realistic_backtest(strategy, data, **kwargs):
    """Run backtest with strategy"""
    backtest = RealisticBacktest(data, **kwargs)
    
    for idx, row in data.iterrows():
        # Get signal from strategy
        signal = strategy.get_signal(idx)
        
        if signal != 0:
            reason = strategy.get_signal_reason(idx)
            backtest.execute_trade(
                signal, 
                row['timestamp'], 
                row['close'],
                reason
            )
    
    return backtest

# Example usage with MA strategy
class MAStrategy:
    def __init__(self, data, short_window=6, long_window=34):
        self.data = data
        self.short_ma = data['close'].rolling(window=short_window).mean()
        self.long_ma = data['close'].rolling(window=long_window).mean()
        
    def get_signal(self, idx):
        if idx < 34:  # Not enough data
            return 0
            
        if self.short_ma.iloc[idx] > self.long_ma.iloc[idx]:
            return 1  # LONG
        else:
            return -1  # SHORT
    
    def get_signal_reason(self, idx):
        return f"MA crossover: {self.short_ma.iloc[idx]:.2f} vs {self.long_ma.iloc[idx]:.2f}"

if __name__ == "__main__":
    # Load your data
    print("Realistic Backtest Framework")
    print("=" * 50)
    print("Features:")
    print("- Multi-part order simulation")
    print("- Realistic slippage modeling") 
    print("- Whipsaw filter (min 2 hours between flips)")
    print("- Accurate fee calculation")
    print("\nAdjust parameters based on your live trading observations!")