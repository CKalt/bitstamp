#!/usr/bin/env python3
"""
Simple adaptive strategy backtest with 1-minute bars
Standalone implementation without external dependencies
"""

import sys
import os
import json
import csv
from datetime import datetime, timedelta
from collections import defaultdict

class SimpleAdaptiveBacktest:
    """Simple adaptive backtest for 1-minute bars"""
    
    def __init__(self, config=None):
        """Initialize backtester"""
        self.config = config or self.get_default_config()
        self.trades = []
        self.position = 0  # -1 = SHORT, 0 = NEUTRAL, 1 = LONG
        self.position_size = 0
        self.entry_price = 0
        self.balance_usd = 10000
        self.balance_btc = 0
        self.fee_percentage = 0.0025
        self.bars = []
        
    def get_default_config(self):
        """Default configuration matching ck test server"""
        return {
            "Short_Window": 4,
            "Long_Window": 20,
            "proximity_threshold": 0.003,
            "min_trade_gap_minutes": 30,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "regime_lookback": 50
        }
    
    def load_data(self, filename, start_date=None, end_date=None):
        """Load and process tick data into 1-minute bars"""
        print(f"Loading data from {filename}...")
        
        # Parse dates if provided
        if start_date:
            start_ts = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date
        else:
            start_ts = datetime.now() - timedelta(days=7)
            
        if end_date:
            end_ts = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date
        else:
            end_ts = datetime.now()
        
        # Read tick data and aggregate to 1-minute bars
        minute_data = defaultdict(lambda: {'prices': [], 'volumes': []})
        
        line_count = 0
        with open(filename, 'r') as f:
            for line in f:
                line_count += 1
                if line_count % 100000 == 0:
                    print(f"  Processed {line_count:,} lines...")
                
                try:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        timestamp_str = parts[0]
                        price = float(parts[1])
                        amount = float(parts[2])
                        
                        # Parse timestamp
                        ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                        
                        # Filter by date range
                        if ts < start_ts or ts > end_ts:
                            continue
                        
                        # Round to minute
                        minute = ts.replace(second=0, microsecond=0)
                        minute_data[minute]['prices'].append(price)
                        minute_data[minute]['volumes'].append(amount)
                        
                except (ValueError, IndexError):
                    continue
        
        # Convert to bars
        print(f"Creating 1-minute bars...")
        self.bars = []
        for minute in sorted(minute_data.keys()):
            if minute_data[minute]['prices']:
                prices = minute_data[minute]['prices']
                self.bars.append({
                    'timestamp': minute,
                    'open': prices[0],
                    'high': max(prices),
                    'low': min(prices),
                    'close': prices[-1],
                    'volume': sum(minute_data[minute]['volumes'])
                })
        
        print(f"Loaded {len(self.bars)} 1-minute bars")
        if self.bars:
            print(f"Date range: {self.bars[0]['timestamp']} to {self.bars[-1]['timestamp']}")
        
        return len(self.bars)
    
    def calculate_ma(self, prices, period):
        """Calculate simple moving average"""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
        gains = []
        losses = []
        
        for i in range(1, period + 1):
            change = prices[-i] - prices[-i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def detect_regime(self, prices, volumes):
        """Detect market regime"""
        if len(prices) < self.config['regime_lookback']:
            return 'neutral'
        
        # Calculate volatility
        returns = []
        for i in range(1, min(20, len(prices))):
            returns.append((prices[-i] - prices[-i-1]) / prices[-i-1])
        
        if not returns:
            return 'neutral'
        
        volatility = sum(abs(r) for r in returns) / len(returns)
        
        # Calculate trend strength
        ma_short = self.calculate_ma(prices, self.config['Short_Window'])
        ma_long = self.calculate_ma(prices, self.config['Long_Window'])
        
        if ma_short and ma_long:
            trend_strength = abs((ma_short - ma_long) / ma_long)
        else:
            trend_strength = 0
        
        # Classify regime
        if volatility > 0.02:
            return 'volatile'
        elif trend_strength > 0.005:
            return 'trending'
        else:
            return 'ranging'
    
    def get_signal(self, bar_index):
        """Get trading signal based on current market conditions"""
        if bar_index < self.config['regime_lookback']:
            return 0
        
        # Get recent prices
        prices = [self.bars[i]['close'] for i in range(max(0, bar_index - 200), bar_index + 1)]
        volumes = [self.bars[i]['volume'] for i in range(max(0, bar_index - 200), bar_index + 1)]
        
        # Detect regime
        regime = self.detect_regime(prices, volumes)
        
        # Get signal based on regime
        if regime == 'trending':
            return self.get_trend_signal(prices)
        elif regime == 'ranging':
            return self.get_range_signal(prices)
        elif regime == 'volatile':
            return self.get_volatile_signal(prices)
        
        return 0
    
    def get_trend_signal(self, prices):
        """Signal for trending markets (MA crossover)"""
        ma_short = self.calculate_ma(prices, self.config['Short_Window'])
        ma_long = self.calculate_ma(prices, self.config['Long_Window'])
        
        if not ma_short or not ma_long:
            return 0
        
        # Check proximity threshold
        ma_diff_pct = abs((ma_short - ma_long) / ma_long)
        if ma_diff_pct < self.config['proximity_threshold']:
            return 0
        
        # Generate signal
        if self.position <= 0 and ma_short > ma_long:
            return 1  # Buy
        elif self.position >= 0 and ma_short < ma_long:
            return -1  # Sell
        
        return 0
    
    def get_range_signal(self, prices):
        """Signal for ranging markets (RSI)"""
        rsi = self.calculate_rsi(prices, self.config['rsi_period'])
        
        if self.position <= 0 and rsi < self.config['rsi_oversold']:
            return 1  # Buy
        elif self.position >= 0 and rsi > self.config['rsi_overbought']:
            return -1  # Sell
        
        return 0
    
    def get_volatile_signal(self, prices):
        """Signal for volatile markets (Bollinger Bands)"""
        period = self.config['bb_period']
        if len(prices) < period:
            return 0
        
        # Calculate Bollinger Bands
        ma = self.calculate_ma(prices, period)
        if not ma:
            return 0
        
        # Calculate standard deviation
        squared_diffs = [(p - ma) ** 2 for p in prices[-period:]]
        std_dev = (sum(squared_diffs) / period) ** 0.5
        
        upper_band = ma + (std_dev * self.config['bb_std_dev'])
        lower_band = ma - (std_dev * self.config['bb_std_dev'])
        
        current_price = prices[-1]
        
        if self.position <= 0 and current_price < lower_band:
            return 1  # Buy
        elif self.position >= 0 and current_price > upper_band:
            return -1  # Sell
        
        return 0
    
    def execute_trade(self, signal, price, timestamp):
        """Execute a trade"""
        trade = {
            'timestamp': str(timestamp),
            'price': price,
            'signal': signal,
            'type': 'buy' if signal > 0 else 'sell'
        }
        
        if signal > 0:  # Buy
            usd_to_spend = self.balance_usd * 0.95
            btc_amount = (usd_to_spend / price) * (1 - self.fee_percentage)
            
            self.balance_usd -= usd_to_spend
            self.balance_btc += btc_amount
            self.position = 1
            self.position_size = btc_amount
            self.entry_price = price
            
            trade['amount'] = btc_amount
            trade['usd_value'] = usd_to_spend
            
            print(f"{timestamp}: BUY {btc_amount:.8f} BTC @ ${price:,.2f}")
            
        else:  # Sell
            btc_to_sell = self.balance_btc
            usd_received = (btc_to_sell * price) * (1 - self.fee_percentage)
            
            self.balance_btc = 0
            self.balance_usd += usd_received
            
            if self.entry_price > 0:
                pnl = usd_received - (self.position_size * self.entry_price)
                pnl_pct = (pnl / (self.position_size * self.entry_price)) * 100
                trade['pnl'] = pnl
                trade['pnl_pct'] = pnl_pct
                print(f"{timestamp}: SELL {btc_to_sell:.8f} BTC @ ${price:,.2f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            
            self.position = 0
            self.position_size = 0
            self.entry_price = 0
            
            trade['amount'] = btc_to_sell
            trade['usd_value'] = usd_received
        
        self.trades.append(trade)
    
    def run(self):
        """Run the backtest"""
        if not self.bars:
            print("No data loaded!")
            return
        
        print(f"\nRunning backtest...")
        print(f"Starting balance: ${self.balance_usd:,.2f}")
        print("-" * 80)
        
        last_trade_time = None
        
        for i in range(len(self.bars)):
            bar = self.bars[i]
            timestamp = bar['timestamp']
            price = bar['close']
            
            # Skip if too soon after last trade
            if last_trade_time:
                minutes_since = (timestamp - last_trade_time).total_seconds() / 60
                if minutes_since < self.config['min_trade_gap_minutes']:
                    continue
            
            # Get signal
            signal = self.get_signal(i)
            
            # Execute trade if signal
            if signal != 0:
                self.execute_trade(signal, price, timestamp)
                last_trade_time = timestamp
        
        # Close position at end if needed
        if self.position != 0 and self.bars:
            final_bar = self.bars[-1]
            self.execute_trade(-self.position, final_bar['close'], final_bar['timestamp'])
        
        self.print_results()
    
    def print_results(self):
        """Print backtest results"""
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        
        # Calculate final value
        final_value = self.balance_usd
        if self.balance_btc > 0 and self.bars:
            final_value += self.balance_btc * self.bars[-1]['close']
        
        initial_value = 10000
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        print(f"Initial Balance: ${initial_value:,.2f}")
        print(f"Final Balance: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Trades: {len(self.trades)}")
        
        if self.trades:
            winning = [t for t in self.trades if t.get('pnl', 0) > 0]
            losing = [t for t in self.trades if t.get('pnl', 0) < 0]
            
            print(f"Winning Trades: {len(winning)}")
            print(f"Losing Trades: {len(losing)}")
            
            if winning:
                avg_win = sum(t['pnl'] for t in winning) / len(winning)
                print(f"Average Win: ${avg_win:.2f}")
            
            if losing:
                avg_loss = sum(t['pnl'] for t in losing) / len(losing)
                print(f"Average Loss: ${avg_loss:.2f}")
        
        print("=" * 80)
    
    def save_results(self, filename):
        """Save results to JSON file"""
        final_value = self.balance_usd
        if self.balance_btc > 0 and self.bars:
            final_value += self.balance_btc * self.bars[-1]['close']
        
        results = {
            'config': self.config,
            'start_date': str(self.bars[0]['timestamp']) if self.bars else None,
            'end_date': str(self.bars[-1]['timestamp']) if self.bars else None,
            'initial_balance': 10000,
            'final_balance': final_value,
            'total_trades': len(self.trades),
            'trades': self.trades
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='btcusd.log')
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--start', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', help='End date YYYY-MM-DD')
    parser.add_argument('--output', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Create backtester
    bt = SimpleAdaptiveBacktest()
    
    # Calculate dates
    if args.end:
        end_date = args.end
    else:
        end_date = datetime.now()
    
    if args.start:
        start_date = args.start
    else:
        if isinstance(end_date, str):
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end_dt = end_date
        start_date = end_dt - timedelta(days=args.days)
    
    # Load data
    bt.load_data(args.data, start_date, end_date)
    
    # Run backtest
    bt.run()
    
    # Save results
    if args.output:
        bt.save_results(args.output)


if __name__ == '__main__':
    main()