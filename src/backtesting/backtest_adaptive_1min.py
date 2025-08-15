#!/usr/bin/env python3
"""
Backtest adaptive strategy with 1-minute bars
Matches the paper trading configuration on ck gg tst
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tdr_core.strategies import AdaptiveMultiStrategy
from src.backtesting.data.data_loader import DataLoader

class AdaptiveBacktest1Min:
    """Backtest adaptive strategy using 1-minute bars"""
    
    def __init__(self, config_file=None):
        """Initialize backtester with configuration"""
        self.config = self.load_config(config_file)
        self.strategy = None
        self.data = None
        self.trades = []
        self.position = 0  # -1 = SHORT, 0 = NEUTRAL, 1 = LONG
        self.position_size = 0
        self.entry_price = 0
        self.balance_usd = 10000  # Start with $10k for testing
        self.balance_btc = 0
        self.fee_percentage = 0.0025  # 0.25% fee
        
    def load_config(self, config_file):
        """Load configuration from file or use defaults"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Default config matching ck test server
        return {
            "strategy_type": "adaptive",
            "enable_adaptive_strategy": True,
            "Short_Window": 4,
            "Long_Window": 20,
            "regime_lookback": 50,
            "regime_switch_threshold": 0.7,
            "min_trade_gap_minutes": 30,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "bb_std_dev": 2.0,
            "volume_threshold": 1.5,
            "macd_threshold": 0.001,
            "signal_confirmation_bars": 2,
            "pivot_buffer": 100,
            "pivot_lookback_hours": 2,
            "enable_pivot_protection": True,
            "enable_trailing_pivots": False,
            "proximity_threshold": 0.003,
            "do_live_trades": False,
            "candle_interval": "1min",
            "auto_resume": True,
            "max_trades_per_day": 100
        }
    
    def load_data(self, data_file, start_date=None, end_date=None):
        """Load and prepare 1-minute bar data"""
        print(f"Loading data from {data_file}...")
        
        # Use DataLoader to load tick data
        loader = DataLoader(data_file)
        df = loader.load_data()
        
        if df.empty:
            raise ValueError("No data loaded")
        
        # Filter by date range if specified
        if start_date:
            df = df[df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['timestamp'] <= pd.to_datetime(end_date)]
        
        # Resample to 1-minute bars
        print("Resampling to 1-minute bars...")
        df.set_index('timestamp', inplace=True)
        
        # Create OHLCV data
        bars = pd.DataFrame()
        bars['open'] = df['price'].resample('1min').first()
        bars['high'] = df['price'].resample('1min').max()
        bars['low'] = df['price'].resample('1min').min()
        bars['close'] = df['price'].resample('1min').last()
        bars['volume'] = df['amount'].resample('1min').sum()
        
        # Drop NaN values
        bars.dropna(inplace=True)
        
        # Reset index to have timestamp as column
        bars.reset_index(inplace=True)
        
        print(f"Loaded {len(bars)} 1-minute bars from {bars['timestamp'].min()} to {bars['timestamp'].max()}")
        
        self.data = bars
        return bars
    
    def initialize_strategy(self):
        """Initialize the adaptive strategy"""
        print("Initializing adaptive strategy...")
        
        # Create strategy instance
        self.strategy = AdaptiveMultiStrategy(
            config=self.config,
            logger=self.get_logger()
        )
        
        # Set initial position
        self.strategy.position = self.position
        self.strategy.position_size = self.position_size
        self.strategy.balance_usd = self.balance_usd
        self.strategy.balance_btc = self.balance_btc
        
    def get_logger(self):
        """Create a simple logger for the strategy"""
        import logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        return logging.getLogger(__name__)
    
    def run_backtest(self):
        """Run the backtest"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first")
        
        if self.strategy is None:
            self.initialize_strategy()
        
        print(f"\nRunning backtest from {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        print(f"Starting balance: ${self.balance_usd:,.2f}")
        print("-" * 80)
        
        last_trade_time = None
        
        # Process each 1-minute bar
        for i in range(len(self.data)):
            bar = self.data.iloc[i]
            timestamp = bar['timestamp']
            price = bar['close']
            
            # Skip if too soon after last trade
            if last_trade_time:
                minutes_since_trade = (timestamp - last_trade_time).total_seconds() / 60
                if minutes_since_trade < self.config['min_trade_gap_minutes']:
                    continue
            
            # Get signal from strategy
            signal = self.get_signal(i)
            
            # Execute trade if signal
            if signal != 0:
                self.execute_trade(signal, price, timestamp)
                last_trade_time = timestamp
        
        # Close any open position at end
        if self.position != 0:
            final_price = self.data.iloc[-1]['close']
            final_time = self.data.iloc[-1]['timestamp']
            self.execute_trade(-self.position, final_price, final_time)
        
        # Print results
        self.print_results()
    
    def get_signal(self, bar_index):
        """Get trading signal from adaptive strategy"""
        if bar_index < self.config['regime_lookback']:
            return 0  # Not enough data
        
        # Get recent data for strategy
        lookback = min(bar_index, 200)  # Use up to 200 bars
        recent_data = self.data.iloc[bar_index - lookback:bar_index + 1].copy()
        
        # Detect market regime
        regime = self.detect_regime(recent_data)
        
        # Get signal based on regime
        if regime == 'trending':
            return self.get_trend_signal(recent_data)
        elif regime == 'ranging':
            return self.get_range_signal(recent_data)
        elif regime == 'volatile':
            return self.get_volatile_signal(recent_data)
        
        return 0
    
    def detect_regime(self, data):
        """Detect market regime (trending/ranging/volatile)"""
        # Calculate indicators
        returns = data['close'].pct_change()
        volatility = returns.std()
        
        # Calculate trend strength using MA alignment
        ma_short = data['close'].rolling(self.config['Short_Window']).mean()
        ma_long = data['close'].rolling(self.config['Long_Window']).mean()
        
        trend_strength = abs((ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1])
        
        # Classify regime
        if volatility > 0.02:  # High volatility threshold
            return 'volatile'
        elif trend_strength > 0.005:  # Strong trend threshold
            return 'trending'
        else:
            return 'ranging'
    
    def get_trend_signal(self, data):
        """Get signal for trending market"""
        # Use MA crossover for trending markets
        ma_short = data['close'].rolling(self.config['Short_Window']).mean()
        ma_long = data['close'].rolling(self.config['Long_Window']).mean()
        
        # Check proximity threshold
        ma_diff_pct = abs((ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1])
        if ma_diff_pct < self.config['proximity_threshold']:
            return 0  # MAs too close
        
        # Generate signal
        if self.position <= 0 and ma_short.iloc[-1] > ma_long.iloc[-1]:
            return 1  # Buy signal
        elif self.position >= 0 and ma_short.iloc[-1] < ma_long.iloc[-1]:
            return -1  # Sell signal
        
        return 0
    
    def get_range_signal(self, data):
        """Get signal for ranging market"""
        # Use RSI for ranging markets
        rsi = self.calculate_rsi(data['close'])
        
        if self.position <= 0 and rsi < self.config['rsi_oversold']:
            return 1  # Buy at oversold
        elif self.position >= 0 and rsi > self.config['rsi_overbought']:
            return -1  # Sell at overbought
        
        return 0
    
    def get_volatile_signal(self, data):
        """Get signal for volatile market"""
        # Use Bollinger Bands for volatile markets
        bb_middle = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        bb_upper = bb_middle + (bb_std * self.config['bb_std_dev'])
        bb_lower = bb_middle - (bb_std * self.config['bb_std_dev'])
        
        current_price = data['close'].iloc[-1]
        
        if self.position <= 0 and current_price < bb_lower.iloc[-1]:
            return 1  # Buy at lower band
        elif self.position >= 0 and current_price > bb_upper.iloc[-1]:
            return -1  # Sell at upper band
        
        return 0
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def execute_trade(self, signal, price, timestamp):
        """Execute a trade"""
        trade = {
            'timestamp': timestamp,
            'price': price,
            'signal': signal,
            'type': 'buy' if signal > 0 else 'sell'
        }
        
        if signal > 0:  # Buy
            # Calculate BTC amount to buy
            usd_to_spend = self.balance_usd * 0.95  # Use 95% of USD
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
            # Sell all BTC
            btc_to_sell = self.balance_btc
            usd_received = (btc_to_sell * price) * (1 - self.fee_percentage)
            
            self.balance_btc = 0
            self.balance_usd += usd_received
            
            # Calculate P&L
            if self.entry_price > 0:
                pnl = usd_received - (self.position_size * self.entry_price)
                pnl_pct = (pnl / (self.position_size * self.entry_price)) * 100
                trade['pnl'] = pnl
                trade['pnl_pct'] = pnl_pct
                print(f"{timestamp}: SELL {btc_to_sell:.8f} BTC @ ${price:,.2f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            else:
                print(f"{timestamp}: SELL {btc_to_sell:.8f} BTC @ ${price:,.2f}")
            
            self.position = 0
            self.position_size = 0
            self.entry_price = 0
            
            trade['amount'] = btc_to_sell
            trade['usd_value'] = usd_received
        
        self.trades.append(trade)
    
    def print_results(self):
        """Print backtest results"""
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        
        # Calculate final value
        final_value = self.balance_usd
        if self.balance_btc > 0:
            final_price = self.data.iloc[-1]['close']
            final_value += self.balance_btc * final_price
        
        initial_value = 10000
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        print(f"Initial Balance: ${initial_value:,.2f}")
        print(f"Final Balance: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Trades: {len(self.trades)}")
        
        if self.trades:
            winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
            
            print(f"Winning Trades: {len(winning_trades)}")
            print(f"Losing Trades: {len(losing_trades)}")
            
            if winning_trades:
                avg_win = np.mean([t['pnl'] for t in winning_trades])
                print(f"Average Win: ${avg_win:.2f}")
            
            if losing_trades:
                avg_loss = np.mean([t['pnl'] for t in losing_trades])
                print(f"Average Loss: ${avg_loss:.2f}")
            
            # Calculate Sharpe ratio (simplified)
            if len(self.trades) > 1:
                returns = [t.get('pnl_pct', 0) for t in self.trades if 'pnl_pct' in t]
                if returns:
                    sharpe = np.mean(returns) / (np.std(returns) + 0.0001)
                    print(f"Sharpe Ratio: {sharpe:.2f}")
        
        print("=" * 80)
    
    def save_results(self, output_file):
        """Save results to JSON file"""
        results = {
            'config': self.config,
            'start_date': str(self.data['timestamp'].min()),
            'end_date': str(self.data['timestamp'].max()),
            'initial_balance': 10000,
            'final_balance': self.balance_usd + (self.balance_btc * self.data.iloc[-1]['close']),
            'total_trades': len(self.trades),
            'trades': [
                {
                    'timestamp': str(t['timestamp']),
                    'type': t['type'],
                    'price': t['price'],
                    'amount': t.get('amount', 0),
                    'pnl': t.get('pnl', 0),
                    'pnl_pct': t.get('pnl_pct', 0)
                }
                for t in self.trades
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest adaptive strategy with 1-minute bars')
    parser.add_argument('--data', default='btcusd.log', help='Data file to use')
    parser.add_argument('--config', help='Configuration file (JSON)')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=7, help='Number of days to backtest')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    # Create backtester
    backtester = AdaptiveBacktest1Min(args.config)
    
    # Calculate date range
    if args.end:
        end_date = pd.to_datetime(args.end)
    else:
        end_date = datetime.now()
    
    if args.start:
        start_date = pd.to_datetime(args.start)
    else:
        start_date = end_date - timedelta(days=args.days)
    
    # Load data
    backtester.load_data(args.data, start_date, end_date)
    
    # Run backtest
    backtester.run_backtest()
    
    # Save results if requested
    if args.output:
        backtester.save_results(args.output)


if __name__ == '__main__':
    main()