#!/usr/bin/env python3
"""
Enhanced Backtesting System with Pivot Protection
Properly simulates the complete trading system including:
- Pivot protection with sticky levels
- Profit-aware trailing pivots
- 100% position flips (always fully invested)
- Realistic Bitstamp fees
- Adaptive multi-strategy regime detection
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import argparse
import logging
from dataclasses import dataclass, field
from collections import defaultdict

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

# Import existing components
from data.loader import parse_log_file
from indicators.technical_indicators import (
    ensure_datetime_index, 
    add_moving_averages,
    calculate_rsi,
    calculate_bollinger_bands
)

@dataclass
class Position:
    """Track position details"""
    direction: int  # 1 for LONG, -1 for SHORT
    entry_price: float
    entry_time: datetime
    size: float  # BTC amount for LONG, USD amount for SHORT
    cost_basis: float  # Total cost including fees

@dataclass
class PivotTracker:
    """Track pivot protection levels"""
    support_level: float = 0
    resistance_level: float = 0
    levels_locked: bool = False
    profit_locked: float = 0
    protection_tier: str = ""
    last_update: datetime = None
    buffer_zone: float = 100
    recent_high: float = 0
    recent_low: float = 0

@dataclass
class Trade:
    """Record individual trades"""
    timestamp: datetime
    direction: str  # 'BUY' or 'SELL'
    price: float
    amount: float
    fee: float
    slippage: float
    reason: str
    regime: str
    position_before: Optional[Position] = None
    position_after: Optional[Position] = None
    profit: float = 0
    pivot_triggered: bool = False

class EnhancedBacktester:
    """
    Enhanced backtesting system that accurately simulates the live trading system
    including pivot protection, regime detection, and realistic execution.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logger()
        
        # Trading parameters
        self.initial_balance = config.get('initial_balance', 10000)
        self.fee_rate = config.get('fee_rate', 0.0012)  # 0.12% Bitstamp fee
        self.slippage_rate = config.get('slippage_rate', 0.0005)  # 0.05% slippage
        
        # Pivot protection parameters
        self.enable_pivot_protection = config.get('enable_pivot_protection', True)
        self.pivot_buffer = config.get('pivot_buffer', 100)
        self.pivot_lookback_hours = config.get('pivot_lookback_hours', 2)
        self.enable_trailing_pivots = config.get('enable_trailing_pivots', True)
        self.pivot_profit_tiers = config.get('pivot_profit_tiers', [
            {"threshold": 0.05, "protection_ratio": 0.70},
            {"threshold": 0.10, "protection_ratio": 0.80},
            {"threshold": 0.15, "protection_ratio": 0.85},
            {"threshold": 0.20, "protection_ratio": 0.90}
        ])
        self.pivot_respect_technical_levels = config.get('pivot_respect_technical_levels', True)
        
        # Adaptive strategy parameters
        self.regime_lookback = config.get('regime_lookback', 100)
        self.regime_switch_threshold = config.get('regime_switch_threshold', 0.80)
        self.min_strategy_switch_minutes = config.get('min_strategy_switch_minutes', 120)
        self.min_trade_gap_minutes = config.get('min_trade_gap_minutes', 15)
        self.signal_confirmation_bars = config.get('signal_confirmation_bars', 2)
        self.whipsaw_threshold = config.get('whipsaw_threshold', 8.0)
        
        # MA parameters
        self.short_window = config.get('short_window', 10)
        self.long_window = config.get('long_window', 20)
        
        # Initialize state
        self.position: Optional[Position] = None
        self.pivot_tracker = PivotTracker()
        self.trades: List[Trade] = []
        self.regime_history = []
        self.current_regime = 'trending'
        self.last_trade_time: Optional[datetime] = None
        self.last_regime_switch_time: Optional[datetime] = None
        self.signal_history = []
        
        # Performance tracking
        self.equity_curve = []
        self.regime_performance = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'profit': 0, 'time_in_regime': 0
        })
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for backtesting"""
        logger = logging.getLogger('EnhancedBacktester')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def calculate_fees_and_slippage(self, price: float, amount: float, is_buy: bool) -> Tuple[float, float]:
        """Calculate realistic fees and slippage"""
        # Fee is percentage of total value
        fee = price * amount * self.fee_rate
        
        # Slippage: buyers pay more, sellers receive less
        if is_buy:
            slippage = price * self.slippage_rate
        else:
            slippage = -price * self.slippage_rate
            
        return fee, slippage
        
    def detect_market_regime(self, df: pd.DataFrame) -> Tuple[str, float, Dict]:
        """
        Detect current market regime: trending, ranging, or volatile
        Returns: (regime, confidence, metrics)
        """
        if len(df) < self.regime_lookback:
            return 'trending', 0.5, {}
            
        recent_data = df.tail(self.regime_lookback)
        
        # Calculate metrics
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Trend strength using ADX concept
        high_low_range = recent_data['close'].rolling(14).max() - recent_data['close'].rolling(14).min()
        trend_strength = abs(recent_data['close'].iloc[-1] - recent_data['close'].iloc[-14]) / high_low_range.iloc[-1] if high_low_range.iloc[-1] > 0 else 0
        
        # Whipsaw detection
        ma_short = recent_data['close'].rolling(self.short_window).mean()
        ma_long = recent_data['close'].rolling(self.long_window).mean()
        crossovers = ((ma_short > ma_long) != (ma_short.shift(1) > ma_long.shift(1))).sum()
        whipsaw_ratio = crossovers / len(recent_data) * 100
        
        # Classify regime
        if volatility > 0.4:  # High volatility
            regime = 'volatile'
            confidence = min(volatility / 0.4, 1.0)
        elif trend_strength > 0.6:  # Strong trend
            regime = 'trending'
            confidence = trend_strength
        elif whipsaw_ratio > self.whipsaw_threshold:  # Many false signals
            regime = 'ranging'
            confidence = min(whipsaw_ratio / self.whipsaw_threshold, 1.0)
        else:
            regime = 'trending'  # Default
            confidence = 0.5
            
        metrics = {
            'volatility': volatility,
            'trend_strength': trend_strength,
            'whipsaw_ratio': whipsaw_ratio,
            'crossovers': crossovers
        }
        
        return regime, confidence, metrics
        
    def update_pivot_levels(self, df: pd.DataFrame, current_price: float, current_time: datetime):
        """Update pivot protection levels based on recent price action"""
        if not self.enable_pivot_protection:
            return
            
        # Get recent price data
        lookback_time = current_time - timedelta(hours=self.pivot_lookback_hours)
        recent_data = df[df.index >= lookback_time] if len(df) > 0 else df
        
        if len(recent_data) < 10:  # Need minimum data
            return
            
        recent_high = recent_data['close'].max()
        recent_low = recent_data['close'].min()
        
        # Update tracker
        self.pivot_tracker.recent_high = recent_high
        self.pivot_tracker.recent_low = recent_low
        
        # Set levels if not locked or no position
        if not self.pivot_tracker.levels_locked or self.position is None:
            if self.position and self.position.direction == 1:  # LONG
                # Support level protects longs
                self.pivot_tracker.support_level = recent_low - (self.pivot_buffer / 2)
                self.pivot_tracker.resistance_level = recent_high + (self.pivot_buffer / 2)
            elif self.position and self.position.direction == -1:  # SHORT
                # Resistance level protects shorts
                self.pivot_tracker.support_level = recent_low - (self.pivot_buffer / 2)
                self.pivot_tracker.resistance_level = recent_high + (self.pivot_buffer / 2)
                
            self.pivot_tracker.levels_locked = True
            self.pivot_tracker.last_update = current_time
            self.logger.info(f"📊 Pivot levels set: Support=${self.pivot_tracker.support_level:.0f}, Resistance=${self.pivot_tracker.resistance_level:.0f}")
            
    def update_trailing_pivot_protection(self, current_price: float):
        """Update pivot levels to protect profits (one-way ratchet)"""
        if not self.enable_trailing_pivots or not self.position:
            return
            
        if not self.pivot_tracker.levels_locked:
            return
            
        entry_price = self.position.entry_price
        position_value = self.position.size * current_price if self.position.direction == 1 else self.position.size
        
        if self.position.direction == 1:  # LONG position
            profit_per_unit = current_price - entry_price
            profit_percentage = profit_per_unit / entry_price
            
            # Find applicable protection tier
            protection_ratio = 0
            for tier in sorted(self.pivot_profit_tiers, key=lambda x: x['threshold'], reverse=True):
                if profit_percentage >= tier['threshold']:
                    protection_ratio = tier['protection_ratio']
                    break
                    
            if protection_ratio > 0:
                # Calculate new support level to protect profits
                profit_to_protect = profit_per_unit * protection_ratio
                new_support = entry_price + profit_to_protect
                
                # Only raise support, never lower it
                current_support = self.pivot_tracker.support_level
                if new_support > current_support:
                    old_support = self.pivot_tracker.support_level
                    self.pivot_tracker.support_level = new_support
                    self.pivot_tracker.profit_locked = new_support - entry_price
                    self.pivot_tracker.protection_tier = f"{int(protection_ratio * 100)}%"
                    
                    self.logger.info(f"🔥 TRAILING PIVOT: Support raised from ${old_support:.0f} to ${new_support:.0f}")
                    self.logger.info(f"   Profit locked: ${self.pivot_tracker.profit_locked:.0f} ({self.pivot_tracker.protection_tier} of ${profit_per_unit:.0f} gain)")
                    
        else:  # SHORT position
            profit_per_unit = entry_price - current_price
            profit_percentage = profit_per_unit / entry_price
            
            # Find applicable protection tier
            protection_ratio = 0
            for tier in sorted(self.pivot_profit_tiers, key=lambda x: x['threshold'], reverse=True):
                if profit_percentage >= tier['threshold']:
                    protection_ratio = tier['protection_ratio']
                    break
                    
            if protection_ratio > 0:
                # Calculate new resistance level to protect profits
                profit_to_protect = profit_per_unit * protection_ratio
                new_resistance = entry_price - profit_to_protect
                
                # Only lower resistance, never raise it
                current_resistance = self.pivot_tracker.resistance_level
                if new_resistance < current_resistance:
                    old_resistance = self.pivot_tracker.resistance_level
                    self.pivot_tracker.resistance_level = new_resistance
                    self.pivot_tracker.profit_locked = entry_price - new_resistance
                    self.pivot_tracker.protection_tier = f"{int(protection_ratio * 100)}%"
                    
                    self.logger.info(f"🔥 TRAILING PIVOT: Resistance lowered from ${old_resistance:.0f} to ${new_resistance:.0f}")
                    self.logger.info(f"   Profit locked: ${self.pivot_tracker.profit_locked:.0f} ({self.pivot_tracker.protection_tier} of ${profit_per_unit:.0f} gain)")
                    
    def check_pivot_break(self, current_price: float) -> Tuple[bool, str]:
        """Check if price has broken pivot levels"""
        if not self.enable_pivot_protection or not self.position:
            return False, ""
            
        if self.position.direction == 1:  # LONG position
            if current_price <= self.pivot_tracker.support_level:
                reason = f"Pivot break: below support ${self.pivot_tracker.support_level:.0f}"
                return True, reason
        else:  # SHORT position
            if current_price >= self.pivot_tracker.resistance_level:
                reason = f"Pivot break: above resistance ${self.pivot_tracker.resistance_level:.0f}"
                return True, reason
                
        return False, ""
        
    def generate_ma_signal(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Generate moving average crossover signal"""
        if len(df) < self.long_window:
            return 0, ""
            
        ma_short = df['close'].rolling(self.short_window).mean()
        ma_long = df['close'].rolling(self.long_window).mean()
        
        # Check for crossover
        if ma_short.iloc[-1] > ma_long.iloc[-1] and ma_short.iloc[-2] <= ma_long.iloc[-2]:
            return 1, f"MA crossover: {self.short_window} > {self.long_window}"
        elif ma_short.iloc[-1] < ma_long.iloc[-1] and ma_short.iloc[-2] >= ma_long.iloc[-2]:
            return -1, f"MA crossover: {self.short_window} < {self.long_window}"
            
        return 0, ""
        
    def generate_rsi_signal(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Generate RSI-based signal for ranging markets"""
        if len(df) < 14:
            return 0, ""
            
        rsi = calculate_rsi(df, period=14)
        current_rsi = rsi['rsi'].iloc[-1]
        
        if current_rsi < 30:
            return 1, f"RSI oversold: {current_rsi:.1f}"
        elif current_rsi > 70:
            return -1, f"RSI overbought: {current_rsi:.1f}"
            
        return 0, ""
        
    def should_trade_signal(self, signal: int, current_time: datetime) -> bool:
        """Check if signal should be traded based on confirmation and timing"""
        if signal == 0:
            return False
            
        # Check trade gap
        if self.last_trade_time:
            time_since_last = (current_time - self.last_trade_time).total_seconds() / 60
            if time_since_last < self.min_trade_gap_minutes:
                return False
                
        # Check signal confirmation
        self.signal_history.append(signal)
        if len(self.signal_history) > self.signal_confirmation_bars:
            self.signal_history.pop(0)
            
        # Need consistent signals
        if len(self.signal_history) >= self.signal_confirmation_bars:
            if all(s == signal for s in self.signal_history):
                return True
                
        return False
        
    def execute_trade(self, current_price: float, current_time: datetime, 
                     reason: str, pivot_triggered: bool = False) -> Optional[Trade]:
        """Execute a position flip (always 100% invested)"""
        if not self.position:
            # Initial position - start LONG
            self.position = Position(
                direction=1,
                entry_price=current_price,
                entry_time=current_time,
                size=self.initial_balance / current_price,  # BTC amount
                cost_basis=self.initial_balance
            )
            self.logger.info(f"📈 Initial LONG position: {self.position.size:.8f} BTC @ ${current_price:.2f}")
            return None
            
        # Calculate fees and slippage
        if self.position.direction == 1:  # Currently LONG, flipping to SHORT
            # Selling BTC
            fee, slippage = self.calculate_fees_and_slippage(current_price, self.position.size, False)
            execution_price = current_price + slippage  # Sell at slightly worse price
            
            # Calculate proceeds
            gross_proceeds = self.position.size * execution_price
            net_proceeds = gross_proceeds - fee
            
            # Calculate profit
            profit = net_proceeds - self.position.cost_basis
            
            # Record trade
            trade = Trade(
                timestamp=current_time,
                direction='SELL',
                price=execution_price,
                amount=self.position.size,
                fee=fee,
                slippage=abs(slippage),
                reason=reason,
                regime=self.current_regime,
                position_before=self.position,
                profit=profit,
                pivot_triggered=pivot_triggered
            )
            
            # Update position to SHORT
            self.position = Position(
                direction=-1,
                entry_price=execution_price,
                entry_time=current_time,
                size=net_proceeds,  # USD amount
                cost_basis=net_proceeds
            )
            
            trade.position_after = self.position
            
        else:  # Currently SHORT, flipping to LONG
            # Buying BTC
            fee, slippage = self.calculate_fees_and_slippage(current_price, 
                                                            self.position.size / current_price, True)
            execution_price = current_price + slippage  # Buy at slightly worse price
            
            # Calculate BTC purchased
            available_usd = self.position.size - fee
            btc_amount = available_usd / execution_price
            
            # Record trade
            trade = Trade(
                timestamp=current_time,
                direction='BUY',
                price=execution_price,
                amount=btc_amount,
                fee=fee,
                slippage=abs(slippage),
                reason=reason,
                regime=self.current_regime,
                position_before=self.position,
                profit=0,  # Profit calculated on sells
                pivot_triggered=pivot_triggered
            )
            
            # Update position to LONG
            self.position = Position(
                direction=1,
                entry_price=execution_price,
                entry_time=current_time,
                size=btc_amount,  # BTC amount
                cost_basis=self.position.size  # USD spent
            )
            
            trade.position_after = self.position
            
        # Reset pivot levels for new position
        self.pivot_tracker.levels_locked = False
        
        # Update tracking
        self.trades.append(trade)
        self.last_trade_time = current_time
        
        # Log trade
        direction = "LONG" if self.position.direction == 1 else "SHORT"
        self.logger.info(f"{'🎯 PIVOT' if pivot_triggered else '📊'} {trade.direction} @ ${execution_price:.2f} - Now {direction}")
        if trade.profit != 0:
            self.logger.info(f"   Profit: ${trade.profit:.2f}")
            
        return trade
        
    def calculate_current_equity(self, current_price: float) -> float:
        """Calculate current equity value"""
        if not self.position:
            return self.initial_balance
            
        if self.position.direction == 1:  # LONG
            return self.position.size * current_price
        else:  # SHORT
            return self.position.size
            
    def run_backtest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run the complete backtest"""
        self.logger.info(f"Starting backtest with {len(df)} data points")
        
        # Ensure datetime index
        df = ensure_datetime_index(df)
        
        # Add technical indicators
        df = add_moving_averages(df, self.short_window, self.long_window)
        
        # Need minimum data
        min_periods = max(self.long_window, self.regime_lookback) + 10
        if len(df) < min_periods:
            raise ValueError(f"Insufficient data for backtesting. Need at least {min_periods} periods")
            
        # Main backtest loop
        for i in range(min_periods, len(df)):
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            df_slice = df.iloc[:i+1]
            
            # Track equity
            current_equity = self.calculate_current_equity(current_price)
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': current_equity,
                'price': current_price,
                'position': self.position.direction if self.position else 0
            })
            
            # Update market regime periodically
            if i % 60 == 0:  # Every hour
                new_regime, confidence, metrics = self.detect_market_regime(df_slice)
                
                # Check if regime should switch
                if (self.current_regime != new_regime and 
                    confidence >= self.regime_switch_threshold and
                    (not self.last_regime_switch_time or 
                     (current_time - self.last_regime_switch_time).total_seconds() >= self.min_strategy_switch_minutes * 60)):
                    
                    self.logger.info(f"🔄 Regime switch: {self.current_regime} → {new_regime} (confidence: {confidence:.1%})")
                    self.current_regime = new_regime
                    self.last_regime_switch_time = current_time
                    self.regime_history.append({
                        'timestamp': current_time,
                        'regime': new_regime,
                        'confidence': confidence,
                        'metrics': metrics
                    })
                    
            # Track time in regime
            self.regime_performance[self.current_regime]['time_in_regime'] += 1
            
            # Initialize position if needed
            if not self.position:
                self.execute_trade(current_price, current_time, "Initial position")
                continue
                
            # Update pivot levels
            self.update_pivot_levels(df_slice, current_price, current_time)
            
            # Update trailing pivots
            self.update_trailing_pivot_protection(current_price)
            
            # Check for pivot break FIRST (overrides other signals)
            pivot_break, pivot_reason = self.check_pivot_break(current_price)
            if pivot_break:
                trade = self.execute_trade(current_price, current_time, pivot_reason, pivot_triggered=True)
                if trade:
                    self.regime_performance[self.current_regime]['trades'] += 1
                    if trade.profit > 0:
                        self.regime_performance[self.current_regime]['wins'] += 1
                    self.regime_performance[self.current_regime]['profit'] += trade.profit
                continue
                
            # Generate signals based on regime
            signal = 0
            signal_reason = ""
            
            if self.current_regime == 'trending':
                signal, signal_reason = self.generate_ma_signal(df_slice)
            elif self.current_regime == 'ranging':
                signal, signal_reason = self.generate_rsi_signal(df_slice)
            elif self.current_regime == 'volatile':
                # In volatile regime, rely more on pivot protection
                signal = 0
                
            # Check if we should trade the signal
            if signal != 0 and signal != self.position.direction:
                if self.should_trade_signal(signal, current_time):
                    trade = self.execute_trade(current_price, current_time, 
                                             f"{self.current_regime.capitalize()}: {signal_reason}")
                    if trade:
                        self.regime_performance[self.current_regime]['trades'] += 1
                        if trade.profit > 0:
                            self.regime_performance[self.current_regime]['wins'] += 1
                        self.regime_performance[self.current_regime]['profit'] += trade.profit
                        
        # Calculate final metrics
        final_equity = self.calculate_current_equity(df['close'].iloc[-1])
        total_return = (final_equity - self.initial_balance) / self.initial_balance * 100
        
        # Analyze trades
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.profit > 0]
        losing_trades = [t for t in self.trades if t.profit < 0]
        pivot_trades = [t for t in self.trades if t.pivot_triggered]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = sum(t.profit for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.profit for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Calculate Sharpe ratio
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        daily_returns = equity_df['equity'].resample('D').last().pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 1 else 0
        
        # Calculate maximum drawdown
        equity_series = equity_df['equity']
        cummax = equity_series.expanding().max()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()
        
        results = {
            'initial_balance': self.initial_balance,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'total_fees': sum(t.fee for t in self.trades),
            'total_slippage': sum(t.slippage for t in self.trades),
            'pivot_trades': len(pivot_trades),
            'pivot_win_rate': len([t for t in pivot_trades if t.profit > 0]) / len(pivot_trades) * 100 if pivot_trades else 0,
            'regime_performance': dict(self.regime_performance),
            'regime_switches': len(self.regime_history),
            'final_position': 'LONG' if self.position and self.position.direction == 1 else 'SHORT',
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'regime_history': self.regime_history
        }
        
        return results
        
    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        
        print(f"\nPerformance Metrics:")
        print(f"  Initial Balance:     ${results['initial_balance']:,.2f}")
        print(f"  Final Equity:        ${results['final_equity']:,.2f}")
        print(f"  Total Return:        {results['total_return_pct']:.2f}%")
        print(f"  Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:        {results['max_drawdown_pct']:.2f}%")
        
        print(f"\nTrading Statistics:")
        print(f"  Total Trades:        {results['total_trades']}")
        print(f"  Win Rate:            {results['win_rate']:.1f}%")
        print(f"  Average Win:         ${results['avg_win']:.2f}")
        print(f"  Average Loss:        ${results['avg_loss']:.2f}")
        print(f"  Profit Factor:       {results['profit_factor']:.2f}")
        
        print(f"\nExecution Costs:")
        print(f"  Total Fees:          ${results['total_fees']:.2f}")
        print(f"  Total Slippage:      ${results['total_slippage']:.2f}")
        print(f"  Cost % of P&L:       {(results['total_fees'] + results['total_slippage']) / abs(results['final_equity'] - results['initial_balance']) * 100:.1f}%")
        
        print(f"\nPivot Protection:")
        print(f"  Pivot Trades:        {results['pivot_trades']}")
        print(f"  Pivot Win Rate:      {results['pivot_win_rate']:.1f}%")
        
        print(f"\nRegime Performance:")
        for regime, stats in results['regime_performance'].items():
            if stats['trades'] > 0:
                print(f"\n  {regime.upper()}:")
                print(f"    Trades:          {stats['trades']}")
                print(f"    Win Rate:        {stats['wins'] / stats['trades'] * 100:.1f}%")
                print(f"    Total Profit:    ${stats['profit']:.2f}")
                print(f"    Time in Regime:  {stats['time_in_regime'] / 60:.1f} hours")
                
        print("\n" + "="*60)


def main():
    """Main entry point for backtesting"""
    parser = argparse.ArgumentParser(description='Enhanced Backtesting System with Pivot Protection')
    parser.add_argument('--data', type=str, default='../btcusd.log',
                       help='Path to historical data file')
    parser.add_argument('--config', type=str, default='best_strategy.json',
                       help='Path to strategy configuration file')
    parser.add_argument('--start-date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--initial-balance', type=float, default=10000,
                       help='Initial balance for backtesting')
    parser.add_argument('--save-results', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        print(f"Warning: Config file {args.config} not found, using defaults")
        
    # Override with command line arguments
    if args.initial_balance:
        config['initial_balance'] = args.initial_balance
        
    # Load data
    print(f"Loading data from {args.data}...")
    df = parse_log_file(args.data)
    
    if df is None or len(df) == 0:
        print("Error: No data loaded")
        return
        
    print(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
    
    # Filter by date if specified
    if args.start_date:
        start_date = pd.to_datetime(args.start_date)
        df = df[df.index >= start_date]
        
    if args.end_date:
        end_date = pd.to_datetime(args.end_date)
        df = df[df.index <= end_date]
        
    # Run backtest
    backtester = EnhancedBacktester(config)
    results = backtester.run_backtest(df)
    
    # Print summary
    backtester.print_summary(results)
    
    # Save results if requested
    if args.save_results:
        # Convert non-serializable objects for JSON
        save_results = results.copy()
        save_results['trades'] = [
            {
                'timestamp': t.timestamp.isoformat(),
                'direction': t.direction,
                'price': t.price,
                'amount': t.amount,
                'fee': t.fee,
                'profit': t.profit,
                'reason': t.reason,
                'pivot_triggered': t.pivot_triggered
            }
            for t in results['trades']
        ]
        save_results['equity_curve'] = [
            {
                'timestamp': e['timestamp'].isoformat(),
                'equity': e['equity'],
                'price': e['price'],
                'position': e['position']
            }
            for e in results['equity_curve']
        ]
        
        with open(args.save_results, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()