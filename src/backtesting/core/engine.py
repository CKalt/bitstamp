"""
Backtesting Engine - Exactly replicates live trading behavior
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path

# Import existing live trading components to ensure exact behavior match
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from tdr_core.strategy_core import AdaptiveStrategyCore
from strategies import ExchangeHandlerBase

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Represents a single trade in backtesting"""
    timestamp: datetime
    side: str  # 'buy' or 'sell'
    price: float
    amount: float
    fee: float
    total_cost: float
    position_after: float
    balance_btc_after: float
    balance_usd_after: float
    signal_reason: str
    market_regime: str
    
    def to_dict(self):
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_btc: float = 0.0
    initial_usd: float = 100000.0
    fee_percentage: float = 0.0012  # 0.12% Bitstamp fee
    max_trades_per_day: int = 5
    max_trades_per_hour: int = 3
    min_trade_gap_minutes: int = 15
    min_btc_trade_size: float = 1e-8
    always_in_market: bool = True  # Always 100% BTC or 100% USD
    enable_pivot_protection: bool = True
    enable_trailing_stops: bool = True
    emergency_exit_loss: float = -2000.0
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


class BacktestPositionTracker:
    """Tracks position and balances exactly like live system"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.balance_btc = config.initial_btc
        self.balance_usd = config.initial_usd
        self.position_size = 0.0  # BTC amount (negative for short)
        self.position_cost_basis = 0.0  # USD value of position
        self.trades: List[BacktestTrade] = []
        self.daily_trade_count = {}
        self.hourly_trade_count = {}
        
    @property
    def position(self) -> int:
        """Returns 1 for long (BTC), -1 for short (USD), matching live system"""
        if abs(self.position_size) < self.config.min_btc_trade_size:
            return -1  # Effectively in USD
        return 1 if self.position_size > 0 else -1
    
    @property
    def entry_price(self) -> Optional[float]:
        """Calculate entry price from position tracking"""
        if abs(self.position_size) < self.config.min_btc_trade_size:
            return None
        return abs(self.position_cost_basis / self.position_size)
    
    def can_trade(self, timestamp: datetime) -> Tuple[bool, str]:
        """Check if trading is allowed based on limits"""
        # Check daily limit
        date_key = timestamp.date()
        if self.daily_trade_count.get(date_key, 0) >= self.config.max_trades_per_day:
            return False, f"Daily trade limit reached ({self.config.max_trades_per_day})"
        
        # Check hourly limit
        hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
        if self.hourly_trade_count.get(hour_key, 0) >= self.config.max_trades_per_hour:
            return False, f"Hourly trade limit reached ({self.config.max_trades_per_hour})"
        
        # Check minimum gap
        if self.trades:
            last_trade_time = self.trades[-1].timestamp
            gap = (timestamp - last_trade_time).total_seconds() / 60
            if gap < self.config.min_trade_gap_minutes:
                return False, f"Too soon after last trade (gap: {gap:.1f} min)"
        
        return True, "OK"
    
    def execute_trade(self, timestamp: datetime, side: str, price: float, 
                     signal_reason: str, market_regime: str) -> Optional[BacktestTrade]:
        """Execute a trade with exact fee calculation matching live system"""
        can_trade, reason = self.can_trade(timestamp)
        if not can_trade:
            logger.debug(f"Trade rejected: {reason}")
            return None
        
        if side == 'buy':
            # Calculate max BTC we can buy with available USD
            max_btc = self.balance_usd / (price * (1 + self.config.fee_percentage))
            if max_btc < self.config.min_btc_trade_size:
                logger.debug(f"Insufficient USD balance for min trade size")
                return None
            
            btc_amount = max_btc if self.config.always_in_market else max_btc
            total_cost = btc_amount * price * (1 + self.config.fee_percentage)
            fee = btc_amount * price * self.config.fee_percentage
            
            # Update balances
            self.balance_usd -= total_cost
            self.balance_btc += btc_amount
            
            # Update position tracking (going long)
            self.position_size = btc_amount
            self.position_cost_basis = total_cost
            
        else:  # sell
            # Sell all BTC
            if self.balance_btc < self.config.min_btc_trade_size:
                logger.debug(f"Insufficient BTC balance for min trade size")
                return None
            
            btc_amount = self.balance_btc
            gross_proceeds = btc_amount * price
            fee = gross_proceeds * self.config.fee_percentage
            net_proceeds = gross_proceeds - fee
            
            # Update balances
            self.balance_btc = 0.0
            self.balance_usd += net_proceeds
            
            # Update position tracking (going short/USD)
            self.position_size = 0.0
            self.position_cost_basis = 0.0
        
        # Create trade record
        trade = BacktestTrade(
            timestamp=timestamp,
            side=side,
            price=price,
            amount=btc_amount,
            fee=fee,
            total_cost=total_cost if side == 'buy' else net_proceeds,
            position_after=self.position,
            balance_btc_after=self.balance_btc,
            balance_usd_after=self.balance_usd,
            signal_reason=signal_reason,
            market_regime=market_regime
        )
        
        self.trades.append(trade)
        
        # Update trade counts
        date_key = timestamp.date()
        hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
        self.daily_trade_count[date_key] = self.daily_trade_count.get(date_key, 0) + 1
        self.hourly_trade_count[hour_key] = self.hourly_trade_count.get(hour_key, 0) + 1
        
        return trade
    
    def get_total_value(self, current_price: float) -> float:
        """Calculate total portfolio value in USD"""
        return self.balance_usd + (self.balance_btc * current_price)
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L for current position"""
        if self.position == 1 and self.entry_price:
            # Long position
            current_value = self.position_size * current_price
            return current_value - self.position_cost_basis
        return 0.0


class SimulatedExchangeHandler(ExchangeHandlerBase):
    """Simulates exchange for backtesting using historical data"""
    
    def __init__(self, position_tracker: BacktestPositionTracker):
        self.position_tracker = position_tracker
        self.current_price = None
        self.current_timestamp = None
        self.current_signal_reason = ""
        self.current_market_regime = "unknown"
        
    def set_context(self, timestamp: datetime, price: float, signal_reason: str, market_regime: str):
        """Set current market context for trade execution"""
        self.current_timestamp = timestamp
        self.current_price = price
        self.current_signal_reason = signal_reason
        self.current_market_regime = market_regime
    
    def get_balance(self):
        """Return current balances"""
        return {
            'BTC': self.position_tracker.balance_btc,
            'USD': self.position_tracker.balance_usd
        }
    
    def place_order(self, side, amount, price=None):
        """Simulate order execution"""
        if not self.current_timestamp or not self.current_price:
            raise ValueError("Context not set for order execution")
        
        # Use current market price if not specified
        exec_price = price or self.current_price
        
        # Execute trade through position tracker
        trade = self.position_tracker.execute_trade(
            timestamp=self.current_timestamp,
            side=side,
            price=exec_price,
            signal_reason=self.current_signal_reason,
            market_regime=self.current_market_regime
        )
        
        if trade:
            logger.info(f"Executed {side} {trade.amount:.6f} BTC @ ${exec_price:.2f}")
            return {'id': f"sim_{len(self.position_tracker.trades)}", 'status': 'filled'}
        else:
            return {'id': None, 'status': 'rejected'}
    
    def get_position(self):
        """Get current position for compatibility"""
        return self.position_tracker.position
    
    def get_entry_price(self):
        """Get entry price for current position"""
        return self.position_tracker.entry_price


class BacktestEngine:
    """
    Main backtesting engine that orchestrates the simulation
    Uses existing strategy classes to ensure exact behavior match
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.position_tracker = BacktestPositionTracker(config)
        self.exchange_handler = SimulatedExchangeHandler(self.position_tracker)
        
        # Initialize strategy using existing core
        self.strategy = AdaptiveStrategyCore(
            exchange_handler=self.exchange_handler,
            initial_capital=config.initial_usd
        )
        
        # Results tracking
        self.equity_curve = []
        self.signals = []
        self.regime_history = []
        
    def run(self, data: pd.DataFrame, start_date: Optional[datetime] = None, 
            end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Run backtest on historical data
        
        Args:
            data: DataFrame with OHLCV data and timestamp index
            start_date: Start of backtest period
            end_date: End of backtest period
            
        Returns:
            Dictionary with backtest results
        """
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        logger.info(f"Running backtest from {data.index[0]} to {data.index[-1]}")
        logger.info(f"Initial capital: ${self.config.initial_usd:,.2f}")
        
        # Process each bar
        for timestamp, row in data.iterrows():
            self._process_bar(timestamp, row)
        
        # Calculate final metrics
        results = self._calculate_results(data)
        
        logger.info(f"Backtest complete: {len(self.position_tracker.trades)} trades")
        logger.info(f"Final value: ${results['final_value']:,.2f}")
        logger.info(f"Total return: {results['total_return']:.2%}")
        
        return results
    
    def _process_bar(self, timestamp: datetime, bar: pd.Series):
        """Process a single bar of data"""
        current_price = bar['close']
        
        # Update exchange context
        self.exchange_handler.set_context(
            timestamp=timestamp,
            price=current_price,
            signal_reason="",
            market_regime="unknown"
        )
        
        # Get current position before signal
        position_before = self.position_tracker.position
        
        # Generate signal from strategy
        signal_data = self.strategy.generate_signal(
            data=bar.to_frame().T,  # Convert to DataFrame format expected by strategy
            current_position=position_before
        )
        
        # Extract signal info
        signal = signal_data.get('signal', 0)
        regime = signal_data.get('regime', 'unknown')
        reason = signal_data.get('reason', '')
        
        # Update context with signal info
        self.exchange_handler.set_context(
            timestamp=timestamp,
            price=current_price,
            signal_reason=reason,
            market_regime=regime
        )
        
        # Execute trade if signal differs from position
        if signal != 0 and signal != position_before:
            side = 'buy' if signal == 1 else 'sell'
            self.exchange_handler.place_order(side=side, amount=None)  # Amount calculated internally
        
        # Track equity and signals
        total_value = self.position_tracker.get_total_value(current_price)
        self.equity_curve.append({
            'timestamp': timestamp,
            'value': total_value,
            'price': current_price,
            'position': self.position_tracker.position,
            'btc_balance': self.position_tracker.balance_btc,
            'usd_balance': self.position_tracker.balance_usd
        })
        
        self.signals.append({
            'timestamp': timestamp,
            'signal': signal,
            'regime': regime,
            'reason': reason
        })
        
        self.regime_history.append({
            'timestamp': timestamp,
            'regime': regime
        })
    
    def _calculate_results(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive backtest results"""
        equity_df = pd.DataFrame(self.equity_curve)
        trades_data = [t.to_dict() for t in self.position_tracker.trades]
        
        initial_value = self.config.initial_usd
        final_value = equity_df['value'].iloc[-1] if len(equity_df) > 0 else initial_value
        
        results = {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': (final_value - initial_value) / initial_value,
            'num_trades': len(self.position_tracker.trades),
            'trades': trades_data,
            'equity_curve': equity_df.to_dict('records'),
            'signals': self.signals,
            'regime_history': self.regime_history,
            'config': asdict(self.config)
        }
        
        # Add performance metrics if we have trades
        if len(self.position_tracker.trades) > 0:
            results.update(self._calculate_trade_metrics())
        
        return results
    
    def _calculate_trade_metrics(self) -> Dict[str, Any]:
        """Calculate detailed trade-based metrics"""
        trades = self.position_tracker.trades
        
        # Calculate P&L for each trade
        pnls = []
        for i in range(1, len(trades)):
            if trades[i].side == 'sell' and trades[i-1].side == 'buy':
                # Complete round trip
                buy_cost = trades[i-1].total_cost
                sell_proceeds = trades[i].total_cost  # For sells, this is net proceeds
                pnl = sell_proceeds - buy_cost
                pnl_pct = pnl / buy_cost
                pnls.append({
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'buy_time': trades[i-1].timestamp,
                    'sell_time': trades[i].timestamp,
                    'duration': (trades[i].timestamp - trades[i-1].timestamp).total_seconds() / 3600
                })
        
        if not pnls:
            return {}
        
        # Calculate metrics
        pnl_values = [p['pnl'] for p in pnls]
        pnl_pcts = [p['pnl_pct'] for p in pnls]
        
        wins = [p for p in pnl_values if p > 0]
        losses = [p for p in pnl_values if p < 0]
        
        metrics = {
            'win_rate': len(wins) / len(pnl_values) if pnl_values else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses else float('inf'),
            'max_win': max(pnl_values) if pnl_values else 0,
            'max_loss': min(pnl_values) if pnl_values else 0,
            'avg_trade_pnl': np.mean(pnl_values) if pnl_values else 0,
            'avg_trade_pnl_pct': np.mean(pnl_pcts) if pnl_pcts else 0,
            'trade_durations': [p['duration'] for p in pnls],
            'round_trips': pnls
        }
        
        return metrics