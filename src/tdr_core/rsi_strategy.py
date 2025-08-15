#!/usr/bin/env python3
"""
Adaptive RSI Strategy for ranging/sideways markets
Better suited for choppy conditions than MA crossover
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

class AdaptiveRSIStrategy:
    """
    RSI-based strategy that adapts thresholds based on market volatility
    Works better in ranging markets where MA crossovers fail
    """
    
    def __init__(
        self,
        data_manager,
        amount,
        symbol,
        logger,
        rsi_period=14,
        oversold=30,
        overbought=70,
        adaptive=True,
        volatility_window=20,
        live_trading=False,
        **kwargs
    ):
        self.data_manager = data_manager
        self.amount = amount
        self.symbol = symbol
        self.logger = logger
        self.rsi_period = rsi_period
        self.base_oversold = oversold
        self.base_overbought = overbought
        self.adaptive = adaptive
        self.volatility_window = volatility_window
        self.live_trading = live_trading
        
        # Position tracking
        self.position = kwargs.get('initial_position', -1)  # Start SHORT
        self.balance_btc = kwargs.get('initial_balance_btc', 0.0)
        self.balance_usd = kwargs.get('initial_balance_usd', amount)
        self.entry_price = 0
        self.trades = []
        
        # Adaptive thresholds
        self.current_oversold = oversold
        self.current_overbought = overbought
        
        self.logger.info(f"Initialized AdaptiveRSI Strategy - Period: {rsi_period}, "
                        f"Oversold: {oversold}, Overbought: {overbought}, "
                        f"Adaptive: {adaptive}")
    
    def calculate_rsi(self, prices):
        """Calculate RSI indicator"""
        if len(prices) < self.rsi_period + 1:
            return None
            
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=self.rsi_period).mean()
        avg_loss = losses.rolling(window=self.rsi_period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None
    
    def calculate_volatility(self, prices):
        """Calculate recent volatility for adaptive thresholds"""
        if len(prices) < self.volatility_window:
            return 0.02  # Default 2% volatility
            
        returns = prices.pct_change().dropna()
        volatility = returns.rolling(window=self.volatility_window).std().iloc[-1]
        return volatility if not pd.isna(volatility) else 0.02
    
    def adapt_thresholds(self, volatility):
        """Adapt RSI thresholds based on market volatility"""
        if not self.adaptive:
            return
            
        # In high volatility, widen the thresholds (more extreme RSI needed)
        # In low volatility, tighten the thresholds (trade on smaller RSI moves)
        
        # Normal volatility ~1-2% daily
        # High volatility >3% daily
        # Low volatility <1% daily
        
        if volatility > 0.03:  # High volatility
            # Widen thresholds - need more extreme RSI
            adjustment = min(10, volatility * 200)  # Max 10 point adjustment
            self.current_oversold = max(20, self.base_oversold - adjustment)
            self.current_overbought = min(80, self.base_overbought + adjustment)
        elif volatility < 0.01:  # Low volatility
            # Tighten thresholds - trade on smaller moves
            adjustment = 5
            self.current_oversold = min(40, self.base_oversold + adjustment)
            self.current_overbought = max(60, self.base_overbought - adjustment)
        else:  # Normal volatility
            self.current_oversold = self.base_oversold
            self.current_overbought = self.base_overbought
    
    def evaluate_signal(self, df, current_position, timestamp=None):
        """
        Evaluate RSI signal for trading decision
        Returns dict with signal details
        """
        # Need enough data for RSI calculation
        if len(df) < self.rsi_period + 1:
            return {
                'action': 'NO_TRADE',
                'reason': 'Insufficient data for RSI',
                'position': current_position
            }
        
        # Get price data
        prices = df['close'] if 'close' in df.columns else df['price']
        current_price = prices.iloc[-1]
        
        # Calculate RSI
        rsi = self.calculate_rsi(prices)
        if rsi is None:
            return {
                'action': 'NO_TRADE',
                'reason': 'RSI calculation failed',
                'position': current_position
            }
        
        # Calculate volatility and adapt thresholds
        volatility = self.calculate_volatility(prices)
        self.adapt_thresholds(volatility)
        
        # Determine signal
        action = 'NO_TRADE'
        reason = f'RSI: {rsi:.1f} (OS:{self.current_oversold:.0f}/OB:{self.current_overbought:.0f})'
        
        if current_position == -1:  # Currently SHORT (holding USD)
            if rsi < self.current_oversold:
                action = 'BUY'
                reason = f'RSI oversold: {rsi:.1f} < {self.current_oversold:.0f}'
        elif current_position == 1:  # Currently LONG (holding BTC)
            if rsi > self.current_overbought:
                action = 'SELL'
                reason = f'RSI overbought: {rsi:.1f} > {self.current_overbought:.0f}'
        
        # Log evaluation
        self.logger.info(f"RSI Signal: {action} - {reason} - Volatility: {volatility:.4f}")
        
        return {
            'action': action,
            'reason': reason,
            'position': current_position,
            'rsi': rsi,
            'oversold': self.current_oversold,
            'overbought': self.current_overbought,
            'volatility': volatility,
            'price': current_price,
            'timestamp': timestamp or datetime.now()
        }
    
    def execute_trade(self, signal, current_price):
        """Execute trade based on signal"""
        if signal['action'] == 'BUY':
            # Buy BTC with USD
            fee = self.balance_usd * 0.0025
            self.balance_btc = (self.balance_usd - fee) / current_price
            self.balance_usd = 0
            self.position = 1
            self.entry_price = current_price
            
            trade = {
                'timestamp': signal['timestamp'],
                'type': 'BUY',
                'price': current_price,
                'btc_amount': self.balance_btc,
                'rsi': signal['rsi'],
                'reason': signal['reason']
            }
            self.trades.append(trade)
            self.logger.info(f"Executed BUY: {self.balance_btc:.8f} BTC @ ${current_price:,.2f}")
            return trade
            
        elif signal['action'] == 'SELL':
            # Sell BTC for USD
            gross_usd = self.balance_btc * current_price
            fee = gross_usd * 0.0025
            self.balance_usd = gross_usd - fee
            
            # Calculate P&L
            pnl = self.balance_usd - self.amount
            pnl_pct = (pnl / self.amount) * 100
            
            trade = {
                'timestamp': signal['timestamp'],
                'type': 'SELL',
                'price': current_price,
                'usd_amount': self.balance_usd,
                'rsi': signal['rsi'],
                'reason': signal['reason'],
                'pnl': pnl,
                'pnl_pct': pnl_pct
            }
            
            self.balance_btc = 0
            self.position = -1
            self.entry_price = 0
            self.trades.append(trade)
            
            self.logger.info(f"Executed SELL: ${self.balance_usd:,.2f} @ ${current_price:,.2f} "
                           f"(P&L: ${pnl:,.2f} / {pnl_pct:.2f}%)")
            return trade
        
        return None
    
    def get_status(self):
        """Get current strategy status"""
        current_value = self.balance_usd if self.position == -1 else 0
        
        return {
            'strategy': 'AdaptiveRSI',
            'position': self.position,
            'balance_btc': self.balance_btc,
            'balance_usd': self.balance_usd,
            'entry_price': self.entry_price,
            'current_value': current_value,
            'total_trades': len(self.trades),
            'rsi_period': self.rsi_period,
            'current_oversold': self.current_oversold,
            'current_overbought': self.current_overbought,
            'adaptive': self.adaptive
        }