# src/tdr_core/strategy_core.py
"""
Core strategy logic shared between live trading and backtesting.
This module contains the pure strategy logic without dependencies on live trading components.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
# No longer importing from technical_indicators to avoid datetime issues
# All calculations are done manually within this module

class AdaptiveStrategyCore:
    """
    Core adaptive strategy logic that can be used by both live trading and backtesting.
    This contains only the pure strategy logic without live trading dependencies.
    """
    
    def __init__(self, short_window: int = 10, long_window: int = 46):
        self.short_window = short_window
        self.long_window = long_window
        
        # Default parameters (can be overridden)
        self.regime_switch_threshold = 0.40
        self.signal_confirmation_bars = 2
        self.min_trade_gap_minutes = 15
        self.whipsaw_threshold = 8.0
        self.regime_lookback = 100
        self.bb_std_dev = 2.0
        self.emergency_loss_threshold = -2000
        self.max_trades_per_day = 5
        
    def detect_market_regime(self, df: pd.DataFrame) -> Tuple[str, float, Dict]:
        """
        Detect current market regime based on price action.
        
        Returns:
            Tuple of (regime, confidence, metrics)
            regime: 'trending', 'ranging', or 'volatile'
            confidence: 0.0 to 1.0
            metrics: dict with analysis metrics
        """
        if len(df) < self.regime_lookback:
            return 'trending', 0.5, {}
            
        recent_df = df.tail(self.regime_lookback).copy()
        
        # Ensure we have the right column
        if 'close' not in recent_df.columns and 'price' in recent_df.columns:
            recent_df['close'] = recent_df['price']
        
        # Calculate trend metrics
        price_start = recent_df['close'].iloc[0]
        price_end = recent_df['close'].iloc[-1]
        price_high = recent_df['close'].max()
        price_low = recent_df['close'].min()
        
        total_return = abs(price_end - price_start) / price_start
        price_range = (price_high - price_low) / ((price_high + price_low) / 2)
        trend_strength = total_return / price_range if price_range > 0 else 0
        
        # Volatility
        returns = recent_df['close'].pct_change().dropna()
        volatility = returns.std() if len(returns) > 1 else 0
        
        # Whipsaw detection - manual calculation to avoid datetime issues
        df_temp = recent_df.copy()
        df_temp[f'MA_{self.short_window}'] = df_temp['close'].rolling(window=self.short_window).mean()
        df_temp[f'MA_{self.long_window}'] = df_temp['close'].rolling(window=self.long_window).mean()
        
        # Generate signals manually
        df_temp['MA_Signal'] = 0
        df_temp.loc[df_temp[f'MA_{self.short_window}'] > df_temp[f'MA_{self.long_window}'], 'MA_Signal'] = 1
        df_temp.loc[df_temp[f'MA_{self.short_window}'] < df_temp[f'MA_{self.long_window}'], 'MA_Signal'] = -1
        
        signal_changes = (df_temp['MA_Signal'].diff() != 0).sum()
        whipsaw_ratio = (signal_changes / len(df_temp)) * 100
        
        # Range-bound detection
        ma_20 = recent_df['close'].rolling(20).mean()
        price_vs_ma = (recent_df['close'] - ma_20) / ma_20
        range_bound_score = 1.0 - abs(price_vs_ma.mean()) if not price_vs_ma.empty else 0
        
        # Regime scoring
        regime_scores = {'trending': 0.0, 'ranging': 0.0, 'volatile': 0.0}
        
        # TRENDING indicators
        if trend_strength > 0.3:
            regime_scores['trending'] += 2.0
        if whipsaw_ratio < 2.0:
            regime_scores['trending'] += 2.0
        if volatility < 0.02:
            regime_scores['trending'] += 1.5
            
        # Check for sustained directional movement
        if len(recent_df) >= 10:
            recent_closes = recent_df['close'].tail(10)
            directional_moves = 0
            for i in range(1, len(recent_closes)):
                if (recent_closes.iloc[i] > recent_closes.iloc[i-1] and price_end > price_start) or \
                   (recent_closes.iloc[i] < recent_closes.iloc[i-1] and price_end < price_start):
                    directional_moves += 1
            if directional_moves >= 6:
                regime_scores['trending'] += 1.0
        
        # RANGING indicators
        if range_bound_score > 0.8:
            regime_scores['ranging'] += 2.0
        if whipsaw_ratio > self.whipsaw_threshold:
            regime_scores['ranging'] += 2.0
        if trend_strength < 0.15:
            regime_scores['ranging'] += 1.0
        
        # VOLATILE indicators
        if volatility > 0.035:
            regime_scores['volatile'] += 2.0
        if price_range > 0.08:
            regime_scores['volatile'] += 1.0
        
        # Determine regime
        best_regime = max(regime_scores.items(), key=lambda x: x[1])
        regime = best_regime[0]
        max_score = best_regime[1]
        confidence = min(0.95, max_score / 6.0)
        
        metrics = {
            'whipsaw_ratio': whipsaw_ratio,
            'trend_strength': trend_strength,
            'volatility': volatility,
            'range_bound_score': range_bound_score,
            'regime_scores': regime_scores
        }
        
        return regime, confidence, metrics
    
    def generate_trending_signal(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Generate MA crossover signals for trending markets."""
        if len(df) < self.long_window + 1:
            return 0, "Insufficient data for trending strategy"
        
        # Ensure we have the right column
        if 'close' not in df.columns and 'price' in df.columns:
            df = df.copy()
            df['close'] = df['price']
        
        # Manual MA calculation to avoid datetime issues
        df_ma = df.copy()
        df_ma[f'MA_{self.short_window}'] = df_ma['close'].rolling(window=self.short_window).mean()
        df_ma[f'MA_{self.long_window}'] = df_ma['close'].rolling(window=self.long_window).mean()
        
        # Generate signals manually
        df_ma['MA_Signal'] = 0
        df_ma.loc[df_ma[f'MA_{self.short_window}'] > df_ma[f'MA_{self.long_window}'], 'MA_Signal'] = 1
        df_ma.loc[df_ma[f'MA_{self.short_window}'] < df_ma[f'MA_{self.long_window}'], 'MA_Signal'] = -1
        
        current_signal = df_ma.iloc[-1]['MA_Signal']
        prev_signal = df_ma.iloc[-2]['MA_Signal'] if len(df_ma) > 1 else 0
        
        # Check for signal confirmation
        if self.signal_confirmation_bars > 1:
            confirmation_count = 0
            for i in range(1, min(self.signal_confirmation_bars + 1, len(df_ma))):
                if df_ma.iloc[-i]['MA_Signal'] == current_signal:
                    confirmation_count += 1
            
            if confirmation_count < self.signal_confirmation_bars:
                return 0, "Waiting for signal confirmation"
        
        if current_signal != prev_signal:
            if current_signal == 1:
                return 1, "MA crossover BUY signal"
            elif current_signal == -1:
                return -1, "MA crossover SELL signal"
        
        return 0, "No signal change"
    
    def generate_ranging_signal(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Generate mean reversion signals for ranging markets."""
        if len(df) < 20:
            return 0, "Insufficient data for ranging strategy"
        
        # Ensure we have the right column
        if 'close' not in df.columns and 'price' in df.columns:
            df = df.copy()
            df['close'] = df['price']
        
        # Calculate RSI manually to avoid datetime issues
        df_rsi = df.copy()
        price_diff = df_rsi['close'].diff()
        gain = price_diff.where(price_diff > 0, 0)
        loss = -price_diff.where(price_diff < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # Calculate Bollinger Bands manually
        bb_ma = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        bb_upper = bb_ma + (bb_std * self.bb_std_dev)
        bb_lower = bb_ma - (bb_std * self.bb_std_dev)
        
        current_price = df.iloc[-1]['close']
        current_bb_upper = bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else current_price * 1.02
        current_bb_lower = bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else current_price * 0.98
        current_bb_middle = bb_ma.iloc[-1] if not pd.isna(bb_ma.iloc[-1]) else current_price
        
        signal = 0
        reason = "No mean reversion signal"
        
        # Mean reversion logic with volatility adjustment
        volatility = df['close'].pct_change().tail(10).std()
        
        # Dynamic thresholds based on recent volatility
        rsi_oversold = 30 if volatility < 0.02 else 25
        rsi_overbought = 70 if volatility < 0.02 else 75
        
        # Enhanced mean reversion conditions
        price_below_lower = current_price <= current_bb_lower
        price_above_upper = current_price >= current_bb_upper
        
        # Additional confirmation: price movement
        price_momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        
        if price_below_lower and current_rsi < rsi_oversold:
            # Extra confirmation for high volatility
            if volatility > 0.03 and price_momentum < -0.02:
                signal = 1
                reason = f"Strong oversold: Price at lower BB, RSI={current_rsi:.1f}, momentum={price_momentum:.1%}"
            else:
                signal = 1
                reason = f"Buy signal: Price at lower BB, RSI={current_rsi:.1f}"
        elif price_above_upper and current_rsi > rsi_overbought:
            # Extra confirmation for high volatility
            if volatility > 0.03 and price_momentum > 0.02:
                signal = -1
                reason = f"Strong overbought: Price at upper BB, RSI={current_rsi:.1f}, momentum={price_momentum:.1%}"
            else:
                signal = -1
                reason = f"Sell signal: Price at upper BB, RSI={current_rsi:.1f}"
        # Additional mean reversion opportunity
        elif current_rsi < 20:
            signal = 1
            reason = f"Extreme oversold: RSI={current_rsi:.1f}"
        elif current_rsi > 80:
            signal = -1
            reason = f"Extreme overbought: RSI={current_rsi:.1f}"
        
        return signal, reason
    
    def generate_volatile_signal(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Generate breakout signals for volatile markets."""
        if len(df) < 20:
            return 0, "Insufficient data for volatile strategy"
        
        # Ensure we have the right column
        if 'close' not in df.columns and 'price' in df.columns:
            df = df.copy()
            df['close'] = df['price']
        
        # Calculate tighter Bollinger Bands manually for volatile markets
        bb_ma = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        bb_upper = bb_ma + (bb_std * 1.5)
        bb_lower = bb_ma - (bb_std * 1.5)
        
        current_price = df.iloc[-1]['close']
        current_bb_upper = bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else current_price * 1.015
        current_bb_lower = bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else current_price * 0.985
        
        # Calculate momentum
        momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        
        # Volume analysis (if available)
        volume_surge = False
        if 'volume' in df.columns:
            recent_volume = df['volume'].tail(5).mean()
            avg_volume = df['volume'].tail(20).mean()
            volume_surge = recent_volume > avg_volume * 1.5
        
        # Breakout logic
        if current_price > current_bb_upper:
            if momentum > 0.01 or volume_surge:
                return 1, f"Volatility breakout BUY (momentum={momentum:.1%})"
            else:
                return 0, "Breakout lacks momentum confirmation"
        elif current_price < current_bb_lower:
            if momentum < -0.01 or volume_surge:
                return -1, f"Volatility breakout SELL (momentum={momentum:.1%})"
            else:
                return 0, "Breakout lacks momentum confirmation"
        
        return 0, "No breakout signal"
    
    def should_generate_signal(self, df: pd.DataFrame, current_regime: str) -> Tuple[int, str]:
        """
        Main signal generation method that routes to appropriate strategy based on regime.
        
        Returns:
            Tuple of (signal, reason)
            signal: 1 for buy, -1 for sell, 0 for no action
            reason: explanation of the signal
        """
        if current_regime == 'trending':
            return self.generate_trending_signal(df)
        elif current_regime == 'ranging':
            return self.generate_ranging_signal(df)
        elif current_regime == 'volatile':
            return self.generate_volatile_signal(df)
        else:
            return 0, f"Unknown regime: {current_regime}"
    
    def check_emergency_exit(self, current_loss: float) -> bool:
        """Check if emergency exit conditions are met."""
        return current_loss < self.emergency_loss_threshold
    
    def update_parameters(self, params: Dict):
        """Update strategy parameters."""
        if 'short_window' in params:
            self.short_window = params['short_window']
        if 'long_window' in params:
            self.long_window = params['long_window']
        if 'regime_switch_threshold' in params:
            self.regime_switch_threshold = params['regime_switch_threshold']
        if 'signal_confirmation_bars' in params:
            self.signal_confirmation_bars = params['signal_confirmation_bars']
        if 'min_trade_gap_minutes' in params:
            self.min_trade_gap_minutes = params['min_trade_gap_minutes']
        if 'whipsaw_threshold' in params:
            self.whipsaw_threshold = params['whipsaw_threshold']
        if 'emergency_loss_threshold' in params:
            self.emergency_loss_threshold = params['emergency_loss_threshold']
        if 'max_trades_per_day' in params:
            self.max_trades_per_day = params['max_trades_per_day']