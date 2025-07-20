"""
Backtest Data Loader - Loads and prepares historical data exactly like live system
"""
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.loader import parse_log_file

logger = logging.getLogger(__name__)


class BacktestDataManager:
    """
    Manages historical data for backtesting, replicating live data flow exactly
    """
    
    def __init__(self, log_file_path: str = "btcusd.log", 
                 primary_timeframe: str = "15T",  # Default to 15-minute like main branch
                 secondary_timeframe: str = "1H"):
        self.log_file_path = log_file_path
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframe = secondary_timeframe
        self.raw_trades_df = None
        self.ohlcv_1min_df = None
        self.ohlcv_primary_df = None  # Was ohlcv_1hour_df
        self.ohlcv_secondary_df = None
        
    def load_data(self, start_date: Optional[datetime] = None, 
                  end_date: Optional[datetime] = None,
                  progress_callback=None) -> pd.DataFrame:
        """
        Load historical data from btcusd.log and prepare for backtesting
        
        Returns primary timeframe OHLCV data (default 15-minute like main branch)
        """
        logger.info(f"Loading data from {self.log_file_path}")
        
        # Step 1: Parse raw trades from log file using existing parser
        self.raw_trades_df = parse_log_file(
            self.log_file_path,
            progress_callback=progress_callback
        )
        
        if self.raw_trades_df.empty:
            raise ValueError("No data loaded from log file")
        
        logger.info(f"Loaded {len(self.raw_trades_df)} raw trades")
        
        # Step 2: Filter by date range if specified
        if start_date or end_date:
            self.raw_trades_df = self._filter_by_date(
                self.raw_trades_df, start_date, end_date
            )
        
        # Step 3: Convert raw trades to 1-minute OHLCV (matching live system)
        self.ohlcv_1min_df = self._create_1min_candles(self.raw_trades_df)
        
        # Step 4: Resample to primary timeframe (default 15-minute like main branch)
        self.ohlcv_primary_df = self._resample_to_timeframe(self.ohlcv_1min_df, self.primary_timeframe)
        
        # Step 5: If secondary timeframe requested, create that too
        if self.secondary_timeframe:
            self.ohlcv_secondary_df = self._resample_to_timeframe(self.ohlcv_1min_df, self.secondary_timeframe)
        
        # Step 6: Add required fields for strategy
        self._prepare_strategy_data()
        
        logger.info(f"Prepared {len(self.ohlcv_primary_df)} {self.primary_timeframe} candles")
        logger.info(f"Date range: {self.ohlcv_primary_df.index[0]} to {self.ohlcv_primary_df.index[-1]}")
        
        return self.ohlcv_primary_df
    
    def _filter_by_date(self, df: pd.DataFrame, 
                       start_date: Optional[datetime], 
                       end_date: Optional[datetime]) -> pd.DataFrame:
        """Filter DataFrame by date range"""
        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('datetime', inplace=True)
            elif 'datetime' in df.columns:
                df.set_index('datetime', inplace=True)
                
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        logger.info(f"Filtered to {len(df)} trades in date range")
        return df
    
    def _create_1min_candles(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw trades to 1-minute OHLCV candles
        Matches live system aggregation in DataManager
        """
        # Ensure we have a datetime index
        if not isinstance(trades_df.index, pd.DatetimeIndex):
            trades_df['datetime'] = pd.to_datetime(trades_df['timestamp'], unit='s')
            trades_df.set_index('datetime', inplace=True)
        
        # Aggregate trades to 1-minute candles
        ohlcv = trades_df.resample('1T').agg({
            'price': ['first', 'max', 'min', 'last'],
            'amount': 'sum',
            'timestamp': 'last'
        })
        
        # Flatten column names
        ohlcv.columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        
        # Forward fill gaps (markets can have periods with no trades)
        ohlcv = ohlcv.fillna(method='ffill')
        
        # Drop any remaining NaN rows (at the beginning)
        ohlcv = ohlcv.dropna()
        
        # Add trade count
        trade_counts = trades_df.resample('1T').size()
        ohlcv['trades'] = trade_counts.reindex(ohlcv.index, fill_value=0)
        
        return ohlcv
    
    def _resample_to_timeframe(self, df_1min: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample 1-minute data to specified timeframe
        Matches exact resampling from main branch
        """
        df_resampled = df_1min.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min', 
            'close': 'last',
            'volume': 'sum',
            'trades': 'sum',
            'timestamp': 'last'
        }).dropna()
        
        return df_resampled
    
    def _prepare_strategy_data(self):
        """
        Add fields required by AdaptiveStrategyCore
        """
        # Add source field (all historical for backtesting)
        self.ohlcv_primary_df['source'] = 'historical'
        
        # Ensure timestamp column exists
        if 'timestamp' not in self.ohlcv_primary_df.columns:
            self.ohlcv_primary_df['timestamp'] = self.ohlcv_primary_df.index.astype(int) // 10**9
            
        # Also prepare secondary timeframe if it exists
        if self.ohlcv_secondary_df is not None:
            self.ohlcv_secondary_df['source'] = 'historical'
            if 'timestamp' not in self.ohlcv_secondary_df.columns:
                self.ohlcv_secondary_df['timestamp'] = self.ohlcv_secondary_df.index.astype(int) // 10**9
    
    def get_data_for_backtest(self, start_idx: int = 0, end_idx: Optional[int] = None) -> pd.DataFrame:
        """
        Get slice of data for backtesting with proper lookback
        
        Args:
            start_idx: Starting index for backtest
            end_idx: Ending index (None for all data)
            
        Returns:
            DataFrame suitable for strategy.generate_signal()
        """
        if end_idx is None:
            return self.ohlcv_primary_df.iloc[start_idx:].copy()
        else:
            return self.ohlcv_primary_df.iloc[start_idx:end_idx].copy()
    
    def get_price_at_time(self, timestamp: datetime) -> float:
        """Get the close price at a specific time"""
        # Find the most recent candle before or at timestamp
        mask = self.ohlcv_primary_df.index <= timestamp
        if mask.any():
            return self.ohlcv_primary_df.loc[mask, 'close'].iloc[-1]
        else:
            return self.ohlcv_primary_df['close'].iloc[0]
    
    def simulate_tick_data(self, bar: pd.Series, num_ticks: int = 10) -> List[Tuple[datetime, float]]:
        """
        Simulate intra-bar tick data for more realistic execution
        Uses a combination of random walk and mean reversion to high/low
        
        Args:
            bar: OHLCV bar data
            num_ticks: Number of ticks to generate within the bar
            
        Returns:
            List of (timestamp, price) tuples
        """
        open_price = bar['open']
        high_price = bar['high']
        low_price = bar['low']
        close_price = bar['close']
        
        # Generate tick timestamps evenly distributed across the bar
        bar_start = bar.name
        # Get the timeframe duration
        if self.primary_timeframe.endswith('T'):
            minutes = int(self.primary_timeframe[:-1])
            bar_end = bar_start + timedelta(minutes=minutes)
        elif self.primary_timeframe.endswith('H'):
            hours = int(self.primary_timeframe[:-1])
            bar_end = bar_start + timedelta(hours=hours)
        else:
            # Default to 15 minutes
            bar_end = bar_start + timedelta(minutes=15)
        tick_times = pd.date_range(start=bar_start, end=bar_end, periods=num_ticks + 1)[:-1]
        
        # Generate price path that touches high and low
        prices = [open_price]
        
        # Randomly decide when to hit high and low
        high_idx = np.random.randint(1, num_ticks - 1)
        low_idx = np.random.randint(1, num_ticks - 1)
        while low_idx == high_idx:
            low_idx = np.random.randint(1, num_ticks - 1)
        
        # Generate path
        for i in range(1, num_ticks - 1):
            if i == high_idx:
                prices.append(high_price)
            elif i == low_idx:
                prices.append(low_price)
            else:
                # Random walk between open and close with bias toward close
                prev_price = prices[-1]
                progress = i / (num_ticks - 1)
                target = open_price + (close_price - open_price) * progress
                
                # Add some randomness
                volatility = (high_price - low_price) * 0.1
                random_component = np.random.normal(0, volatility)
                
                # Weighted average of target and random walk
                new_price = 0.7 * target + 0.3 * (prev_price + random_component)
                
                # Ensure within high/low bounds
                new_price = max(low_price, min(high_price, new_price))
                prices.append(new_price)
        
        # Ensure last price is close price
        prices.append(close_price)
        
        return list(zip(tick_times, prices))
    
    def get_data_info(self) -> Dict[str, any]:
        """Get information about loaded data"""
        if self.ohlcv_primary_df is None or self.ohlcv_primary_df.empty:
            return {"status": "No data loaded"}
        
        return {
            "raw_trades": len(self.raw_trades_df) if self.raw_trades_df is not None else 0,
            "1min_candles": len(self.ohlcv_1min_df) if self.ohlcv_1min_df is not None else 0,
            f"{self.primary_timeframe}_candles": len(self.ohlcv_primary_df),
            "date_range": {
                "start": str(self.ohlcv_primary_df.index[0]),
                "end": str(self.ohlcv_primary_df.index[-1])
            },
            "price_range": {
                "min": self.ohlcv_primary_df['low'].min(),
                "max": self.ohlcv_primary_df['high'].max()
            },
            "total_volume": self.ohlcv_primary_df['volume'].sum()
        }


class BacktestDataValidator:
    """Validates backtest data integrity"""
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> List[str]:
        """Validate OHLCV data for common issues"""
        issues = []
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check OHLC relationships
        invalid_candles = df[(df['high'] < df['low']) | 
                           (df['high'] < df['open']) | 
                           (df['high'] < df['close']) |
                           (df['low'] > df['open']) |
                           (df['low'] > df['close'])]
        
        if len(invalid_candles) > 0:
            issues.append(f"Found {len(invalid_candles)} candles with invalid OHLC relationships")
        
        # Check for gaps in time series
        if isinstance(df.index, pd.DatetimeIndex):
            time_diffs = df.index.to_series().diff()
            expected_freq = pd.Timedelta('1H')
            gaps = time_diffs[time_diffs > expected_freq * 1.5]
            if len(gaps) > 0:
                issues.append(f"Found {len(gaps)} time gaps larger than expected")
        
        # Check for zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                invalid_prices = df[df[col] <= 0]
                if len(invalid_prices) > 0:
                    issues.append(f"Found {len(invalid_prices)} rows with non-positive {col} prices")
        
        # Check for zero or negative volume
        if 'volume' in df.columns:
            zero_volume = df[df['volume'] < 0]
            if len(zero_volume) > 0:
                issues.append(f"Found {len(zero_volume)} rows with negative volume")
        
        return issues