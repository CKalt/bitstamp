#!/usr/bin/env python3
"""
Data persistence layer for fast server restarts
Saves processed historical data to avoid reloading from raw logs
"""

import pandas as pd
import pickle
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataPersistence:
    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_cache_path(self, symbol, data_type="processed"):
        """Get path for cached data file"""
        return os.path.join(self.cache_dir, f"{symbol}_{data_type}.pkl")
    
    def get_metadata_path(self, symbol):
        """Get path for cache metadata"""
        return os.path.join(self.cache_dir, f"{symbol}_metadata.json")
    
    def save_processed_data(self, symbol, df, source_file_info):
        """Save processed dataframe with metadata"""
        try:
            # Save dataframe
            cache_path = self.get_cache_path(symbol)
            df.to_pickle(cache_path)
            
            # Save metadata
            metadata = {
                "symbol": symbol,
                "source_file": source_file_info.get("path", ""),
                "source_size": source_file_info.get("size", 0),
                "source_modified": source_file_info.get("modified", ""),
                "rows": len(df),
                "date_range": {
                    "start": str(df.index.min()),
                    "end": str(df.index.max())
                },
                "cache_created": datetime.now().isoformat(),
                "columns": list(df.columns)
            }
            
            with open(self.get_metadata_path(symbol), 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Saved {len(df)} rows to cache for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            return False
    
    def load_processed_data(self, symbol, source_file_info=None):
        """Load processed data if cache is valid"""
        try:
            cache_path = self.get_cache_path(symbol)
            metadata_path = self.get_metadata_path(symbol)
            
            # Check if cache exists
            if not os.path.exists(cache_path) or not os.path.exists(metadata_path):
                logger.info(f"No cache found for {symbol}")
                return None
                
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Validate cache if source info provided
            if source_file_info:
                if (metadata.get("source_size") != source_file_info.get("size") or
                    metadata.get("source_modified") != source_file_info.get("modified")):
                    logger.info(f"Cache invalidated - source file changed")
                    return None
            
            # Load dataframe
            df = pd.read_pickle(cache_path)
            
            logger.info(f"Loaded {len(df)} rows from cache for {symbol}")
            logger.info(f"Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None
    
    def clear_cache(self, symbol=None):
        """Clear cache for symbol or all symbols"""
        if symbol:
            files = [
                self.get_cache_path(symbol),
                self.get_metadata_path(symbol)
            ]
        else:
            files = [os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir)]
        
        for file_path in files:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed cache file: {file_path}")