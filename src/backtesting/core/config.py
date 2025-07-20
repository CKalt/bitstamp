"""
Configuration system for backtesting with YAML support
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Strategy-specific configuration"""
    name: str
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfigSchema:
    """Complete backtest configuration schema"""
    # General settings
    name: str = "Backtest"
    description: str = ""
    
    # Data settings
    data_source: str = "btcusd.log"
    start_date: Optional[str] = None  # ISO format: "2024-01-01"
    end_date: Optional[str] = None
    
    # Capital and position settings
    initial_btc: float = 0.0
    initial_usd: float = 100000.0
    always_in_market: bool = True
    
    # Fee and cost settings
    fee_percentage: float = 0.0012  # 0.12% Bitstamp fee
    slippage_bps: float = 0.0  # Basis points of slippage
    
    # Trade limits
    max_trades_per_day: int = 5
    max_trades_per_hour: int = 3
    min_trade_gap_minutes: int = 15
    min_btc_trade_size: float = 1e-8
    
    # Risk management
    enable_pivot_protection: bool = True
    enable_trailing_stops: bool = True
    emergency_exit_loss: float = -2000.0
    pivot_buffer: float = 100.0
    
    # Strategy settings
    strategy: StrategyConfig = None
    
    # Adaptive strategy specific
    regime_detection: Dict[str, Any] = field(default_factory=lambda: {
        'lookback_bars': 100,
        'whipsaw_threshold': 0.65,
        'trend_strength_threshold': 0.3,
        'volatility_window': 50,
        'confidence_threshold': 0.6
    })
    
    # Sub-strategies for adaptive strategy
    trending_strategy: Dict[str, Any] = field(default_factory=lambda: {
        'short_window': 10,
        'long_window': 30,
        'confirmation_bars': 2
    })
    
    ranging_strategy: Dict[str, Any] = field(default_factory=lambda: {
        'bb_window': 20,
        'bb_std_dev': 2.0,
        'rsi_window': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'exit_at_opposite_band': True
    })
    
    volatile_strategy: Dict[str, Any] = field(default_factory=lambda: {
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'breakout_threshold': 100
    })
    
    # Pivot protection settings
    pivot_protection: Dict[str, Any] = field(default_factory=lambda: {
        'lookback_hours': 24,
        'buffer_amount': 100.0,
        'sticky_duration_hours': 4,
        'profit_thresholds': [
            {'profit_pct': 5, 'protection_pct': 70},
            {'profit_pct': 10, 'protection_pct': 80},
            {'profit_pct': 15, 'protection_pct': 85},
            {'profit_pct': 20, 'protection_pct': 90}
        ]
    })
    
    # Output settings
    output_dir: str = "backtest_results"
    save_trades: bool = True
    save_signals: bool = True
    save_equity_curve: bool = True
    generate_plots: bool = False
    
    # Optimization settings (for future use)
    optimization: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'metric': 'sharpe_ratio',
        'method': 'grid_search'
    })


class BacktestConfigManager:
    """Manages loading and validation of backtest configurations"""
    
    def __init__(self, config_dir: str = "config/strategies"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
    def load_yaml(self, config_path: str) -> BacktestConfigSchema:
        """Load configuration from YAML file"""
        path = Path(config_path)
        if not path.exists():
            # Check in config directory
            path = self.config_dir / config_path
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Loading configuration from {path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return self._dict_to_config(config_dict)
    
    def load_json(self, config_path: str) -> BacktestConfigSchema:
        """Load configuration from JSON file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return self._dict_to_config(config_dict)
    
    def save_yaml(self, config: BacktestConfigSchema, output_path: str):
        """Save configuration to YAML file"""
        config_dict = self._config_to_dict(config)
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved configuration to {path}")
    
    def create_default_config(self, strategy_type: str = "adaptive") -> BacktestConfigSchema:
        """Create a default configuration for a given strategy type"""
        config = BacktestConfigSchema()
        
        if strategy_type == "adaptive":
            config.strategy = StrategyConfig(
                name="Adaptive Multi-Strategy",
                type="adaptive",
                parameters={}
            )
        elif strategy_type == "ma_crossover":
            config.strategy = StrategyConfig(
                name="MA Crossover",
                type="ma_crossover",
                parameters={
                    "short_window": 10,
                    "long_window": 30,
                    "confirmation_bars": 2
                }
            )
        elif strategy_type == "mean_reversion":
            config.strategy = StrategyConfig(
                name="Mean Reversion",
                type="mean_reversion",
                parameters={
                    "bb_window": 20,
                    "bb_std_dev": 2.0,
                    "rsi_window": 14
                }
            )
        
        return config
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> BacktestConfigSchema:
        """Convert dictionary to configuration object"""
        # Handle strategy configuration
        if 'strategy' in config_dict and isinstance(config_dict['strategy'], dict):
            strategy_dict = config_dict['strategy']
            config_dict['strategy'] = StrategyConfig(
                name=strategy_dict.get('name', 'Unknown'),
                type=strategy_dict.get('type', 'adaptive'),
                parameters=strategy_dict.get('parameters', {})
            )
        
        # Create config object
        config = BacktestConfigSchema()
        
        # Update with provided values
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def _config_to_dict(self, config: BacktestConfigSchema) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        result = {}
        
        for field_name in config.__dataclass_fields__:
            value = getattr(config, field_name)
            
            if field_name == 'strategy' and value is not None:
                result[field_name] = {
                    'name': value.name,
                    'type': value.type,
                    'parameters': value.parameters
                }
            elif value is not None:
                result[field_name] = value
        
        return result
    
    def validate_config(self, config: BacktestConfigSchema) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate capital
        if config.initial_btc < 0 or config.initial_usd < 0:
            issues.append("Initial capital cannot be negative")
        
        if config.initial_btc == 0 and config.initial_usd == 0:
            issues.append("Must have some initial capital (BTC or USD)")
        
        # Validate fees
        if config.fee_percentage < 0 or config.fee_percentage > 0.1:
            issues.append("Fee percentage must be between 0 and 10%")
        
        # Validate trade limits
        if config.max_trades_per_day < 1:
            issues.append("Max trades per day must be at least 1")
        
        if config.max_trades_per_hour < 1:
            issues.append("Max trades per hour must be at least 1")
        
        if config.min_trade_gap_minutes < 0:
            issues.append("Min trade gap cannot be negative")
        
        # Validate strategy
        if config.strategy is None:
            issues.append("Strategy must be specified")
        
        # Validate dates
        if config.start_date and config.end_date:
            try:
                from datetime import datetime
                start = datetime.fromisoformat(config.start_date)
                end = datetime.fromisoformat(config.end_date)
                if start >= end:
                    issues.append("Start date must be before end date")
            except ValueError:
                issues.append("Invalid date format (use ISO format: YYYY-MM-DD)")
        
        return issues


def create_example_configs():
    """Create example configuration files"""
    manager = BacktestConfigManager()
    
    # Example 1: Conservative adaptive strategy
    conservative = manager.create_default_config("adaptive")
    conservative.name = "Conservative Adaptive Strategy"
    conservative.description = "Lower risk adaptive strategy with tighter limits"
    conservative.max_trades_per_day = 3
    conservative.max_trades_per_hour = 2
    conservative.min_trade_gap_minutes = 30
    conservative.regime_detection['confidence_threshold'] = 0.7
    conservative.trending_strategy['confirmation_bars'] = 3
    
    manager.save_yaml(conservative, "config/strategies/adaptive_conservative.yaml")
    
    # Example 2: Aggressive adaptive strategy
    aggressive = manager.create_default_config("adaptive")
    aggressive.name = "Aggressive Adaptive Strategy"
    aggressive.description = "Higher risk adaptive strategy for volatile markets"
    aggressive.max_trades_per_day = 10
    aggressive.max_trades_per_hour = 5
    aggressive.min_trade_gap_minutes = 10
    aggressive.regime_detection['confidence_threshold'] = 0.5
    aggressive.trending_strategy['confirmation_bars'] = 1
    aggressive.volatile_strategy['breakout_threshold'] = 50
    
    manager.save_yaml(aggressive, "config/strategies/adaptive_aggressive.yaml")
    
    # Example 3: Simple MA crossover
    ma_cross = manager.create_default_config("ma_crossover")
    ma_cross.name = "Classic MA Crossover"
    ma_cross.description = "Simple moving average crossover strategy"
    ma_cross.enable_pivot_protection = False
    ma_cross.enable_trailing_stops = False
    
    manager.save_yaml(ma_cross, "config/strategies/ma_crossover_simple.yaml")
    
    logger.info("Created example configuration files")


if __name__ == "__main__":
    # Create example configs when run directly
    create_example_configs()