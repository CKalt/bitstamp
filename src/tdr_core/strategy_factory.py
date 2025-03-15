###############################################################################
# File Path: src/tdr_core/strategy_factory.py
###############################################################################
# NEW FILE: Creates a StrategyFactory that centralizes the instantiation of
# known strategy classes. This allows do_auto_trade to avoid if/elif blocks.
###############################################################################

from tdr_core.strategies import MACrossoverStrategy, RSITradingStrategy
# If you have a BollingerBandsStrategy or MACDStrategy in the same module:
# from tdr_core.strategies import BollingerBandsStrategy, MACDStrategy
# If RAMMStrategy is in src/strategies/ramm_strategy.py, import it here:
from strategies.ramm_strategy import RAMMStrategy

class StrategyFactory:
    """
    A simple factory to map string names (like "MA", "RSI", "RAMM") to actual
    strategy classes. This helps unify the creation logic in shell.py.
    """
    STRATEGY_MAP = {
        "MA": MACrossoverStrategy,
        "RSI": RSITradingStrategy,
        "RAMM": RAMMStrategy,
        # "Bollinger Bands": BollingerBandsStrategy,
        # "MACD": MACDStrategy,
        # "Adaptive_VWMA": AdaptiveVWMAStrategy,  # if you create such a class
    }

    @staticmethod
    def create(strategy_name, data_manager, logger, **kwargs):
        cls = StrategyFactory.STRATEGY_MAP.get(strategy_name)
        if not cls:
            raise ValueError(f"Unsupported or unknown strategy: {strategy_name}")
        return cls(data_manager=data_manager, logger=logger, **kwargs)
