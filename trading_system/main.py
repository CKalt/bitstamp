import logging
from trading_system import TradingSystem

logging.basicConfig(filename='trading_system.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


file_paths = {
    "bchbtc": "logs/bchbtc.log",
    "bchusd": "logs/bchusd.log",
    "btcusd": "logs/btcusd.log"
}
trading_system = TradingSystem(mode="playback", file_paths=file_paths)
trading_system.start()
