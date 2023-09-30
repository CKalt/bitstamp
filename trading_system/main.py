
from trading_system import TradingSystem

file_paths = {
    "bchbtc": "logs/bchbtc.log",
    "bchusd": "logs/bchusd.log",
    "btcusd": "logs/btcusd.log"
}
trading_system = TradingSystem(mode="playback", file_paths=file_paths)
trading_system.start()
