from trading_system import TradingSystem

if __name__ == "__main__":
    file_paths = {
        "bchbtc": "test_bchbtc.log",
        "bchusd": "test_bchusd.log",
        "btcusd": "test_btcusd.log"
    }
    trading_system = TradingSystem(mode="playback", file_paths=file_paths)
    trading_system.start()
